"""Outcome-Verified Comparative Self-Distillation trainer."""

import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from agent_system.multi_turn_rollout import adjust_batch
from agent_system.multi_turn_rollout.ovcsd_branch import InterventionAttempt, TeacherInterventionRunner
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ovcsd_utils import (
    OVCSDTrajectory,
    PrefixTree,
    action_span_token_mask,
    canonicalize,
    compute_suffix_row_weights,
    find_first_divergence,
    is_all_fail_group,
    local_contrast_advantages,
)
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask


class OVCSDRayTrainer(RLSDRayTrainer):
    """Augment all-failure groups with verified teacher branches."""

    def __init__(self, *args, skill_provider: SkillProvider = None, branch_pool=None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        self.ovcsd = self.config.algorithm.get("ovcsd", {})
        self.branch_pool = branch_pool
        self.skill_provider = skill_provider
        self.intervention_runner = None
        assert not self.config.algorithm.filter_groups.enable
        assert self.config.actor_rollout_ref.actor.loss_agg_mode in {"token-mean", "seq-mean-token-sum"}

    def _collect_groups(self, batch):
        by_traj = defaultdict(list)
        for row, key in enumerate(batch.non_tensor_batch["traj_uid"]):
            by_traj[str(key)].append(row)
        groups = defaultdict(list)
        for traj_uid, rows in by_traj.items():
            rows.sort(key=lambda i: int(batch.non_tensor_batch["turn_step"][i]))
            first = rows[0]
            traj = OVCSDTrajectory(
                traj_uid=traj_uid,
                uid=str(batch.non_tensor_batch["uid"][first]),
                reward=float(batch.non_tensor_batch["episode_rewards"][first]),
                restore_key=batch.non_tensor_batch["ovcsd_restore_key"][first],
                row_idx=rows,
                anchors=[str(batch.non_tensor_batch["anchor_obs"][i]) for i in rows],
                actions=[str(batch.non_tensor_batch["ovcsd_env_action"][i]) for i in rows],
            )
            groups[traj.uid].append(traj)
        return groups

    def _intervene(self, groups):
        if self.intervention_runner is None:
            self.intervention_runner = TeacherInterventionRunner(self.config, self.tokenizer, self.actor_rollout_wg, self.skill_provider, self.branch_pool)
        fail = {uid: ts for uid, ts in groups.items() if is_all_fail_group([t.reward for t in ts], self.ovcsd.get("r_succ", 10.0), self.ovcsd.get("eps_r", 1e-6))}
        num_all_fail = len(fail)
        limit = self.ovcsd.get("max_intervene_groups", -1)
        if limit > 0:
            fail = dict(list(fail.items())[:limit])
        active, verified, stats = {}, [], defaultdict(float)
        stats["num_all_fail_groups"] = num_all_fail
        for uid, trajectories in fail.items():
            usable = [t for t in trajectories if t.restore_key is not None and t.actions]
            if not usable:
                continue
            tree = PrefixTree(usable)
            eligible, fallback = tree.eligible_nodes(self.ovcsd.get("min_support", 2), self.ovcsd.get("fallback_max_depth", 4))
            node = tree.start_node(eligible, range(len(usable)))
            if node is None:
                continue
            active[uid] = [tree, eligible, set(range(len(usable))), node, 0]
            stats["fallback_groups"] += fallback
            stats["avail_shared_depth"] += tree.max_shared_depth(self.ovcsd.get("min_support", 2))
            stats["mean_start_depth"] += tree.nodes[node].depth
        stats["num_intervened_groups"] = len(active)
        while active:
            attempts, owners = [], []
            for uid, (tree, _eligible, _uncovered, node, _tried) in active.items():
                rep = tree.trajectories[tree.representative(node)]
                depth = tree.nodes[node].depth
                for _ in range(self.ovcsd.get("num_teacher_continuations", 2)):
                    attempts.append(InterventionAttempt(uid, node, depth, rep.restore_key, rep.actions[:depth], rep.canon_anchors[: depth + 1]))
                    owners.append(uid)
            attempts = self.intervention_runner.run(attempts)
            stats["teacher_attempts"] += len(attempts)
            next_active = {}
            for uid, (tree, eligible, uncovered, node, tried) in active.items():
                current = [a for a, owner in zip(attempts, owners) if owner == uid]
                stats["restore_failures"] += sum(not a.restore_ok for a in current)
                stats["replay_env_steps"] += sum(a.replay_env_steps for a in current)
                stats["teacher_env_steps"] += sum(a.teacher_env_steps for a in current)
                stats["teacher_prompt_truncated"] += sum(a.teacher_prompt_truncated for a in current)
                stats["teacher_actions"] += sum(a.teacher_actions for a in current)
                stats["teacher_valid_actions"] += sum(a.teacher_valid_actions for a in current)
                stats["teacher_clipped_responses"] += sum(a.teacher_clipped_responses for a in current)
                stats["teacher_specific_skill_attempts"] += sum(a.has_specific_skill for a in current)
                stats["teacher_terminated_without_success"] += sum(a.terminated_without_success for a in current)
                stats["teacher_budget_exhausted"] += sum(a.budget_exhausted for a in current)
                winner = next((a for a in current if a.success), None)
                tried += 1
                if winner:
                    covered = uncovered.intersection(tree.nodes[node].members)
                    verified.append((uid, tree, winner, covered))
                    uncovered -= covered
                    stats["verified_attempts"] += 1
                    stats["verified_branches"] += len(covered)
                    nxt = tree.start_node(eligible, uncovered) if (self.ovcsd.get("cover_all_branches", False) and uncovered) else None
                else:
                    nxt = tree.nearest_eligible_ancestor(node, eligible)
                if nxt is not None and tried < self.ovcsd.get("max_nodes_per_group", 3):
                    next_active[uid] = [tree, eligible, uncovered, nxt, tried]
            active = next_active
        stats["groups_with_verified_branch"] = len({v[0] for v in verified})
        stats["verification_reject_rate"] = 1 - stats["verified_attempts"] / max(1, stats["teacher_attempts"])
        stats["teacher_action_valid_rate"] = stats["teacher_valid_actions"] / max(1, stats["teacher_actions"])
        stats["teacher_response_clip_rate"] = stats["teacher_clipped_responses"] / max(1, stats["teacher_actions"])
        stats["teacher_specific_skill_rate"] = stats["teacher_specific_skill_attempts"] / max(1, stats["teacher_attempts"])
        denom = max(1, stats["num_intervened_groups"])
        stats["avail_shared_depth"] /= denom
        stats["mean_start_depth"] /= denom
        return fail, verified, {f"ovcsd/{k}": v for k, v in stats.items()}

    def _scope_mask(self, response, attention):
        mask = attention.float().cpu()
        if self.ovcsd.get("token_scope", "response") == "action":
            pieces = [self.tokenizer.decode([int(token)]) for token in response]
            mask *= torch.tensor(action_span_token_mask(pieces))
        return mask

    def _prompt_tokens_match(self, original_prompt, rebuilt_prompt):
        """Compare prompt content while ignoring left-padding differences."""
        pad_token_id = self.tokenizer.pad_token_id
        original_tokens = original_prompt[original_prompt != pad_token_id]
        rebuilt_tokens = rebuilt_prompt[rebuilt_prompt != pad_token_id]
        return torch.equal(original_tokens, rebuilt_tokens)

    def _new_row(self, batch, template, prompt_text, step, role):
        row = batch.select_idxs([template])
        original_prompt = row.batch["prompts"].clone()
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False, **self.config.data.get("apply_chat_template_kwargs", {}))
        prompt, prompt_mask = verl_F.tokenize_and_postprocess_data(rendered, self.tokenizer, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True, truncation="left")
        response, response_mask = step.response_ids[None], step.response_attention[None]
        attention = torch.cat([prompt_mask, response_mask], -1)
        row.batch["prompts"], row.batch["responses"] = prompt, response
        row.batch["input_ids"] = torch.cat([prompt, response], -1)
        row.batch["attention_mask"] = attention
        row.batch["position_ids"] = compute_position_id_with_mask(attention)
        uid = f"ovcsd-{uuid.uuid4()}"
        row.non_tensor_batch["uid"] = np.array([uid], dtype=object)
        row.non_tensor_batch["traj_uid"] = np.array([uid], dtype=object)
        row.non_tensor_batch["episode_rewards"] = np.array([0.0], dtype=object)
        row.non_tensor_batch["is_action_valid"] = np.array([True], dtype=bool)
        row.non_tensor_batch["ovcsd_role"] = np.array([role], dtype=object)
        prompt_match = self._prompt_tokens_match(original_prompt[0], prompt[0])
        return row, self._scope_mask(response[0], response_mask[0]), prompt_match

    def _augment(self, batch):
        original_n, length = len(batch), batch.batch["responses"].shape[-1]
        defaults = {"ovcsd_role": "student", "ovcsd_adv": 0.0, "ovcsd_zero_adv": False, "ovcsd_teacher_prompt": None}
        for key, value in defaults.items():
            batch.non_tensor_batch[key] = np.array([value] * original_n, dtype=object)
        groups = self._collect_groups(batch)
        fail, verified, metrics = self._intervene(groups)
        if self.ovcsd.get("zero_fail_group_adv", True):
            batch.non_tensor_batch["ovcsd_zero_adv"] = np.array([str(uid) in fail for uid in batch.non_tensor_batch["uid"]], dtype=object)
        extras, masks, advs, pg, sites, suffix_cache = [], {}, {}, {}, [], {}
        pairs = discarded = divergences = prompt_matches = prompt_comparisons = 0
        for verified_id, (_uid, tree, attempt, covered) in enumerate(verified):
            t_anchors = [step.anchor for step in attempt.steps]
            t_actions = [canonicalize(step.env_action) for step in attempt.steps]
            by_divergence = defaultdict(list)
            for member in covered:
                traj = tree.trajectories[member]
                j = find_first_divergence(t_anchors, t_actions, traj.canon_anchors, traj.canon_actions, attempt.depth)
                pairs += 1
                if j is None:
                    discarded += 1
                else:
                    by_divergence[j].append(traj)
            for j, trajectories in by_divergence.items():
                template = trajectories[0].row_idx[attempt.depth + j]
                row, mask, prompt_match = self._new_row(batch, template, attempt.steps[j].student_prompt_text, attempt.steps[j], "teacher_div")
                prompt_matches += int(prompt_match)
                prompt_comparisons += 1
                if not prompt_match:
                    # Comparative advantages are only valid when teacher and
                    # student actions originate from the exact same prompt.
                    discarded += len(trajectories)
                    continue
                divergences += 1
                teacher_adv, student_adv = local_contrast_advantages(len(trajectories))
                for traj in trajectories:
                    index = traj.row_idx[attempt.depth + j]
                    masks[index] = self._scope_mask(batch.batch["responses"][index], batch.batch["attention_mask"][index, -length:])
                    advs[index] = student_adv
                    batch.non_tensor_batch["ovcsd_role"][index] = "student_div"
                    batch.non_tensor_batch["ovcsd_adv"][index] = student_adv
                row.non_tensor_batch["ovcsd_adv"] = np.array([teacher_adv], dtype=object)
                row.non_tensor_batch["ovcsd_zero_adv"] = np.array([True], dtype=bool)
                index = original_n + len(extras)
                masks[index], advs[index], pg[index] = mask, teacher_adv, 1.0
                extras.append(row)
                site = []
                for js in range(j + 1, len(attempt.steps)):
                    cache_key = (verified_id, js)
                    if cache_key not in suffix_cache:
                        suffix, suffix_mask, _ = self._new_row(batch, template, attempt.steps[js].student_prompt_text, attempt.steps[js], "suffix")
                        suffix.non_tensor_batch["ovcsd_teacher_prompt"] = np.array([attempt.steps[js].teacher_prompt_text], dtype=object)
                        suffix.non_tensor_batch["ovcsd_zero_adv"] = np.array([True], dtype=bool)
                        suffix_index = original_n + len(extras)
                        suffix_cache[cache_key] = (suffix_index, suffix_mask)
                        masks[suffix_index], pg[suffix_index] = suffix_mask, 0.0
                        extras.append(suffix)
                    suffix_index, suffix_mask = suffix_cache[cache_key]
                    site.append((suffix_index, int(suffix_mask.sum())))
                sites.append(site)
        if extras:
            batch = DataProto.concat([batch] + extras)
        total = len(batch)
        adv_mask, suffix_weight = torch.zeros(total, length), torch.zeros(total, length)
        pg_mask = torch.ones(total)
        for index, mask in masks.items():
            adv_mask[index] = mask
        for index, value in pg.items():
            pg_mask[index] = value
        for index, value in compute_suffix_row_weights(sites).items():
            suffix_weight[index] = masks[index] * value
        batch.batch["ovcsd_adv_mask"] = adv_mask
        batch.batch["ovcsd_pg_mask"] = pg_mask
        batch.batch["ovcsd_suffix_weight"] = suffix_weight
        metrics.update(
            {
                "ovcsd/pairs": pairs,
                "ovcsd/discarded_pairs": discarded,
                "ovcsd/divergence_sites": divergences,
                "ovcsd/suffix_sites": sum(bool(s) for s in sites),
                "ovcsd/suffix_rows": len(suffix_cache),
                "ovcsd/suffix_tokens": int((suffix_weight > 0).sum()),
                "ovcsd/teacher_prompt_match_rate": prompt_matches / max(1, prompt_comparisons),
                "ovcsd/privileged_interaction_overhead": (metrics.get("ovcsd/replay_env_steps", 0) + metrics.get("ovcsd/teacher_env_steps", 0)) / max(1, original_n),
            }
        )
        return batch, metrics

    def _apply_ovcsd_advantages(self, batch):
        advantages = batch.batch["advantages"]
        zero = torch.as_tensor(batch.non_tensor_batch["ovcsd_zero_adv"].astype(bool), device=advantages.device)
        advantages[zero] = 0
        has = batch.batch["ovcsd_adv_mask"].sum(-1) > 0
        values = torch.as_tensor(
            batch.non_tensor_batch["ovcsd_adv"].astype(float),
            device=advantages.device,
            dtype=advantages.dtype,
        )
        advantages[has] = values[has, None] * batch.batch["ovcsd_adv_mask"][has].to(advantages.device)
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages
        return batch

    def _teacher_topk(self, batch):
        """Evaluate suffix actions under the frozen skill-conditioned teacher."""
        selected = batch.batch["ovcsd_suffix_weight"].sum(-1) > 0
        bs, length = batch.batch["responses"].shape
        topk = self.ovcsd.get("topk", 16)
        all_ids = torch.zeros(bs, length, topk, dtype=torch.int32)
        all_logprobs = torch.zeros(bs, length, topk, dtype=torch.float32)
        if not selected.any():
            batch.batch["teacher_topk_ids"] = all_ids
            batch.batch["teacher_topk_logprobs"] = all_logprobs
            return batch
        indices = selected.nonzero().flatten().tolist()
        rows = batch.select_idxs(indices)
        prompts, prompt_masks = [], []
        for index in indices:
            rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": str(batch.non_tensor_batch["ovcsd_teacher_prompt"][index])}], tokenize=False, add_generation_prompt=True, **self.config.data.get("apply_chat_template_kwargs", {}))
            ids, mask = verl_F.tokenize_and_postprocess_data(rendered, self.tokenizer, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True, truncation="left")
            prompts.append(ids)
            prompt_masks.append(mask)
        prompts, prompt_masks = torch.cat(prompts), torch.cat(prompt_masks)
        rows.batch["prompts"] = prompts
        rows.batch["input_ids"] = torch.cat([prompts, rows.batch["responses"]], -1)
        rows.batch["attention_mask"] = torch.cat([prompt_masks, rows.batch["attention_mask"][:, -length:]], -1)
        rows.batch["position_ids"] = compute_position_id_with_mask(rows.batch["attention_mask"])
        rows.meta_info["topk"] = topk
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        padded, pad_size = pad_dataproto_to_divisor(rows, self.actor_rollout_wg.world_size)
        output = unpad_dataproto(self.actor_rollout_wg.compute_teacher_topk(padded), pad_size)
        all_ids[selected] = output.batch["teacher_topk_ids"]
        all_logprobs[selected] = output.batch["teacher_topk_logprobs"]
        batch.batch["teacher_topk_ids"] = all_ids
        batch.batch["teacher_topk_logprobs"] = all_logprobs
        return batch

    def _after_validation(self, step: int, val_metrics: dict) -> dict:
        """Hook for trainers that consume validation results."""

        return {}

    def fit(self):
        """
        Run skill-free PPO with OVCSD interventions before reward evaluation.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch.batch.pop("rollout_log_probs", None)
                    with _timer("ovcsd_intervention", timing_raw):
                        batch, ovcsd_metrics = self._augment(batch)
                        metrics.update(ovcsd_metrics)

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"] * batch.batch["ovcsd_pg_mask"][:, None]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with _timer("ovcsd_teacher_topk", timing_raw):
                        batch = self._teacher_topk(batch)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        batch = self._apply_ovcsd_advantages(batch)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        metrics.update(self._after_validation(self.global_steps, val_metrics))

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
