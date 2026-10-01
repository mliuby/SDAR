"""Replay support and outcome-verified teacher interventions for OVCSD."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, List

import ray
import torch

import verl.utils.torch_functional as verl_F
from agent_system.environments.env_manager import AlfWorldEnvironmentManager, WebshopEnvironmentManager
from agent_system.memory import SimpleMemory
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ovcsd_utils import canonicalize
from verl.utils.model import compute_position_id_with_mask

@dataclass(frozen=True)
class WebshopRestoreKey:
    session_idx: int
    task: str


def webshop_restore_matches(key, task, anchor, expected_anchor):
    return (
        isinstance(key, WebshopRestoreKey)
        and canonicalize(task) == canonicalize(key.task)
        and canonicalize(anchor) == expected_anchor
    )


class OVCSDTrajectoryCollector(TrajectoryCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rec = None
        self._attached_env = None

    def attach_recorder(self, envs):
        assert not self.config.algorithm.filter_groups.enable
        if self._attached_env is envs:
            return
        original_projection, original_reset = envs.projection_f, envs.reset

        def projection(*args, **kwargs):
            projected = original_projection(*args, **kwargs)
            if self._rec is not None:
                self._rec["actions"].append(list(projected[0]))
            return projected

        def reset(*args, **kwargs):
            observations, infos = original_reset(*args, **kwargs)
            keys = [info.get("extra.gamefile", info.get("session_idx")) for info in infos]
            if isinstance(envs, WebshopEnvironmentManager):
                keys = [
                    WebshopRestoreKey(key, envs.tasks[i]) for i, key in enumerate(keys)
                ]
            self._rec = {"actions": [], "keys": keys}
            return observations, infos

        envs.projection_f, envs.reset = projection, reset
        self._attached_env = envs

    def vanilla_multi_turn_loop(self, *args, **kwargs):
        self._rec = None
        return super().vanilla_multi_turn_loop(*args, **kwargs)

    def gather_rollout_data(self, total_batch_list, *args, **kwargs):
        if self._rec is not None:
            for env_idx, trajectory in enumerate(total_batch_list):
                for step_idx, row in enumerate(trajectory):
                    row["ovcsd_env_action"] = self._rec["actions"][step_idx][env_idx]
                    row["ovcsd_restore_key"] = self._rec["keys"][env_idx]
        return super().gather_rollout_data(total_batch_list, *args, **kwargs)


class WebshopEpisode:
    def __init__(self, config):
        self.config = config
        self.manager = WebshopEnvironmentManager.__new__(WebshopEnvironmentManager)
        self.manager.config, self.manager.memory = config, SimpleMemory()
        self.manager.memory.reset(1)
        self.task = ""

    def reset(self, raw, info):
        self.manager.tasks = self.manager.extract_task([raw])
        obs = self.manager.format_obs([raw])
        self.manager.pre_text_obs, self.task = obs, self.manager.tasks[0]
        return self.manager.build_text_obs(obs, [info], init=True)[0], obs[0]

    def step(self, action, raw, info):
        obs = self.manager.format_obs([raw])
        self.manager.memory.store({"text_obs": self.manager.pre_text_obs, "action": [action]})
        self.manager.pre_text_obs = obs
        return self.manager.build_text_obs(obs, [info])[0], obs[0]

    def project(self, text, info):
        from agent_system.environments.env_package.webshop import webshop_projection

        kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        actions, valid = webshop_projection([text], require_think=kwargs.get("enable_thinking", True))
        return actions[0], bool(valid[0])


class AlfworldEpisode:
    def __init__(self, config):
        self.config = config
        self.manager = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
        self.manager.config, self.manager.memory = config, SimpleMemory()
        self.manager.memory.reset(1)
        self.task = ""

    def reset(self, raw, info):
        self.manager.tasks, self.manager.pre_text_obs = [], [raw]
        self.manager.extract_task([raw])
        self.task = self.manager.tasks[0]
        return self.manager.build_text_obs([raw], [info["admissible_commands"]], init=True)[0], raw

    def step(self, action, raw, info):
        self.manager.memory.store({"text_obs": self.manager.pre_text_obs, "action": [action]})
        self.manager.pre_text_obs = [raw]
        return self.manager.build_text_obs([raw], [info["admissible_commands"]])[0], raw

    @staticmethod
    def project(text, info):
        from agent_system.environments.env_package.alfworld import alfworld_projection

        actions, valid = alfworld_projection([text], [info["admissible_commands"]])
        return actions[0], bool(valid[0])


class AlfworldBranchWorker:
    def __init__(self, base_env):
        self.base_env, self.env = base_env, None

    def reset_to(self, gamefile):
        if self.env is not None:
            self.env.close()
        self.base_env.game_files, self.base_env.num_games = [gamefile], 1
        self.env = self.base_env.init_env(batch_size=1)
        obs, infos = self.env.reset()
        return obs[0], {key: value[0] for key, value in infos.items()}

    def step(self, action):
        obs, _, dones, infos = self.env.step([action])
        info = {key: value[0] for key, value in infos.items()}
        return obs[0], 10.0 * float(info["won"]), bool(dones[0]), info


class WebshopBranchWorker:
    def __init__(self, seed, env_kwargs):
        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        self.worker = WebshopWorker(seed, env_kwargs)

    def reset_to(self, restore_key):
        if not isinstance(restore_key, WebshopRestoreKey):
            raise TypeError("WebShop replay requires a WebshopRestoreKey")
        return self.worker.reset(restore_key.session_idx)

    def step(self, action):
        return self.worker.step(action)


class BranchEnvPool:
    def __init__(self, config, size=None):
        self.config = config
        ovcsd = config.algorithm.get("ovcsd", {})
        self.size = int(size or ovcsd.get("branch_pool_size", -1))
        if self.size <= 0:
            self.size = config.data.train_batch_size * ovcsd.get("num_teacher_continuations", 2)
        resources, env_name = dict(config.env.resources_per_worker), config.env.env_name.lower()
        if "webshop" in env_name:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environments/env_package/webshop/webshop/data"))
            suffix = "_1000" if config.env.webshop.use_small else ""
            kwargs = {"observation_mode": "text", "num_products": None, "human_goals": config.env.webshop.human_goals, "file_path": os.path.join(root, f"items_shuffle{suffix}.json"), "attr_path": os.path.join(root, f"items_ins_v2{suffix}.json")}
            actor = ray.remote(**resources)(WebshopBranchWorker)
            self.workers = [actor.remote(config.env.seed, kwargs) for _ in range(self.size)]
            self.episode_cls = WebshopEpisode
        elif "alfworld" in env_name:
            import yaml

            from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environments/env_package/alfworld/configs/config_tw.yaml"))
            with open(path) as handle:
                cfg = yaml.safe_load(handle)
            if cfg.get("env", {}).get("domain_randomization", False):
                warnings.warn("OVCSD replay requires domain_randomization=False", stacklevel=2)
            base = get_environment(cfg["env"]["type"])(cfg, train_eval="train")
            actor = ray.remote(**resources)(AlfworldBranchWorker)
            self.workers = [actor.remote(base) for _ in range(self.size)]
            self.episode_cls = AlfworldEpisode
        else:
            raise NotImplementedError("OVCSD supports WebShop and ALFWorld only")

    def new_episode(self):
        return self.episode_cls(self.config)


@dataclass
class TeacherStep:
    anchor: str
    env_action: str
    student_prompt_text: str
    teacher_prompt_text: str
    response_ids: torch.Tensor
    response_attention: torch.Tensor
    is_valid: bool


@dataclass
class InterventionAttempt:
    group_id: str
    node_key: str
    depth: int
    restore_key: Any
    prefix_actions: List[str]
    expected_anchors: List[str]
    restore_ok: bool = True
    success: bool = False
    reward: float = 0.0
    skill_text: str = ""
    steps: List[TeacherStep] = field(default_factory=list)
    replay_env_steps: int = 0
    teacher_env_steps: int = 0
    teacher_prompt_truncated: int = 0
    teacher_actions: int = 0
    teacher_valid_actions: int = 0
    teacher_clipped_responses: int = 0
    has_specific_skill: bool = False
    terminated_without_success: bool = False
    budget_exhausted: bool = False


class TeacherInterventionRunner:
    def __init__(self, config, tokenizer, actor_rollout_wg, skill_provider, branch_pool):
        self.config, self.tokenizer = config, tokenizer
        self.actor_rollout_wg, self.skill_provider, self.pool = actor_rollout_wg, skill_provider, branch_pool
        self.r_succ = config.algorithm.ovcsd.get("r_succ", 10.0)

    def _encode(self, text, truncation=None):
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, **self.config.data.get("apply_chat_template_kwargs", {}))
        return verl_F.tokenize_and_postprocess_data(rendered, self.tokenizer, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True, truncation=truncation or self.config.data.truncation)

    def _generate(self, encoded):
        ids, attention = (torch.cat([item[i] for item in encoded]) for i in range(2))
        data = DataProto.from_dict(
            tensors={"input_ids": ids, "attention_mask": attention, "position_ids": compute_position_id_with_mask(attention)},
            meta_info={"do_sample": self.config.algorithm.ovcsd.get("teacher_do_sample", True)},
        )
        padded, n = pad_dataproto_to_divisor(data, self.actor_rollout_wg.world_size)
        return unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), n)

    def run(self, attempts):
        for offset in range(0, len(attempts), self.pool.size):
            chunk, workers = attempts[offset : offset + self.pool.size], self.pool.workers
            resets = ray.get([w.reset_to.remote(a.restore_key) for w, a in zip(workers, chunk)])
            states = []
            for attempt, (raw, info) in zip(chunk, resets):
                episode = self.pool.new_episode()
                prompt, anchor = episode.reset(raw, info)
                if isinstance(attempt.restore_key, WebshopRestoreKey):
                    attempt.restore_ok = webshop_restore_matches(
                        attempt.restore_key, episode.task, anchor, attempt.expected_anchors[0]
                    )
                else:
                    attempt.restore_ok = canonicalize(anchor) == attempt.expected_anchors[0]
                states.append([episode, prompt, anchor, info, False])
            for p in range(max((len(a.prefix_actions) for a in chunk), default=0)):
                active = [i for i, a in enumerate(chunk) if a.restore_ok and p < len(a.prefix_actions)]
                results = ray.get([workers[i].step.remote(chunk[i].prefix_actions[p]) for i in active])
                for i, (raw, _, done, info) in zip(active, results):
                    prompt, anchor = states[i][0].step(chunk[i].prefix_actions[p], raw, info)
                    chunk[i].replay_env_steps += 1
                    chunk[i].restore_ok = not done and canonicalize(anchor) == chunk[i].expected_anchors[p + 1]
                    states[i][1:] = [prompt, anchor, info, done]
            for i, attempt in enumerate(chunk):
                if attempt.restore_ok:
                    attempt.skill_text = self.skill_provider.get_privileged_info(attempt.restore_key) if "alfworld" in self.config.env.env_name.lower() else self.skill_provider.get_privileged_info_from_prompt(states[i][0].task)
                    general_skill = self.skill_provider.skill_contents.get("general_skills", "").strip()
                    attempt.has_specific_skill = attempt.skill_text.strip() != general_skill
            for _ in range(self.config.env.max_steps):
                active = [i for i, a in enumerate(chunk) if a.restore_ok and not states[i][4] and a.teacher_env_steps < self.config.env.max_steps - a.depth]
                if not active:
                    break
                encoded, teacher_texts = [], []
                for i in active:
                    teacher = f"[Privileged Skill Information]\n{chunk[i].skill_text}\n\n{states[i][1]}"
                    teacher_texts.append(teacher)
                    try:
                        encoded.append(self._encode(teacher))
                    except RuntimeError:
                        chunk[i].teacher_prompt_truncated += 1
                        encoded.append(self._encode(teacher, "left"))
                output = self._generate(encoded)
                texts = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
                projected = [states[i][0].project(text, states[i][3]) for i, text in zip(active, texts)]
                results = ray.get([workers[i].step.remote(action) for i, (action, _) in zip(active, projected)])
                for pos, (i, (raw, reward, done, info)) in enumerate(zip(active, results)):
                    episode, student, anchor = states[i][:3]
                    response = output.batch["responses"][pos].cpu()
                    mask = output.batch["attention_mask"][pos, -response.shape[0] :].cpu()
                    chunk[i].steps.append(TeacherStep(canonicalize(anchor), projected[pos][0], student, teacher_texts[pos], response, mask, projected[pos][1]))
                    chunk[i].teacher_actions += 1
                    chunk[i].teacher_valid_actions += int(bool(projected[pos][1]))
                    chunk[i].teacher_clipped_responses += int(
                        int(mask.sum()) >= int(self.config.data.max_response_length)
                    )
                    prompt, anchor = episode.step(projected[pos][0], raw, info)
                    chunk[i].teacher_env_steps += 1
                    chunk[i].reward += float(reward)
                    chunk[i].success = chunk[i].reward >= self.r_succ
                    states[i][1:] = [prompt, anchor, info, bool(done) or chunk[i].success]
            for i, attempt in enumerate(chunk):
                if attempt.restore_ok and not attempt.success:
                    attempt.terminated_without_success = bool(states[i][4])
                    attempt.budget_exhausted = not attempt.terminated_without_success
        return attempts
