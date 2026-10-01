# OVCSD 复现说明

本实现复现 *From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents*。学生 rollout、验证和部署均不使用技能；技能只用于训练时从学生已到达状态启动的 teacher 续写，且续写必须在环境中达到成功奖励才进入训练。

| 论文组件 | 实现 |
|---|---|
| Eq. 3 全失败触发 | `is_all_fail_group` |
| 失败轨迹前缀树 | `PrefixTree` |
| Eq. 4–5 回溯与结果验证 | `_intervene`、`TeacherInterventionRunner` |
| Eq. 6 对齐发散 | `find_first_divergence` |
| Eq. 7 局部对比 | `local_contrast_advantages`、`_augment` |
| Eq. 8 后缀蒸馏 | `compute_suffix_row_weights`、`topk_tail_kl` |
| Eq. 9 总目标 | `DataParallelPPOActor.update_policy` |

论文未公开官方实现细节，因此这些设计选择均可配置：每结点 teacher 数、最多回溯数、单轨迹 fallback 深度、是否覆盖全部分支、后缀系数、top-k 和 token 范围。默认每组得到第一条成功分支后停止。

WebShop 用 `session_idx` 恢复，ALFWorld 用 gamefile 恢复；二者均逐状态检查重放一致性。默认配置为：`r_succ=10`、`eps_r=1e-6`、`num_teacher_continuations=2`、`max_nodes_per_group=3`、`min_support=2`、`fallback_max_depth=4`、`cover_all_branches=false`、`zero_fail_group_adv=true`、`token_scope=response`、`suffix_coef=1`、`topk=16`。

```bash
TRAIN_DATA_SIZE=4 GROUP_SIZE=4 bash examples/ovcsd_trainer/run_webshop_qwen3_1b_nas.sh \
  trainer.total_training_steps=3 trainer.val_before_train=False trainer.test_freq=-1

bash examples/ovcsd_trainer/run_alfworld_qwen3.sh
```

与只给学生动作打分的 SDAR 不同，OVCSD 让 teacher 真正执行并只学习验证成功的分支。当前限制：仅 FSDP/FSDP2、SP=1、非动态 batch、非 fused kernel、纯文本模型。`adjust_batch` 可能复制增广行及其权重；分支环境还会额外占用 CPU/内存。
