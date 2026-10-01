import math

import torch
import torch.nn.functional as F

from verl.trainer.ppo.ovcsd_utils import (
    OVCSDTrajectory,
    PrefixTree,
    action_span_token_mask,
    canonicalize,
    compute_suffix_row_weights,
    find_first_divergence,
    is_all_fail_group,
    local_contrast_advantages,
    topk_tail_kl,
)


def trajectory(name, anchors, actions):
    return OVCSDTrajectory(name, "group", 0.0, name, list(range(len(actions))), anchors, actions)


def test_canonicalize():
    assert canonicalize("  Foo\n\t BAR  ") == "foo bar"


def test_all_fail_group():
    assert is_all_fail_group([0, 0, 0], 10, 1e-6)
    assert not is_all_fail_group([0, 10, 0], 10, 1e-6)
    assert is_all_fail_group([0], 10, 1e-6)


def test_prefix_tree_shared_and_fallback():
    trajectories = [
        trajectory("a", ["s0", "s1", "s2", "a3"], ["x", "y", "a", "q"]),
        trajectory("b", ["s0", "s1", "s2", "b3"], ["x", "y", "b", "q"]),
        trajectory("c", ["s0", "c1"], ["z", "q"]),
    ]
    tree = PrefixTree(trajectories)
    eligible, fallback = tree.eligible_nodes(2, 4)
    assert not fallback
    assert tree.max_shared_depth(2) == 2
    deepest = tree.start_node(eligible, {0, 1, 2})
    assert tree.nodes[deepest].depth == 2
    assert tree.nodes[deepest].members == [0, 1]
    ancestor = tree.nearest_eligible_ancestor(deepest, [tree.path[0][0]])
    assert ancestor == tree.path[0][0]

    fallback_tree = PrefixTree(
        [
            trajectory("a", ["s0", "a1", "a2"], ["x", "a", "q"]),
            trajectory("b", ["s0", "b1", "b2"], ["y", "b", "q"]),
        ]
    )
    eligible, fallback = fallback_tree.eligible_nodes(2, 1)
    assert fallback
    assert {fallback_tree.nodes[key].depth for key in eligible} == {0, 1}


def test_first_divergence():
    assert find_first_divergence(["s1", "s2"], ["a", "x"], ["s0", "s1", "s2"], ["z", "a", "b"], 1) == 1
    assert find_first_divergence(["bad"], ["x"], ["s0"], ["y"], 0) is None
    assert find_first_divergence(["s1"], ["x"], ["s0"], ["y"], 1) is None
    assert find_first_divergence(["s0"], ["y"], ["s0"], ["y"], 0) is None


def test_local_contrast_advantages():
    teacher, student = local_contrast_advantages(4)
    assert teacher == 2
    assert student == -0.5
    assert math.isclose(teacher + 4 * student, 0)


def test_suffix_weights():
    weights = compute_suffix_row_weights([[("a", 2), ("shared", 1)], [("shared", 1)]])
    assert math.isclose(weights["a"] * 2 + weights["shared"], 1.0)
    assert weights["shared"] > weights["a"]
    assert compute_suffix_row_weights([[]]) == {}


def test_topk_tail_kl_and_gradients():
    teacher_logits = torch.tensor([[0.3, -0.1, 0.7]], requires_grad=True)
    teacher = teacher_logits.log_softmax(-1)
    student_logits = teacher_logits.detach().clone().requires_grad_(True)
    student = student_logits.log_softmax(-1)
    equal_kl = topk_tail_kl(student, teacher)
    assert torch.allclose(equal_kl, torch.zeros_like(equal_kl), atol=1e-6)

    student_logits = torch.tensor([[0.8, -0.4, 0.2]], requires_grad=True)
    student = student_logits.log_softmax(-1)
    teacher_logits = torch.tensor([[0.1, 0.3, -0.2]], requires_grad=True)
    teacher = teacher_logits.log_softmax(-1)
    actual = topk_tail_kl(student, teacher)
    expected = F.kl_div(student, teacher.detach().exp(), reduction="none").sum(-1)
    assert torch.allclose(actual, expected, atol=2e-6)
    assert actual.item() >= -1e-6
    actual.sum().backward()
    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_action_span_mask():
    tokens = ["prefix ", "<act", "ion>", "go", "</action>", " tail"]
    assert action_span_token_mask(tokens) == [0, 1, 1, 1, 1, 0]
    assert action_span_token_mask(["plain", " text"]) == [1, 1]
