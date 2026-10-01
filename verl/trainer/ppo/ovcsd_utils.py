"""Pure helpers for Outcome-Verified Comparative Self-Distillation (OVCSD)."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import torch


def canonicalize(text: Any) -> str:
    """Normalize observations/actions before deterministic replay comparisons."""

    return " ".join(str(text).lower().split())


@dataclass
class OVCSDTrajectory:
    traj_uid: str
    uid: str
    reward: float
    restore_key: Any
    row_idx: List[int]
    anchors: List[str]
    actions: List[str]
    canon_anchors: List[str] = field(init=False)
    canon_actions: List[str] = field(init=False)

    def __post_init__(self) -> None:
        if not (len(self.row_idx) == len(self.anchors) == len(self.actions)):
            raise ValueError("row_idx, anchors, and actions must have equal length")
        self.canon_anchors = [canonicalize(value) for value in self.anchors]
        self.canon_actions = [canonicalize(value) for value in self.actions]


def is_all_fail_group(rewards: Sequence[float], r_succ: float, eps_r: float) -> bool:
    """Return Eq. (3)'s all-failure trigger (sample std; singleton std is zero)."""

    values = torch.as_tensor(list(rewards), dtype=torch.float64)
    if values.numel() == 0:
        return False
    std = 0.0 if values.numel() == 1 else float(values.std(unbiased=True))
    return float(values.max()) < r_succ and std <= eps_r


def _node_hash(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass
class PrefixNode:
    key: str
    depth: int
    parent: Optional[str]
    members: List[int] = field(default_factory=list)


class PrefixTree:
    """Prefix tree over state-aligned student trajectories."""

    def __init__(self, trajectories: Sequence[OVCSDTrajectory]):
        self.trajectories = list(trajectories)
        self.nodes: Dict[str, PrefixNode] = {}
        self.path: List[List[str]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            traj_path: List[str] = []
            parent: Optional[str] = None
            for depth, anchor in enumerate(traj.canon_anchors):
                if depth >= len(traj.canon_actions):
                    break
                if depth == 0:
                    key = _node_hash("root", anchor)
                else:
                    key = _node_hash(parent or "", traj.canon_actions[depth - 1], anchor)
                node = self.nodes.setdefault(key, PrefixNode(key, depth, parent))
                if node.depth != depth or node.parent != parent:
                    raise ValueError("SHA-1 collision while building OVCSD prefix tree")
                node.members.append(traj_idx)
                traj_path.append(key)
                parent = key
            self.path.append(traj_path)

    def shared_nodes(self, min_support: int = 2) -> List[str]:
        return [key for key, node in self.nodes.items() if len(node.members) >= min_support]

    def eligible_nodes(self, min_support: int, fallback_max_depth: int) -> Tuple[List[str], bool]:
        eligible = self.shared_nodes(min_support)
        has_non_root_shared = any(self.nodes[key].depth > 0 for key in eligible)
        used_fallback = not has_non_root_shared
        if used_fallback:
            for key, node in self.nodes.items():
                if 1 <= node.depth <= fallback_max_depth and key not in eligible:
                    eligible.append(key)
        return eligible, used_fallback

    def start_node(self, eligible: Iterable[str], uncovered: Iterable[int]) -> Optional[str]:
        uncovered_set = set(uncovered)
        candidates = [key for key in eligible if set(self.nodes[key].members).intersection(uncovered_set)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda key: (
                self.nodes[key].depth,
                len(set(self.nodes[key].members).intersection(uncovered_set)),
                key,
            ),
        )

    def nearest_eligible_ancestor(self, node: str, eligible: Iterable[str]) -> Optional[str]:
        eligible_set = set(eligible)
        parent = self.nodes[node].parent
        while parent is not None:
            if parent in eligible_set:
                return parent
            parent = self.nodes[parent].parent
        return None

    def representative(self, node: str) -> int:
        return self.nodes[node].members[0]

    def max_shared_depth(self, min_support: int = 2) -> int:
        return max((self.nodes[key].depth for key in self.shared_nodes(min_support)), default=0)


def find_first_divergence(
    t_anchors: Sequence[str],
    t_actions: Sequence[str],
    s_anchors: Sequence[str],
    s_actions: Sequence[str],
    start_depth: int,
) -> Optional[int]:
    """Find the first action divergence while teacher/student states stay aligned."""

    for j, (teacher_anchor, teacher_action) in enumerate(zip(t_anchors, t_actions)):
        student_depth = start_depth + j
        if student_depth >= len(s_actions) or student_depth >= len(s_anchors):
            return None
        if teacher_anchor != s_anchors[student_depth]:
            return None
        if teacher_action != s_actions[student_depth]:
            return j
    return None


def local_contrast_advantages(m: int) -> Tuple[float, float]:
    if m <= 0:
        raise ValueError("m must be positive")
    root = math.sqrt(m)
    return root, -1.0 / root


def compute_suffix_row_weights(
    sites: Sequence[Sequence[Tuple[Hashable, int]]],
) -> Dict[Hashable, float]:
    """Compute the per-token coefficient for each suffix row."""

    valid_sites = [site for site in sites if sum(max(0, n) for _, n in site) > 0]
    if not valid_sites:
        return {}
    result: Dict[Hashable, float] = {}
    for site in valid_sites:
        n_tokens = sum(max(0, n) for _, n in site)
        coefficient = 1.0 / (len(valid_sites) * n_tokens)
        for row_key, count in site:
            if count > 0:
                result[row_key] = result.get(row_key, 0.0) + coefficient
    return result


def topk_tail_kl(
    student_logp_k: torch.Tensor,
    teacher_logp_k: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL(q||p) over teacher top-k tokens plus one residual-mass bucket."""

    q_logp = teacher_logp_k.detach()
    q = q_logp.exp()
    p = student_logp_k.exp()
    head = (q * (q_logp - student_logp_k)).sum(dim=-1)
    q_res = (1.0 - q.sum(dim=-1)).clamp_min(eps)
    p_res = (1.0 - p.sum(dim=-1)).clamp_min(eps)
    return head + q_res * (q_res.log() - p_res.log())


_ACTION_RE = re.compile(r"<action>.*?</action>", flags=re.IGNORECASE | re.DOTALL)


def action_span_token_mask(token_texts: Sequence[str]) -> List[int]:
    """Mask tokens overlapping the final ``<action>...</action>`` span."""

    text = "".join(token_texts)
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return [1] * len(token_texts)
    start, end = matches[-1].span()
    mask: List[int] = []
    offset = 0
    for token in token_texts:
        token_end = offset + len(token)
        mask.append(int(token_end > start and offset < end))
        offset = token_end
    return mask
