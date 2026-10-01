import numpy as np

from agent_system.multi_turn_rollout.ovcsd_branch import (
    WebshopRestoreKey,
    webshop_restore_matches,
)
from verl.utils.dataset.rl_dataset import collate_fn


def test_webshop_restore_key_survives_rollout_collation():
    key = WebshopRestoreKey(123, "Find a red shoe")
    batch = collate_fn([{"restore_key": key}, {"restore_key": key}])

    assert batch["restore_key"].dtype == np.dtype("O")
    assert batch["restore_key"].shape == (2,)
    assert batch["restore_key"][0] == key


def test_webshop_restore_requires_matching_task_and_anchor():
    key = WebshopRestoreKey(123, "Find a red shoe")

    assert webshop_restore_matches(key, "  FIND a red shoe ", " Search ", "search")
    assert not webshop_restore_matches(key, "Find a blue lamp", "Search", "search")
    assert not webshop_restore_matches(key, "Find a red shoe", "Results", "search")
    assert not webshop_restore_matches(123, "Find a red shoe", "Search", "search")
