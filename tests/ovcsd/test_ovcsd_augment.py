import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ovcsd_ray_trainer import OVCSDRayTrainer
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty


def test_apply_ovcsd_advantages_keeps_normal_rows_and_overrides_sites():
    batch = DataProto.from_dict(
        tensors={
            "advantages": torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]),
            "returns": torch.zeros(3, 2),
            "ovcsd_adv_mask": torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        },
        non_tensors={
            "ovcsd_zero_adv": np.array([False, True, True], dtype=object),
            "ovcsd_adv": np.array([0.0, -0.5, 2.0], dtype=object),
        },
    )
    trainer = OVCSDRayTrainer.__new__(OVCSDRayTrainer)
    result = trainer._apply_ovcsd_advantages(batch)
    assert torch.equal(result.batch["advantages"][0], torch.tensor([3.0, 3.0]))
    assert torch.equal(result.batch["advantages"][1], torch.tensor([-0.5, 0.0]))
    assert torch.equal(result.batch["advantages"][2], torch.tensor([2.0, 2.0]))
    assert torch.equal(result.batch["returns"], result.batch["advantages"])
def test_invalid_action_penalty_accepts_scalar_and_array_validity_values():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(3, 2, dtype=torch.long),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "token_level_scores": torch.zeros(3, 2),
        },
        non_tensors={
            "is_action_valid": np.array(
                [True, np.bool_(False), np.array([True], dtype=bool)], dtype=object
            ),
        },
    )
    result, metrics = apply_invalid_action_penalty(batch, invalid_action_penalty_coef=0.25)
    assert torch.equal(
        result.batch["token_level_scores"],
        torch.tensor([[0.0, 0.0], [0.0, -0.25], [0.0, 0.0]]),
    )
    assert np.isclose(metrics["episode/valid_action_ratio"], 2.0 / 3.0)


def test_prompt_match_ignores_padding_but_not_content():
    trainer = OVCSDRayTrainer.__new__(OVCSDRayTrainer)
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

    assert trainer._prompt_tokens_match(
        torch.tensor([0, 0, 11, 12]), torch.tensor([0, 11, 12])
    )
    assert not trainer._prompt_tokens_match(
        torch.tensor([0, 0, 11, 12]), torch.tensor([0, 11, 13])
    )
