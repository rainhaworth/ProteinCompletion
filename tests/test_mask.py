from pathlib import Path

import numpy as np
import torch

from utils.config import BaseConfig
from utils.mask import idx_to_mask_targets_hanoi
from utils.model_bidirectional import BidirectionalCausalLM


def transitive_closure(mask):
    reach = mask.astype(bool) | np.eye(mask.shape[0], dtype=bool)
    for intermediate in range(mask.shape[0]):
        reach |= reach[:, intermediate, None] & reach[None, intermediate, :]
    return reach


def test_hanoi_mask_matches_for_list_and_tensor_indices_and_retains_anchors():
    known = [3, 6]

    list_mask, list_targets = idx_to_mask_targets_hanoi(known, 10, 10)
    tensor_mask, tensor_targets = idx_to_mask_targets_hanoi(
        torch.tensor(known), 10, 10
    )

    assert np.array_equal(tensor_mask, list_mask)
    assert np.array_equal(tensor_targets, list_targets)

    generated = np.unique(tensor_targets[tensor_targets >= 0])
    assert np.all(tensor_mask[np.ix_(generated, known)] == 1)


def test_hanoi_masks_are_transitively_safe_and_cover_every_hidden_position():
    for seq_len in range(1, 11):
        for known_bits in range(1, 1 << seq_len):
            known = [
                i for i in range(seq_len) if known_bits & (1 << i)
            ]
            mask, targets = idx_to_mask_targets_hanoi(
                torch.tensor(known), seq_len, seq_len
            )
            reach = transitive_closure(mask)

            target_values = []
            for predictor in range(seq_len):
                for direction in range(2):
                    target = int(targets[predictor, direction])
                    if target < 0:
                        continue

                    target_values.append(target)
                    expected_offset = -1 if direction == 0 else 1
                    assert target == predictor + expected_offset
                    assert mask[predictor, target] == 0
                    assert not reach[predictor, target]

            hidden = set(range(seq_len)) - set(known)
            assert set(target_values) == hidden

            for generated in hidden:
                assert np.all(mask[generated, known] == 1)

            for query, key in zip(*np.nonzero(mask)):
                assert np.all(mask[key] <= mask[query])


def test_hanoi_predictors_are_invariant_to_their_hidden_targets():
    torch.manual_seed(0)
    config = BaseConfig(
        vocab_size=32,
        n_positions=10,
        n_ctx=10,
        n_embd=64,
        n_layer=4,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = BidirectionalCausalLM(config).eval()
    seq = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    mask, targets = idx_to_mask_targets_hanoi(torch.tensor([3, 6]), 10, 10)
    attention_mask = torch.tensor(mask)[None, :, :]

    with torch.inference_mode():
        base_logits = model(seq, attention_mask=attention_mask)

        for predictor in range(seq.size(1)):
            for direction in range(2):
                target = int(targets[predictor, direction])
                if target < 0:
                    continue

                changed = seq.clone()
                changed[0, target] = 31
                changed_logits = model(changed, attention_mask=attention_mask)
                head = slice(0, 32) if direction == 0 else slice(32, 64)

                assert torch.equal(
                    base_logits[0, predictor, head],
                    changed_logits[0, predictor, head],
                )


def test_completion_evaluator_is_valid_python():
    evaluator = Path(__file__).parents[1] / "eval-completion.py"
    compile(evaluator.read_text(encoding="utf-8"), str(evaluator), "exec")
