import itertools

import numpy as np
import torch

from utils.data import PackedUnirefData
from utils.mask import idx_to_mask_targets_hanoi


class TokenizerStub:
    def __init__(self, mask_id=31):
        self.mask_id = mask_id

    def token_to_id(self, token):
        if token == "<mask>":
            return self.mask_id
        return None


def hidden_targets(targets):
    return {int(target) for target in targets.flatten().tolist() if target >= 0}


def test_hanoi_mask_covers_every_hidden_position_exhaustively():
    for length in range(1, 11):
        positions = range(length)
        for motif_size in range(1, length + 1):
            for motif in itertools.combinations(positions, motif_size):
                motif_tensor = torch.tensor(motif)
                _, targets = idx_to_mask_targets_hanoi(motif_tensor, length)
                hidden = set(positions) - set(motif)
                assert hidden <= hidden_targets(targets), (length, motif, targets)


def test_hanoi_mask_remains_transitively_leakage_free():
    for length in range(1, 9):
        positions = range(length)
        for motif_size in range(1, length + 1):
            for motif in itertools.combinations(positions, motif_size):
                mask, targets = idx_to_mask_targets_hanoi(torch.tensor(motif), length)
                reachable = mask.bool() | torch.eye(length, dtype=torch.bool)
                for intermediate in range(length):
                    reachable |= (
                        reachable[:, intermediate, None]
                        & reachable[intermediate, None, :]
                    )

                for predictor, predictor_targets in enumerate(targets):
                    for target in predictor_targets.tolist():
                        if target >= 0:
                            assert not reachable[predictor, target], (
                                length,
                                motif,
                                predictor,
                                target,
                            )


def test_hanoi_mask_keeps_two_sided_targets_when_fronts_meet():
    _, targets = idx_to_mask_targets_hanoi(torch.tensor([0, 2]), 3)

    assert targets[0, 1] == 1
    assert targets[2, 0] == 1


def test_packed_esm_uses_mask_id_from_tokenizer(tmp_path):
    data_path = tmp_path / "packed.bin"
    packed = np.memmap(data_path, mode="w+", dtype=np.float64, shape=(8,))
    packed[:] = np.array([1, 4, 5, 2, 3, 1, 6, 2])
    packed.flush()
    del packed

    dataset = PackedUnirefData(
        str(data_path),
        tokenizer=TokenizerStub(mask_id=31),
        max_dim=8,
        model_type="esm",
    )
    sequence, targets, _ = dataset[0]
    masked_positions = targets >= 0

    assert torch.any(masked_positions)
    assert torch.all(sequence[masked_positions] == 31)
    assert not torch.any(sequence[masked_positions] == 3)


def test_packed_esm_rejects_missing_or_conflicting_mask_ids(tmp_path):
    data_path = tmp_path / "packed.bin"
    data_path.write_bytes(bytes(64))

    for tokenizer in (None, TokenizerStub(mask_id=3)):
        try:
            PackedUnirefData(
                str(data_path),
                tokenizer=tokenizer,
                max_dim=8,
                model_type="esm",
            )
        except ValueError as exc:
            assert "mask" in str(exc).lower() or "tokenizer" in str(exc).lower()
        else:
            raise AssertionError("Invalid ESM tokenizer was accepted")
