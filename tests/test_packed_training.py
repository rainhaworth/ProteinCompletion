import itertools

import numpy as np
import torch

from utils.data import PackedUnirefData
from utils.config import BaseConfig
from utils.mask import diag_block_mask, idx_to_mask_targets_hanoi
from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM


class TokenizerStub:
    def __init__(self, mask_id=31):
        self.mask_id = mask_id

    def token_to_id(self, token):
        if token == "<mask>":
            return self.mask_id
        return None


def tiny_config():
    return BaseConfig(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=32,
        n_layer=2,
        n_head=4,
        rotary_dim=8,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )


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


def test_packed_atp_offsets_target_zero_in_later_blocks():
    mask_indices = torch.tensor([0, 3])
    separator_indices = torch.tensor([1, 4])

    _, targets = diag_block_mask(mask_indices, separator_indices, dim=4)

    assert targets[3, 0] == 2


def test_packed_atp_blocks_match_standalone_masks_exhaustively():
    for block_start in (2, 5, 11):
        prefix_length = block_start - 1
        for length in range(1, 8):
            positions = range(length)
            for motif_size in range(1, length + 1):
                for motif in itertools.combinations(positions, motif_size):
                    local_motif = torch.tensor(motif)
                    mask_indices = torch.cat(
                        [torch.tensor([0]), local_motif + block_start]
                    )
                    separator_indices = torch.tensor(
                        [prefix_length, block_start + length]
                    )

                    packed_mask, packed_targets = diag_block_mask(
                        mask_indices,
                        separator_indices,
                        dim=block_start + length,
                    )
                    expected_mask, expected_targets = idx_to_mask_targets_hanoi(
                        local_motif,
                        length,
                    )
                    expected_targets[expected_targets >= 0] += block_start

                    block_slice = slice(block_start, block_start + length)
                    assert torch.equal(
                        packed_mask[block_slice, block_slice], expected_mask
                    ), (block_start, length, motif)
                    assert torch.equal(
                        packed_targets[block_slice], expected_targets
                    ), (block_start, length, motif)
                    assert not torch.any(packed_mask[block_slice, :block_start]), (
                        block_start,
                        length,
                        motif,
                    )


def test_packed_esm_uses_mask_id_from_tokenizer(tmp_path):
    data_path = tmp_path / "packed.bin"
    packed = np.memmap(data_path, mode="w+", dtype=np.uint8, shape=(8,))
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


def test_packed_atp_and_esm_forward_backward_are_finite(tmp_path):
    data_path = tmp_path / 'packed.bin'
    packed = np.memmap(data_path, mode='w+', dtype=np.uint8, shape=(8,))
    packed[:] = np.array([1, 4, 5, 2, 3, 1, 6, 2], dtype=np.uint8)
    packed.flush()
    del packed

    for model_type, model_class in (
        ('atp', BidirectionalCausalLM),
        ('esm', ESMlikeLM),
    ):
        torch.manual_seed(0)
        dataset = PackedUnirefData(
            str(data_path),
            tokenizer=TokenizerStub(mask_id=31),
            max_dim=8,
            model_type=model_type,
        )
        sequence, targets, attention = dataset[0]
        model = model_class(tiny_config())
        logits = model(sequence[None, :], attention_mask=attention[None, :, :])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert torch.isfinite(loss)
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
