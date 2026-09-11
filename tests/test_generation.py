import itertools

import torch
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from utils.config import BaseConfig
from utils.generation import (
    decode_token_ids,
    gen_step_atp,
    gen_step_esmlike,
    make_sample_fn,
    make_mlm_input,
    resolve_terminal_ids,
    update_generation_state,
)
from utils.model_esmlike import ESMlikeLM


MASK_ID = 31
SEP_ID = 3


class RecordingESM(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs = []
        self.attention_masks = []

    def forward(self, seq, attention_mask=None):
        self.inputs.append(seq.detach().clone())
        self.attention_masks.append(attention_mask)
        return torch.nn.functional.one_hot(seq, self.vocab_size).float()


class RecordingATP(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs = []
        self.attention_masks = []

    def forward(self, seq, attention_mask=None):
        self.inputs.append(seq.detach().clone())
        self.attention_masks.append(attention_mask.detach().clone())
        logits = torch.zeros(seq.size(0), seq.size(1), self.vocab_size * 2)
        logits[..., 4] = 1.0
        logits[..., self.vocab_size + 5] = 1.0
        return logits


class TerminalATP(torch.nn.Module):
    def __init__(self, vocab_size=32, bos_id=1, eos_id=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(self, seq, attention_mask=None):
        logits = torch.zeros(seq.size(0), seq.size(1), self.vocab_size * 2)
        logits[..., self.bos_id] = 20.0
        logits[..., self.vocab_size + self.eos_id] = 20.0
        return logits


def greedy_sample(logits):
    vals, toks = torch.max(logits, dim=-1)
    best = torch.argmax(vals)
    return best, toks[best]


def test_make_mlm_input_masks_only_unrevealed_positions_without_mutation():
    seq = torch.tensor([[4, 5, 6, 7, 8]])
    original = seq.clone()

    model_input = make_mlm_input(seq, torch.tensor([1, 3]), MASK_ID)

    assert torch.equal(model_input, torch.tensor([[MASK_ID, 5, MASK_ID, 7, MASK_ID]]))
    assert torch.equal(seq, original)


def test_make_mlm_input_reveals_generated_residue_on_next_step():
    seq = torch.tensor([[4, 5, 9, 7, 8]])

    before = make_mlm_input(seq, torch.tensor([1, 3]), MASK_ID)
    after = make_mlm_input(seq, torch.tensor([1, 2, 3]), MASK_ID)

    assert before[0, 2] == MASK_ID
    assert after[0, 2] == 9


def test_esmlike_generation_masks_unknown_tokens_and_uses_full_attention():
    model = RecordingESM()
    seq = torch.tensor([[4, 5, 6, 7, 8]])

    gen_step_esmlike(
        model,
        seq,
        torch.tensor([1, 3]),
        torch.device("cpu"),
        sample_fn=greedy_sample,
        return_logits=True,
        mask_id=MASK_ID,
    )

    assert torch.equal(model.inputs[-1], torch.tensor([[MASK_ID, 5, MASK_ID, 7, MASK_ID]]))
    assert not torch.any(model.inputs[-1] == SEP_ID)
    assert model.attention_masks[-1] is None


def test_esmlike_logits_are_invariant_to_unrevealed_true_tokens():
    torch.manual_seed(0)
    config = BaseConfig(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=64,
        n_layer=2,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = ESMlikeLM(config).eval()
    seq_a = torch.tensor([[4, 5, 6, 7, 8]])
    seq_b = torch.tensor([[9, 5, 10, 7, 11]])
    known = torch.tensor([1, 3])

    logits_a, _ = gen_step_esmlike(
        model, seq_a, known, torch.device("cpu"), return_logits=True, mask_id=MASK_ID
    )
    logits_b, _ = gen_step_esmlike(
        model, seq_b, known, torch.device("cpu"), return_logits=True, mask_id=MASK_ID
    )

    assert torch.equal(logits_a, logits_b)


def test_esmlike_generation_requires_explicit_mask_id():
    model = RecordingESM()
    seq = torch.tensor([[4, 5, 6]])

    try:
        gen_step_esmlike(model, seq, torch.tensor([1]), torch.device("cpu"))
    except ValueError as exc:
        assert "mask" in str(exc).lower()
    else:
        raise AssertionError("ESM generation accepted a missing mask token ID")


def test_atp_generation_path_is_unchanged():
    model = RecordingATP()
    seq = torch.tensor([[4, 5, 6, 7, 8]])

    gen_step_atp(
        model,
        seq,
        torch.tensor([2]),
        torch.device("cpu"),
        sample_fn=greedy_sample,
        return_logits=True,
    )

    assert torch.equal(model.inputs[-1], seq)
    assert model.attention_masks[-1] is not None


def test_sampling_configuration_controls_the_selected_sampler():
    probabilities = torch.tensor([[0.8, 0.2]])

    assert make_sample_fn('greedy')(probabilities)[1] == 0
    assert make_sample_fn('nucleus', p=0.5)(probabilities)[1] == 0

    with pytest.raises(ValueError, match='p must be'):
        make_sample_fn('nucleus', p=0)(probabilities)


def test_atp_fixed_length_generation_accepts_every_standard_amino_acid_frontier():
    amino_acid_ids = [4 + ord(amino_acid) - ord('A') for amino_acid in 'ACDEFGHIKLMNPQRSTVWY']

    for amino_acid_id in amino_acid_ids:
        right_model = RecordingATP()
        _, right_position = gen_step_atp(
            right_model,
            torch.tensor([[amino_acid_id, 5]]),
            torch.tensor([0]),
            torch.device('cpu'),
            sample_fn=greedy_sample,
            predict_terminals=False,
        )
        assert int(right_position) == 1, amino_acid_id
        assert len(right_model.inputs) == 1

        left_model = RecordingATP()
        _, left_position = gen_step_atp(
            left_model,
            torch.tensor([[5, amino_acid_id]]),
            torch.tensor([1]),
            torch.device('cpu'),
            sample_fn=greedy_sample,
            predict_terminals=False,
        )
        assert int(left_position) == 0, amino_acid_id
        assert len(left_model.inputs) == 1


def test_atp_fixed_length_generation_covers_every_motif_through_length_eight():
    for length in range(1, 9):
        sequence_template = torch.arange(4, 4 + length)[None, :]
        for motif_size in range(1, length + 1):
            for motif in itertools.combinations(range(length), motif_size):
                model = RecordingATP()
                sequence = sequence_template.clone()
                known = torch.tensor(motif)

                for _ in range(length - motif_size):
                    new_token, new_position = gen_step_atp(
                        model,
                        sequence,
                        known,
                        torch.device('cpu'),
                        sample_fn=greedy_sample,
                        predict_terminals=False,
                    )
                    assert new_token is not None, (length, motif)
                    sequence, known = update_generation_state(
                        sequence, known, new_token, new_position
                    )

                assert torch.equal(known, torch.arange(length)), (length, motif)
                assert gen_step_atp(
                    model,
                    sequence,
                    known,
                    torch.device('cpu'),
                    sample_fn=greedy_sample,
                    predict_terminals=False,
                ) == (None, None)


def test_atp_can_generate_both_terminals_and_stop():
    model = TerminalATP()
    sequence = torch.tensor([[6]])
    known = torch.tensor([0])
    invalid_ids = [0, 1, 2, 3] + list(range(24, 32))

    for _ in range(3):
        new_token, new_position = gen_step_atp(
            model,
            sequence,
            known,
            torch.device('cpu'),
            invalid_ids=invalid_ids,
            sample_fn=greedy_sample,
            predict_terminals=True,
            bos_id=1,
            eos_id=2,
        )
        if new_token is None:
            break
        sequence, known = update_generation_state(
            sequence, known, new_token, new_position
        )

    assert torch.equal(sequence, torch.tensor([[1, 6, 2]]))
    assert torch.equal(known, torch.arange(3))
    assert new_token is None


def test_uniref_token_ids_decode_without_spaces():
    tokens = ['<pad>', '<bos>', '<eos>', '<sep>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab['<unk>'] = len(vocab)
    vocab['<mask>'] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, '<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Split('', 'isolated')
    ids = tokenizer.encode('ACD').ids

    assert tokenizer.decode(ids) == 'A C D'
    assert decode_token_ids(tokenizer, ids) == 'ACD'
    assert resolve_terminal_ids(tokenizer) == (1, 2)
