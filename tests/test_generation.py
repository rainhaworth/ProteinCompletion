import torch
import pytest

from utils.config import BaseConfig
from utils.generation import (
    MASK_ID,
    gen_step_atp,
    gen_step_esmlike,
    make_sample_fn,
    make_mlm_input,
    nucleus_sample,
)
from utils.model_esmlike import ESMlikeLM


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


def greedy_sample(logits):
    vals, toks = torch.max(logits, dim=-1)
    best = torch.argmax(vals)
    return best, toks[best]


def test_make_mlm_input_masks_only_unrevealed_positions_without_mutation():
    seq = torch.tensor([[4, 5, 6, 7, 8]])
    original = seq.clone()

    model_input = make_mlm_input(seq, torch.tensor([1, 3]))

    assert torch.equal(model_input, torch.tensor([[MASK_ID, 5, MASK_ID, 7, MASK_ID]]))
    assert torch.equal(seq, original)


def test_make_mlm_input_reveals_generated_residue_on_next_step():
    seq = torch.tensor([[4, 5, 9, 7, 8]])

    before = make_mlm_input(seq, torch.tensor([1, 3]))
    after = make_mlm_input(seq, torch.tensor([1, 2, 3]))

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
    )

    assert torch.equal(model.inputs[-1], torch.tensor([[MASK_ID, 5, MASK_ID, 7, MASK_ID]]))
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
        model, seq_a, known, torch.device("cpu"), return_logits=True
    )
    logits_b, _ = gen_step_esmlike(
        model, seq_b, known, torch.device("cpu"), return_logits=True
    )

    assert torch.equal(logits_a, logits_b)


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
    logits = torch.tensor([[0.8, 0.2]])

    greedy = make_sample_fn('greedy')
    nucleus = make_sample_fn('nucleus', p=0.5)

    assert greedy(logits)[1] == 0
    assert nucleus(logits)[1] == 0

    with pytest.raises(ValueError, match='p must be'):
        nucleus_sample(logits, p=0)
