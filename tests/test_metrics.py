import numpy as np
import torch

from utils.metrics import (
    amino_acid_composition_entropy,
    cross_entropy_to_perplexity,
)


def test_cross_entropy_to_perplexity_uses_natural_exponential():
    assert np.isclose(cross_entropy_to_perplexity(np.log(20.0)), 20.0)


def test_cross_entropy_to_perplexity_accepts_gradient_tracking_tensors():
    cross_entropy = torch.tensor(np.log(20.0), requires_grad=True)

    assert np.isclose(cross_entropy_to_perplexity(cross_entropy), 20.0)


def test_amino_acid_composition_entropy_is_frequency_entropy():
    assert np.isclose(amino_acid_composition_entropy('AACC'), 1.0)
    assert np.isclose(
        amino_acid_composition_entropy('AACC', ignore_indices=[0, 2]),
        1.0,
    )


def test_amino_acid_composition_entropy_handles_no_valid_residues():
    assert np.isnan(amino_acid_composition_entropy('123'))
