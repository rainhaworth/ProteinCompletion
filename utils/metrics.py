import numpy as np
import torch


VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def cross_entropy_to_perplexity(cross_entropy):
    """Convert natural-log cross-entropy to conventional perplexity."""
    if torch.is_tensor(cross_entropy):
        perplexity = torch.exp(cross_entropy.detach()).cpu()
        return perplexity.item() if perplexity.numel() == 1 else perplexity.numpy()

    perplexity = np.exp(cross_entropy)
    return perplexity.item() if np.ndim(perplexity) == 0 else perplexity


def amino_acid_composition_entropy(sequence, ignore_indices=()):
    """Return amino-acid composition entropy in bits."""
    ignored = set(int(index) for index in ignore_indices)
    counts = {amino_acid: 0 for amino_acid in VALID_AAS}

    for index, amino_acid in enumerate(sequence):
        if index not in ignored and amino_acid in counts:
            counts[amino_acid] += 1

    frequencies = np.array([count for count in counts.values() if count], dtype=float)
    if frequencies.size == 0:
        return np.nan

    frequencies /= frequencies.sum()
    return -np.sum(frequencies * np.log2(frequencies))
