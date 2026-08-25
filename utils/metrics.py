import numpy as np


VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def cross_entropy_to_perplexity(cross_entropy):
    """Convert natural-log cross-entropy to conventional perplexity."""
    return np.exp(cross_entropy)


def amino_acid_composition_entropy(sequence, ignore_indices=()):
    """Shannon entropy of amino-acid frequencies, measured in bits."""
    ignored = set(int(i) for i in ignore_indices)
    counts = {amino_acid: 0 for amino_acid in VALID_AAS}

    for index, amino_acid in enumerate(sequence):
        if index not in ignored and amino_acid in counts:
            counts[amino_acid] += 1

    frequencies = np.array([count for count in counts.values() if count], dtype=float)
    if frequencies.size == 0:
        return np.nan

    frequencies /= frequencies.sum()
    return -np.sum(frequencies * np.log2(frequencies))
