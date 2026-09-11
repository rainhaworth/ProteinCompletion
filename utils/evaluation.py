import torch


def normalize_model_type(model_type):
    aliases = {
        'atp': 'atp',
        'bidirectional': 'atp',
        'esm': 'esm',
        'esmlike': 'esm',
    }
    try:
        return aliases[model_type.lower()]
    except KeyError as exc:
        raise ValueError(f'Unknown model type {model_type}') from exc


def score_atp_logits(seq, logits):
    """Score both adjacent-token heads on a fully visible sequence."""
    if seq.ndim != 2 or seq.size(0) != 1:
        raise ValueError('Expected seq with shape [1, length]')
    if logits.ndim != 3 or logits.size(0) != 1:
        raise ValueError('Expected logits with shape [1, length, 2 * vocab]')
    if logits.size(1) != seq.size(1) or logits.size(-1) % 2:
        raise ValueError('ATP logits do not align with the input sequence')
    if seq.size(1) < 2:
        raise ValueError('ATP scoring requires at least two tokens')

    vocab_size = logits.size(-1) // 2
    previous_logits = logits[0, 1:, :vocab_size]
    next_logits = logits[0, :-1, vocab_size:]
    previous_targets = seq[0, :-1]
    next_targets = seq[0, 1:]

    previous_ce = torch.nn.functional.cross_entropy(previous_logits, previous_targets)
    next_ce = torch.nn.functional.cross_entropy(next_logits, next_targets)
    previous_accuracy = (previous_logits.argmax(-1) == previous_targets).float().mean()
    next_accuracy = (next_logits.argmax(-1) == next_targets).float().mean()
    return {
        'cross_entropy': (previous_ce + next_ce) / 2,
        'accuracy': (previous_accuracy + next_accuracy) / 2,
        'previous_cross_entropy': previous_ce,
        'next_cross_entropy': next_ce,
        'previous_accuracy': previous_accuracy,
        'next_accuracy': next_accuracy,
        'targets': 2 * (seq.size(1) - 1),
    }


def score_esm_logits(targets, logits, target_indices=None):
    """Score ESM logits at selected positions."""
    if targets.ndim != 2 or logits.ndim != 3:
        raise ValueError('Expected targets [batch, length] and logits [batch, length, vocab]')
    if logits.shape[:2] != targets.shape:
        raise ValueError('ESM logits do not align with the target sequence')

    if target_indices is None:
        selected_logits = logits.reshape(-1, logits.size(-1))
        selected_targets = targets.reshape(-1)
    else:
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=targets.device
        ).reshape(-1)
        if target_indices.numel() == 0:
            raise ValueError('At least one ESM target position is required')
        selected_logits = logits[:, target_indices, :].reshape(-1, logits.size(-1))
        selected_targets = targets[:, target_indices].reshape(-1)

    cross_entropy = torch.nn.functional.cross_entropy(selected_logits, selected_targets)
    accuracy = (selected_logits.argmax(-1) == selected_targets).float().mean()
    return {
        'cross_entropy': cross_entropy,
        'accuracy': accuracy,
        'targets': selected_targets.numel(),
    }
