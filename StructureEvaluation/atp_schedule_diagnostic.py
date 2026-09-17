"""Compare ATP packed-training visibility with parallel or serial decoding.

This checks information sets, not model quality. It does not change the
production generator or decide how competing PTP/NTP predictions are combined.
Run with ``python -m StructureEvaluation.atp_schedule_diagnostic``.
"""

import argparse
import json

import torch

from utils.mask import idx_to_mask_targets_hanoi


def frontier_events(visible, length):
    """Return every legal adjacent prediction (predictor, target, head)."""
    visible = sorted(set(visible))
    if not visible or visible[0] < 0 or visible[-1] >= length:
        raise ValueError("Visible positions must be nonempty and within the sequence")
    events = []
    for position in visible:
        if position - 1 >= 0 and position - 1 not in visible:
            events.append((position, position - 1, "PTP"))
        if position + 1 < length and position + 1 not in visible:
            events.append((position, position + 1, "NTP"))
    return sorted(events, key=lambda event: (event[1], event[0], event[2]))


def compare_schedule(length, motif, mode):
    if mode not in ("parallel", "serial"):
        raise ValueError("mode must be parallel or serial")
    motif = sorted(set(int(position) for position in motif))
    if not motif or motif[0] < 0 or motif[-1] >= length:
        raise ValueError("Motif positions must be nonempty and within the sequence")
    train_mask, train_targets = idx_to_mask_targets_hanoi(torch.tensor(motif), length)
    visible = set(motif)
    trace = []
    round_number = 0
    while len(visible) < length:
        events = frontier_events(visible, length)
        if mode == "serial":
            events = events[:1]
        before = sorted(visible)
        for predictor, target, head in events:
            training_targets = train_targets[predictor].tolist()
            trained_pair = target == training_targets[0 if head == "PTP" else 1]
            training_visible = torch.nonzero(train_mask[predictor]).flatten().tolist()
            # The inference mask gives every row the same current visibility.
            # In packed training, a visible key row can have an older context;
            # this matters once the predictor reads its hidden state in layer 2.
            contextual_row_mismatches = [
                key for key in sorted(set(training_visible) & visible)
                if set(torch.nonzero(train_mask[key]).flatten().tolist()) != visible
            ]
            trace.append({
                "round": round_number,
                "predictor": predictor,
                "target": target,
                "head": head,
                "trained_pair": trained_pair,
                "inference_visible": before,
                "training_visible": training_visible,
                "training_only": sorted(set(training_visible) - visible),
                "inference_only": sorted(visible - set(training_visible)),
                "contextual_row_mismatches": contextual_row_mismatches,
            })
        visible.update(target for _, target, _ in events)
        round_number += 1
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--motif", type=int, nargs="+", default=[3, 6])
    args = parser.parse_args()
    for mode in ("parallel", "serial"):
        trace = compare_schedule(args.length, args.motif, mode)
        differences = [step for step in trace if step["training_only"] or step["inference_only"]]
        untrained = [step for step in trace if not step["trained_pair"]]
        print(json.dumps({"mode": mode, "predictions": len(trace), "untrained_predictor_target_pairs": len(untrained), "direct_visibility_mismatches": len(differences), "multilayer_context_mismatches": sum(bool(step["contextual_row_mismatches"]) for step in trace), "first_mismatches": differences[:5]}, indent=2))


if __name__ == "__main__":
    main()
