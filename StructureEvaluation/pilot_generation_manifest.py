"""Make a fixed, provisional manifest for motif-completion pipeline tests."""

import argparse
import hashlib
import json
from pathlib import Path

from .prepare_af3_pilot import make_af3_input


def make_manifest(panel_bytes, pilots, seeds=(17, 29)):
    if hashlib.sha256(panel_bytes).hexdigest() != pilots["panel_sha256"]:
        raise ValueError("Panel file differs from the one used to select pilot motifs")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Generation seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Generation seeds must be distinct")

    panel = json.loads(panel_bytes)
    rows = []
    seen_accessions = set()
    for case in pilots["cases"]:
        if case["accession"] in seen_accessions:
            raise ValueError(f"Repeated pilot accession {case['accession']}")
        seen_accessions.add(case["accession"])
        matches = [entry for entry in panel if entry["accession"] == case["accession"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one panel entry for {case['accession']}")
        entry = matches[0]
        positions = case["uniprot_positions_1based"]
        if len(positions) != len(set(positions)) or any(position < 1 or position > entry["length"] for position in positions):
            raise ValueError(f"Invalid motif positions for {case['accession']}")
        if len(positions) >= entry["length"]:
            raise ValueError("Pilot must contain at least one hidden position")
        make_af3_input(entry, case)
        for attempt_index, seed in enumerate(seeds):
            rows.append({
                "accession": case["accession"],
                "reference_pdb": case["reference_pdb"],
                "original_sequence": entry["sequence"],
                "visible_positions_0based": [position - 1 for position in positions],
                "attempt_index": attempt_index,
                "generation_seed": seed,
            })

    return {
        "schema_version": 1,
        "status": "provisional pipeline test; not an approved evaluation set",
        "panel_sha256": pilots["panel_sha256"],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--pilots", type=Path, default=Path("StructureEvaluation/pilot_motifs.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--output", type=Path, default=Path("data/pilot_generation_manifest.json"))
    args = parser.parse_args()

    pilots = json.loads(args.pilots.read_text(encoding="utf-8"))
    manifest = make_manifest(args.panel.read_bytes(), pilots, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['rows'])} provisional generation attempts to {args.output}")


if __name__ == "__main__":
    main()
