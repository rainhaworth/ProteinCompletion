"""Prepare one wild-type AlphaFold 3 input from the provisional motif panel."""

import argparse
import hashlib
import json
from pathlib import Path


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def make_af3_input(panel_entry, pilot_case, seed=1, msa_free=True):
    sequence = panel_entry["sequence"]
    if panel_entry["accession"] != pilot_case["accession"]:
        raise ValueError("Panel entry and pilot accession differ")
    if panel_entry["reference_pdb"] != pilot_case["reference_pdb"]:
        raise ValueError("Panel entry and pilot reference PDB differ")
    if len(sequence) != panel_entry["length"] or set(sequence) - AMINO_ACIDS:
        raise ValueError("Expected a full-length sequence with standard amino acids")
    actual_motif = [sequence[position - 1] for position in pilot_case["uniprot_positions_1based"]]
    if actual_motif != pilot_case["expected_residue_letters"]:
        raise ValueError("Pilot motif residues do not match the panel sequence")
    protein = {"id": "A", "sequence": sequence, "templates": []}
    if msa_free:
        protein["unpairedMsa"] = ""
        protein["pairedMsa"] = ""
    return {
        "name": f"{panel_entry['accession'].lower()}_wt_{'msa_free' if msa_free else 'auto_msa'}",
        "modelSeeds": [seed],
        "sequences": [{"protein": protein}],
        "dialect": "alphafold3",
        "version": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--pilots", type=Path, default=Path("StructureEvaluation/pilot_motifs.json"))
    parser.add_argument("--accession", default="A0A0H2XJL0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--auto-msa", action="store_true", help="Let AlphaFold 3 search for MSAs; omit this for the quick MSA-free smoke run")
    parser.add_argument("--output", type=Path, default=Path("data/af3_wt_pilot.json"))
    args = parser.parse_args()

    panel_bytes = args.panel.read_bytes()
    pilot = json.loads(args.pilots.read_text(encoding="utf-8"))
    if hashlib.sha256(panel_bytes).hexdigest() != pilot["panel_sha256"]:
        raise ValueError("The panel file differs from the one used to select pilot motifs")
    panel_matches = [entry for entry in json.loads(panel_bytes) if entry["accession"] == args.accession]
    pilot_matches = [case for case in pilot["cases"] if case["accession"] == args.accession]
    if len(panel_matches) != 1 or len(pilot_matches) != 1:
        raise ValueError("Expected exactly one matching panel entry and pilot case")
    result = make_af3_input(panel_matches[0], pilot_matches[0], args.seed, not args.auto_msa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} for {args.accession} ({len(result['sequences'][0]['protein']['sequence'])} residues)")


if __name__ == "__main__":
    main()
