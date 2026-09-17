"""Validate a single-chain AlphaFold 3 prediction and summarize its confidence."""

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from Bio.SeqUtils import seq1

from .reference_geometry import read_atoms


def read_single_protein_input(json_path):
    job = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if job.get("dialect") != "alphafold3" or job.get("version") != 1:
        raise ValueError("Expected a native AlphaFold 3 version-1 input")
    if len(job.get("sequences", [])) != 1 or "protein" not in job["sequences"][0]:
        raise ValueError("Expected exactly one protein chain")
    protein = job["sequences"][0]["protein"]
    if not isinstance(protein.get("id"), str) or len(protein["id"]) != 1:
        raise ValueError("Expected one single-letter protein chain ID")
    sequence = protein.get("sequence")
    if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("Expected a nonempty sequence of standard amino acids")
    if not isinstance(job.get("name"), str) or not job["name"]:
        raise ValueError("Expected a nonempty job name")
    seeds = job.get("modelSeeds")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("Expected at least one nonnegative model seed")
    return job


def validate_prediction_sequence(cif_path, expected_sequence, chain="A"):
    """Require C-alpha residues to match every position of the input sequence."""
    atoms, _ = read_atoms(cif_path)
    protein_chains = {
        atom["label_asym_id"] for atom in atoms
        if atom["group_PDB"] == "ATOM" and atom["label_atom_id"] == "CA"
    }
    if protein_chains != {chain}:
        raise ValueError(f"Expected only protein chain {chain}, found {sorted(protein_chains)}")
    residues = {}
    for atom in atoms:
        if atom["group_PDB"] != "ATOM" or atom["label_asym_id"] != chain:
            continue
        if atom["label_atom_id"] != "CA":
            continue
        position = int(atom["label_seq_id"])
        letter = seq1(atom["label_comp_id"], custom_map={"UNK": "X"})
        previous = residues.get(position)
        if previous is not None and previous != letter:
            raise ValueError(f"Conflicting residue identities at {chain}:{position}")
        residues[position] = letter

    expected_positions = set(range(1, len(expected_sequence) + 1))
    missing = sorted(expected_positions - residues.keys())
    extra = sorted(residues.keys() - expected_positions)
    if missing or extra:
        raise ValueError(
            f"Predicted chain {chain} has missing C-alpha positions {missing[:10]} "
            f"and unexpected positions {extra[:10]}"
        )
    predicted_sequence = "".join(residues[position] for position in range(1, len(expected_sequence) + 1))
    if predicted_sequence != expected_sequence:
        mismatches = [
            position for position, (actual, expected) in enumerate(
                zip(predicted_sequence, expected_sequence), start=1
            ) if actual != expected
        ]
        raise ValueError(f"Predicted chain {chain} differs from input at positions {mismatches[:10]}")
    return {"chain": chain, "sequence_length": len(expected_sequence), "ca_residue_count": len(residues)}


def _unit_interval(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def summarize_confidence(output_dir, job_name, chain="A"):
    """Read the top-ranked AF3 confidence files without assuming atom-to-residue order."""
    output_dir = Path(output_dir)
    summary_path = output_dir / f"{job_name}_summary_confidences.json"
    full_path = output_dir / f"{job_name}_confidences.json"
    summary_bytes = summary_path.read_bytes()
    full_bytes = full_path.read_bytes()
    summary = json.loads(summary_bytes)
    full = json.loads(full_bytes)

    chain_ids = summary["chain_ids"]
    chain_ptm = summary["chain_ptm"]
    if chain_ids != [chain] or len(chain_ptm) != 1:
        raise ValueError("Expected confidence scores for the single predicted chain")

    atom_plddts = full["atom_plddts"]
    atom_chain_ids = full["atom_chain_ids"]
    if len(atom_plddts) != len(atom_chain_ids) or not atom_plddts:
        raise ValueError("Atom confidence and chain arrays have different or zero lengths")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 <= value <= 100
        for value in atom_plddts
    ):
        raise ValueError("Atom pLDDT values must be finite numbers between 0 and 100")
    selected_plddts = [value for value, atom_chain in zip(atom_plddts, atom_chain_ids) if atom_chain == chain]
    if not selected_plddts:
        raise ValueError(f"No atom confidences for chain {chain}")
    if set(atom_chain_ids) != {chain}:
        raise ValueError("Atom confidence contains an unexpected chain")

    ranking_score = summary["ranking_score"]
    if isinstance(ranking_score, bool) or not isinstance(ranking_score, (int, float)) or not math.isfinite(ranking_score):
        raise ValueError("ranking_score must be a finite number")
    return {
        "ptm": _unit_interval(summary["ptm"], "ptm"),
        "chain_ptm": _unit_interval(chain_ptm[0], "chain_ptm"),
        "ranking_score": float(ranking_score),
        "chain_mean_atom_plddt": fmean(selected_plddts),
        "chain_atom_count": len(selected_plddts),
        "summary_confidence_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "full_confidence_sha256": hashlib.sha256(full_bytes).hexdigest(),
    }
