import json
import sys

import pytest

from StructureEvaluation.af3_output import (
    read_single_protein_input,
    summarize_confidence,
    validate_prediction_sequence,
)
from StructureEvaluation.score_af3_pilot import main as score_pilot


def write_cif(path, residues, chain="A"):
    fields = (
        "group_PDB label_atom_id label_comp_id label_asym_id label_seq_id "
        "label_alt_id auth_asym_id auth_seq_id pdbx_PDB_ins_code "
        "pdbx_PDB_model_num type_symbol Cartn_x Cartn_y Cartn_z occupancy"
    ).split()
    lines = ["data_test", "loop_"] + [f"_atom_site.{field}" for field in fields]
    for position, author_number, component, xyz in residues:
        x, y, z = xyz
        lines.append(f"ATOM CA {component} {chain} {position} . {chain} {author_number} ? 1 C {x} {y} {z} 1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(directory, name="sample"):
    (directory / f"{name}_summary_confidences.json").write_text(json.dumps({
        "chain_ids": ["A"], "chain_ptm": [0.7], "ptm": 0.7, "ranking_score": 0.6,
    }), encoding="utf-8")
    (directory / f"{name}_confidences.json").write_text(json.dumps({
        "atom_chain_ids": ["A", "A", "A"], "atom_plddts": [80, 90, 100],
    }), encoding="utf-8")


def test_prediction_checks_full_sequence_and_numbering(tmp_path):
    predicted = tmp_path / "predicted.cif"
    residues = [
        (1, 1, "SER", (0, 0, 0)),
        (2, 2, "ASP", (3, 0, 0)),
        (3, 3, "HIS", (0, 4, 0)),
    ]
    write_cif(predicted, residues)
    assert validate_prediction_sequence(predicted, "SDH") == {
        "chain": "A", "sequence_length": 3, "ca_residue_count": 3,
    }

    with pytest.raises(ValueError, match="differs from input"):
        validate_prediction_sequence(predicted, "SAH")
    with pytest.raises(ValueError, match="missing C-alpha"):
        validate_prediction_sequence(predicted, "SDHA")
    with pytest.raises(ValueError, match="only protein chain B"):
        validate_prediction_sequence(predicted, "SDH", "B")

    write_cif(predicted, [(position + 1, author, component, xyz) for position, author, component, xyz in residues])
    with pytest.raises(ValueError, match="missing C-alpha"):
        validate_prediction_sequence(predicted, "SDH")


def test_confidence_summary_validates_chains_lengths_and_ranges(tmp_path):
    write_confidence(tmp_path)
    result = summarize_confidence(tmp_path, "sample")
    assert result["ptm"] == result["chain_ptm"] == 0.7
    assert result["ranking_score"] == 0.6
    assert result["chain_mean_atom_plddt"] == 90
    assert result["chain_atom_count"] == 3
    assert len(result["summary_confidence_sha256"]) == 64
    assert len(result["full_confidence_sha256"]) == 64
    full_path = tmp_path / "sample_confidences.json"
    full = json.loads(full_path.read_text())
    full["atom_plddts"] = [80, 90]
    full_path.write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(ValueError, match="different or zero lengths"):
        summarize_confidence(tmp_path, "sample")

    full["atom_plddts"] = [80, 90, 101]
    full_path.write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 100"):
        summarize_confidence(tmp_path, "sample")


def test_pilot_scorer_joins_input_structure_and_confidence(tmp_path, monkeypatch):
    prediction = tmp_path / "a0a0h2xjl0_wt_msa_free_model.cif"
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = reference_dir / "9C0L.cif"
    write_cif(prediction, [
        (1, 1, "SER", (0, 0, 0)),
        (2, 2, "ASP", (3, 0, 0)),
        (3, 3, "HIS", (0, 4, 0)),
    ])
    write_cif(reference, [
        (7, 93, "SER", (4, 1, 1)),
        (8, 190, "ASP", (7, 1, 1)),
        (9, 220, "HIS", (4, 5, 1)),
    ])
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({
        "name": "a0a0h2xjl0_wt_msa_free", "dialect": "alphafold3", "version": 1,
        "modelSeeds": [1],
        "sequences": [{"protein": {"id": "A", "sequence": "SDH"}}],
    }), encoding="utf-8")
    assert read_single_protein_input(input_path)["name"] == "a0a0h2xjl0_wt_msa_free"
    pilot_path = tmp_path / "pilots.json"
    pilot_path.write_text(json.dumps({"cases": [{
        "accession": "A0A0H2XJL0", "reference_pdb": "9C0L",
        "reference_author_chain": "A", "reference_author_residues": ["93", "190", "220"],
        "uniprot_positions_1based": [1, 2, 3], "expected_residue_letters": ["S", "D", "H"],
    }]}), encoding="utf-8")
    write_confidence(tmp_path, "a0a0h2xjl0_wt_msa_free")
    output = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", [
        "score_af3_pilot", str(prediction), "--input-json", str(input_path),
        "--confidence-dir", str(tmp_path), "--reference-dir", str(reference_dir),
        "--pilots", str(pilot_path), "--output", str(output),
    ])

    score_pilot()

    result = json.loads(output.read_text())
    assert result["prediction_validation"]["ca_residue_count"] == 3
    assert result["motif_ca_rmsd_angstrom"] < 1e-10
    assert result["predicted_confidence"]["chain_mean_atom_plddt"] == 90
