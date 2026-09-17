import json
from pathlib import Path

import numpy as np
import pytest

import StructureEvaluation.reference_geometry as geometry_module
from StructureEvaluation.reference_geometry import (
    author_chain_in_assembly,
    ca_coordinates,
    check_reference_case,
    geometry_metrics,
    ligand_min_distances,
    score_predicted_motif,
)


def test_geometry_is_rigid_motion_invariant_but_detects_distortion_and_reflection():
    reference = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    moved = reference @ rotation + np.array([10, -5, 3])
    same = geometry_metrics(reference, moved)
    assert same["motif_ca_rmsd_angstrom"] < 1e-10
    assert same["motif_ca_pair_distance_mae_angstrom"] < 1e-10

    distorted = moved.copy()
    distorted[3, 2] += 1
    assert geometry_metrics(reference, distorted)["motif_ca_rmsd_angstrom"] > 0.1
    assert geometry_metrics(reference, distorted)["motif_ca_pair_distance_mae_angstrom"] > 0.1

    reflected = reference.copy()
    reflected[:, 0] *= -1
    assert geometry_metrics(reference, reflected)["motif_ca_pair_distance_mae_angstrom"] < 1e-10
    assert geometry_metrics(reference, reflected)["motif_ca_rmsd_angstrom"] > 0.1


def test_author_numbering_and_assembly_are_not_confused_with_label_ids():
    atom = {
        "group_PDB": "ATOM", "label_atom_id": "CA", "label_alt_id": ".",
        "auth_asym_id": "B", "label_asym_id": "A",
        "author_residue": "70", "label_seq_id": "69",
        "occupancy": 1.0, "xyz": np.array([1.0, 2.0, 3.0]),
    }
    assert np.array_equal(ca_coordinates([atom], "B", ["70"], "author"), [[1, 2, 3]])
    assert np.array_equal(ca_coordinates([atom], "A", ["69"], "label"), [[1, 2, 3]])
    with pytest.raises(ValueError, match="Missing C-alpha"):
        ca_coordinates([atom], "B", ["69"], "author")

    cif = {
        "_atom_site.label_asym_id": ["A"],
        "_atom_site.auth_asym_id": ["B"],
        "_atom_site.group_PDB": ["ATOM"],
        "_pdbx_struct_assembly_gen.assembly_id": ["1", "2"],
        "_pdbx_struct_assembly_gen.asym_id_list": ["A,C", "D"],
    }
    assert author_chain_in_assembly(cif, "B", "1")
    assert not author_chain_in_assembly(cif, "B", "2")


def test_ligand_contacts_use_the_selected_author_chain():
    atoms = [
        {"group_PDB": "ATOM", "auth_asym_id": "B", "author_residue": "70", "xyz": np.array([0., 0., 0.])},
        {"group_PDB": "HETATM", "auth_asym_id": "B", "label_comp_id": "LIG", "xyz": np.array([3., 0., 0.])},
        {"group_PDB": "HETATM", "auth_asym_id": "A", "label_comp_id": "LIG", "xyz": np.array([0.1, 0., 0.])},
    ]
    assert ligand_min_distances(atoms, "B", ["70"], "LIG") == {"70": 3.0}


def test_prediction_scoring_uses_sequence_positions_not_pdb_author_numbers(monkeypatch):
    reference_positions = ["70", "73", "130"]
    uniprot_positions = [69, 72, 129]
    points = np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]])
    reference_atoms = [
        {
            "group_PDB": "ATOM", "label_atom_id": "CA", "label_alt_id": ".",
            "auth_asym_id": "B", "label_asym_id": "X", "author_residue": residue,
            "label_seq_id": str(position - 2), "label_comp_id": amino_acid,
            "occupancy": 1.0, "xyz": xyz,
        }
        for residue, position, amino_acid, xyz in zip(reference_positions, uniprot_positions, ["SER", "LYS", "SER"], points)
    ]
    predicted_atoms = [
        {
            "group_PDB": "ATOM", "label_atom_id": "CA", "label_alt_id": ".",
            "auth_asym_id": "A", "label_asym_id": "A", "author_residue": str(position),
            "label_seq_id": str(position), "label_comp_id": amino_acid,
            "occupancy": 1.0, "xyz": xyz + np.array([4., -2., 5.]),
        }
        for position, amino_acid, xyz in zip(uniprot_positions, ["SER", "LYS", "SER"], points)
    ]
    monkeypatch.setattr(
        geometry_module,
        "read_atoms",
        lambda path: (reference_atoms if path == "reference" else predicted_atoms, {}),
    )
    case = {
        "reference_author_chain": "B",
        "reference_author_residues": reference_positions,
        "uniprot_positions_1based": uniprot_positions,
        "expected_residue_letters": ["S", "K", "S"],
    }

    score = score_predicted_motif(case, "reference", "predicted")

    assert score["motif_ca_rmsd_angstrom"] < 1e-10
    assert score["motif_ca_pair_distance_mae_angstrom"] < 1e-10

    predicted_atoms[0]["label_comp_id"] = "ALA"
    with pytest.raises(ValueError, match="Predicted motif residues differ"):
        score_predicted_motif(case, "reference", "predicted")


def test_public_reference_pilots_when_cached():
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "data" / "reference_cif"
    pilots = json.loads((root / "StructureEvaluation" / "pilot_motifs.json").read_text())
    if not all((reference_dir / f"{case['reference_pdb']}.cif").exists() for case in pilots["cases"]):
        pytest.skip("Public reference mmCIF files are not cached locally")
    for case in pilots["cases"]:
        result = check_reference_case(case, reference_dir / f"{case['reference_pdb']}.cif")
        assert result["self_comparison"]["motif_ca_rmsd_angstrom"] < 1e-8
        assert result["self_comparison"]["motif_ca_pair_distance_mae_angstrom"] < 1e-8
        if case["ligand_comp_id"]:
            assert max(result["nearest_ligand_heavy_atom_angstrom"].values()) < 5.0
        if case.get("comparison_author_chain"):
            assert result["alternate_chain_comparison"]["motif_ca_rmsd_angstrom"] > 0.01
