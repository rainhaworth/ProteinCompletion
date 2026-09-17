"""Validate reference motif coordinates and score C-alpha motif geometry.

This is a geometry-pipeline preflight, not a folding accuracy or binding score.
PDB references use SIFTS-resolved author residue numbers. A full-length AF3
prediction can instead use one-based sequence (label) residue numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1


def read_atoms(cif_path):
    cif = MMCIF2Dict(str(cif_path))
    fields = [
        "group_PDB", "label_atom_id", "label_comp_id", "label_asym_id",
        "label_seq_id", "label_alt_id", "auth_asym_id", "auth_seq_id",
        "pdbx_PDB_ins_code", "pdbx_PDB_model_num", "type_symbol",
        "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
    ]
    rows = zip(*(cif[f"_atom_site.{field}"] for field in fields))
    atoms = []
    for values in rows:
        row = dict(zip(fields, values))
        if row["pdbx_PDB_model_num"] != "1" or row["type_symbol"].upper() in ("H", "D"):
            continue
        insertion = row["pdbx_PDB_ins_code"]
        row["author_residue"] = row["auth_seq_id"] + ("" if insertion in (".", "?") else insertion)
        row["xyz"] = np.array([float(row[name]) for name in ("Cartn_x", "Cartn_y", "Cartn_z")])
        row["occupancy"] = float(row["occupancy"])
        atoms.append(row)
    return atoms, cif


def ca_coordinates(atoms, chain, residue_numbers, numbering="author"):
    if numbering not in ("author", "label"):
        raise ValueError("numbering must be 'author' or 'label'")
    chain_field = "auth_asym_id" if numbering == "author" else "label_asym_id"
    residue_field = "author_residue" if numbering == "author" else "label_seq_id"
    selected = {}
    for atom in atoms:
        if atom["group_PDB"] != "ATOM" or atom["label_atom_id"] != "CA":
            continue
        if atom[chain_field] != chain:
            continue
        key = atom[residue_field]
        if key not in selected or atom["occupancy"] > selected[key]["occupancy"]:
            selected[key] = atom
    missing = [str(number) for number in residue_numbers if str(number) not in selected]
    if missing:
        raise ValueError(f"Missing C-alpha atoms in {chain} ({numbering} numbering): {missing}")
    return np.stack([selected[str(number)]["xyz"] for number in residue_numbers])


def ca_residue_letters(atoms, chain, residue_numbers, numbering="author"):
    if numbering not in ("author", "label"):
        raise ValueError("numbering must be 'author' or 'label'")
    chain_field = "auth_asym_id" if numbering == "author" else "label_asym_id"
    residue_field = "author_residue" if numbering == "author" else "label_seq_id"
    selected = {
        atom[residue_field]: seq1(atom["label_comp_id"], custom_map={"UNK": "X"})
        for atom in atoms
        if atom["group_PDB"] == "ATOM"
        and atom[chain_field] == chain
        and atom["label_atom_id"] == "CA"
    }
    return [selected[str(number)] for number in residue_numbers]


def geometry_metrics(reference_xyz, predicted_xyz):
    reference = np.asarray(reference_xyz, dtype=float)
    predicted = np.asarray(predicted_xyz, dtype=float)
    if reference.shape != predicted.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Coordinate arrays must both have shape (n, 3)")
    if len(reference) < 3:
        raise ValueError("At least three motif residues are required for this RMSD check")
    ref_centered = reference - reference.mean(axis=0)
    pred_centered = predicted - predicted.mean(axis=0)
    left, _, right = np.linalg.svd(pred_centered.T @ ref_centered)
    sign = np.linalg.det(left @ right)
    rotation = left @ np.diag([1.0, 1.0, np.sign(sign)]) @ right
    aligned = pred_centered @ rotation
    rmsd = np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=1)))
    pairs = np.triu_indices(len(reference), k=1)
    reference_distances = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)[pairs]
    predicted_distances = np.linalg.norm(predicted[:, None] - predicted[None, :], axis=-1)[pairs]
    return {
        "motif_ca_rmsd_angstrom": float(rmsd),
        "motif_ca_pair_distance_mae_angstrom": float(np.mean(np.abs(predicted_distances - reference_distances))),
    }


def ligand_min_distances(atoms, author_chain, author_residues, ligand_comp_id):
    ligand = np.array([
        atom["xyz"] for atom in atoms
        if atom["group_PDB"] == "HETATM"
        and atom["auth_asym_id"] == author_chain
        and atom["label_comp_id"] == ligand_comp_id
    ])
    if len(ligand) == 0:
        raise ValueError(f"Ligand {ligand_comp_id} has no atoms on author chain {author_chain}")
    distances = {}
    for residue in author_residues:
        protein = np.array([
            atom["xyz"] for atom in atoms
            if atom["group_PDB"] == "ATOM"
            and atom["auth_asym_id"] == author_chain
            and atom["author_residue"] == str(residue)
        ])
        if len(protein) == 0:
            raise ValueError(f"No protein atoms for author residue {residue} on {author_chain}")
        distances[str(residue)] = float(np.linalg.norm(protein[:, None] - ligand[None, :], axis=-1).min())
    return distances


def author_chain_in_assembly(cif, author_chain, assembly_id):
    label_chains = {
        label for label, author, group in zip(
            cif["_atom_site.label_asym_id"],
            cif["_atom_site.auth_asym_id"],
            cif["_atom_site.group_PDB"],
        ) if author == author_chain and group == "ATOM"
    }
    for candidate, members in zip(
        cif["_pdbx_struct_assembly_gen.assembly_id"],
        cif["_pdbx_struct_assembly_gen.asym_id_list"],
    ):
        if candidate == str(assembly_id) and label_chains.intersection(members.split(",")):
            return True
    return False


def check_reference_case(case, cif_path):
    atoms, cif = read_atoms(cif_path)
    chain = case["reference_author_chain"]
    residues = case["reference_author_residues"]
    xyz = ca_coordinates(atoms, chain, residues, "author")
    actual_letters = ca_residue_letters(atoms, chain, residues)
    if actual_letters != case["expected_residue_letters"]:
        raise ValueError(f"Reference residues differ from the selected UniProt sequence: {actual_letters}")
    if not author_chain_in_assembly(cif, chain, case["assembly_id"]):
        raise ValueError(f"Author chain {chain} is not in assembly {case['assembly_id']}")
    result = {
        "accession": case["accession"],
        "reference_pdb": case["reference_pdb"],
        "assembly_id": case["assembly_id"],
        "reference_author_chain": chain,
        "uniprot_positions_1based": case["uniprot_positions_1based"],
        "reference_author_residues": residues,
        "residue_letters": actual_letters,
        "motif_ca_count": len(xyz),
        "self_comparison": geometry_metrics(xyz, xyz),
        "reference_ca_pair_distances_angstrom": [
            {
                "uniprot_positions_1based": [case["uniprot_positions_1based"][i], case["uniprot_positions_1based"][j]],
                "distance": float(np.linalg.norm(xyz[i] - xyz[j])),
            }
            for i in range(len(xyz)) for j in range(i + 1, len(xyz))
        ],
    }
    if case["ligand_comp_id"]:
        result["nearest_ligand_heavy_atom_angstrom"] = ligand_min_distances(
            atoms, chain, residues, case["ligand_comp_id"]
        )
    if case.get("comparison_author_chain"):
        comparison_chain = case["comparison_author_chain"]
        comparison_xyz = ca_coordinates(atoms, comparison_chain, residues, "author")
        result["alternate_reference_chain"] = comparison_chain
        result["alternate_chain_comparison"] = geometry_metrics(xyz, comparison_xyz)
    return result


def score_predicted_motif(case, reference_cif_path, predicted_cif_path, predicted_label_chain="A"):
    """Compare a full-length, sequence-numbered prediction with a mapped PDB motif.

    The predicted chain must use one-based label_seq_id values corresponding to
    the input sequence. Check this assumption on AF3 output before batch use.
    """
    reference_atoms, _ = read_atoms(reference_cif_path)
    predicted_atoms, _ = read_atoms(predicted_cif_path)
    reference_xyz = ca_coordinates(
        reference_atoms,
        case["reference_author_chain"],
        case["reference_author_residues"],
        "author",
    )
    predicted_xyz = ca_coordinates(
        predicted_atoms,
        predicted_label_chain,
        case["uniprot_positions_1based"],
        "label",
    )
    predicted_letters = ca_residue_letters(
        predicted_atoms,
        predicted_label_chain,
        case["uniprot_positions_1based"],
        "label",
    )
    if predicted_letters != case["expected_residue_letters"]:
        raise ValueError(
            "Predicted motif residues differ from the fixed input sequence; "
            "check AF3 chain and label numbering"
        )
    return geometry_metrics(reference_xyz, predicted_xyz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--pilots", type=Path, default=Path("StructureEvaluation/pilot_motifs.json"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot_reference_geometry.json"))
    args = parser.parse_args()
    pilot = json.loads(args.pilots.read_text(encoding="utf-8"))
    results = [
        check_reference_case(case, args.reference_dir / f"{case['reference_pdb']}.cif")
        for case in pilot["cases"]
    ]
    output = {"status": pilot["status"], "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
