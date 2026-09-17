import copy
import json
from pathlib import Path

import pytest

from StructureEvaluation.prepare_af3_pilot import make_af3_input
from StructureEvaluation.reference_geometry import score_predicted_motif


def test_pilot_input_is_template_free_and_checks_sequence_identity():
    root = Path(__file__).resolve().parents[1]
    pilots = json.loads((root / "StructureEvaluation" / "pilot_motifs.json").read_text())
    case = next(item for item in pilots["cases"] if item["accession"] == "A0A0H2XJL0")
    sequence = ["A"] * (max(case["uniprot_positions_1based"]) + 2)
    for position, residue in zip(case["uniprot_positions_1based"], case["expected_residue_letters"]):
        sequence[position - 1] = residue
    entry = {
        "accession": case["accession"],
        "reference_pdb": case["reference_pdb"],
        "sequence": "".join(sequence),
        "length": len(sequence),
    }
    msa_free = make_af3_input(entry, case)
    protein = msa_free["sequences"][0]["protein"]

    assert msa_free["dialect"] == "alphafold3"
    assert msa_free["version"] == 1
    assert msa_free["modelSeeds"] == [1]
    assert msa_free["name"].endswith("_msa_free")
    assert protein["id"] == "A"
    assert protein["sequence"] == entry["sequence"]
    assert protein["templates"] == []
    assert protein["unpairedMsa"] == protein["pairedMsa"] == ""

    auto_msa_input = make_af3_input(entry, case, msa_free=False)
    auto_msa_protein = auto_msa_input["sequences"][0]["protein"]
    assert auto_msa_input["name"].endswith("_auto_msa")
    assert "unpairedMsa" not in auto_msa_protein
    assert "pairedMsa" not in auto_msa_protein
    assert auto_msa_protein["templates"] == []

    wrong_sequence = copy.deepcopy(entry)
    wrong_sequence["sequence"] = (
        wrong_sequence["sequence"][:92] + "A" + wrong_sequence["sequence"][93:]
    )
    with pytest.raises(ValueError, match="Pilot motif residues"):
        make_af3_input(wrong_sequence, case)


def test_deposited_reference_is_not_mistaken_for_a_full_length_prediction():
    root = Path(__file__).resolve().parents[1]
    reference = root / "data" / "reference_cif" / "9C0L.cif"
    if not reference.exists():
        pytest.skip("Public reference mmCIF is not cached locally")
    pilots = json.loads((root / "StructureEvaluation" / "pilot_motifs.json").read_text())
    case = next(item for item in pilots["cases"] if item["accession"] == "A0A0H2XJL0")

    with pytest.raises(ValueError, match="Predicted motif residues differ"):
        score_predicted_motif(case, reference, reference)
