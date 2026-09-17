import hashlib
import json

import pytest

from StructureEvaluation.pilot_generation_manifest import make_manifest


def example_panel_and_pilots():
    panel = [{
        "accession": "EXAMPLE", "reference_pdb": "1ABC",
        "sequence": "ACDEFG", "length": 6,
    }]
    panel_bytes = json.dumps(panel).encode("utf-8")
    pilots = {
        "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "cases": [{
            "accession": "EXAMPLE", "reference_pdb": "1ABC",
            "uniprot_positions_1based": [2, 5],
            "expected_residue_letters": ["C", "F"],
        }],
    }
    return panel_bytes, pilots


def test_manifest_fixes_context_and_distinguishes_attempts():
    panel_bytes, pilots = example_panel_and_pilots()
    first = make_manifest(panel_bytes, pilots, seeds=[17, 29])
    assert first == make_manifest(panel_bytes, pilots, seeds=[17, 29])
    assert first["status"].startswith("provisional")
    assert len(first["rows"]) == 2
    assert [row["attempt_index"] for row in first["rows"]] == [0, 1]
    assert [row["generation_seed"] for row in first["rows"]] == [17, 29]
    assert all(row["visible_positions_0based"] == [1, 4] for row in first["rows"])
    assert all(row["original_sequence"] == "ACDEFG" for row in first["rows"])
    assert all("model" not in row for row in first["rows"])


def test_manifest_rejects_mismatched_panel_and_invalid_design():
    panel_bytes, pilots = example_panel_and_pilots()
    with pytest.raises(ValueError, match="Panel file differs"):
        make_manifest(panel_bytes + b" ", pilots)
    with pytest.raises(ValueError, match="distinct"):
        make_manifest(panel_bytes, pilots, seeds=[17, 17])

    pilots["cases"][0]["uniprot_positions_1based"] = [2, 2]
    with pytest.raises(ValueError, match="Invalid motif positions"):
        make_manifest(panel_bytes, pilots)
