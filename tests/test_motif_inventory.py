import json

from StructureEvaluation.candidate_motif_panel import (
    candidate_features,
    exact_position_range,
    parse_sifts_residues,
    write_review_table,
)


SIFTS = b'''<entry xmlns="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd">
  <residue dbSource="PDBe" dbResNum="1" dbResName="ALA">
    <crossRefDb dbSource="PDB" dbAccessionId="9TEST" dbChainId="A" dbResNum="10A" dbResName="ALA" />
    <crossRefDb dbSource="UniProt" dbAccessionId="PTEST" dbResNum="1" dbResName="A" />
  </residue>
  <residue dbSource="PDBe" dbResNum="2" dbResName="CYS">
    <crossRefDb dbSource="PDB" dbAccessionId="9TEST" dbChainId="A" dbResNum="11" dbResName="CYS" />
    <crossRefDb dbSource="UniProt" dbAccessionId="PTEST" dbResNum="2" dbResName="C" />
    <residueDetail property="Annotation">Not_Observed</residueDetail>
  </residue>
  <residue dbSource="PDBe" dbResNum="3" dbResName="ASP">
    <crossRefDb dbSource="PDB" dbAccessionId="9TEST" dbChainId="B" dbResNum="3" dbResName="ASP" />
    <crossRefDb dbSource="UniProt" dbAccessionId="PTEST" dbResNum="3" dbResName="D" />
  </residue>
</entry>'''


def test_sifts_mapping_preserves_chain_and_author_residue_labels():
    chains, mismatches = parse_sifts_residues(SIFTS, "PTEST", "9TEST", "ACD")

    assert not mismatches
    assert chains == {
        "A": {"1": [{"pdb_author_residue": "10A", "pdb_resname": "ALA"}]},
        "B": {"3": [{"pdb_author_residue": "3", "pdb_resname": "ASP"}]},
    }


def test_candidate_features_require_exact_positions_and_observed_chain():
    chains, _ = parse_sifts_residues(SIFTS, "PTEST", "9TEST", "ACD")
    record = {
        "features": [
            {"type": "Active site", "location": {"start": {"value": 1, "modifier": "EXACT"}, "end": {"value": 1, "modifier": "EXACT"}}, "evidences": [{"evidenceCode": "ECO:0000269"}]},
            {"type": "Binding site", "location": {"start": {"value": 1, "modifier": "EXACT"}, "end": {"value": 3, "modifier": "EXACT"}}},
            {"type": "Binding site", "location": {"start": {"value": 2, "modifier": "UNSURE"}, "end": {"value": 2, "modifier": "EXACT"}}},
            {"type": "Domain", "location": {"start": {"value": 1, "modifier": "EXACT"}, "end": {"value": 3, "modifier": "EXACT"}}},
        ]
    }

    features, counts = candidate_features(record, "ACD", chains)

    assert counts == {"Active site": 1, "Binding site": 2, "Domain": 1}
    assert len(features) == 2
    assert features[0]["fully_observed_chains"] == ["A"]
    assert features[0]["evidence_codes"] == ["ECO:0000269"]
    assert features[1]["fully_observed_chains"] == []
    assert exact_position_range({"start": {"value": 0, "modifier": "EXACT"}, "end": {"value": 1, "modifier": "EXACT"}}, 3) is None


def test_inventory_records_remain_serializable():
    chains, _ = parse_sifts_residues(SIFTS, "PTEST", "9TEST", "ACD")
    features, _ = candidate_features({"features": []}, "ACD", chains)
    assert json.loads(json.dumps(features)) == []


def test_review_table_keeps_feature_identity_and_positions(tmp_path):
    path = tmp_path / "review.csv"
    write_review_table([{
        "accession": "PTEST",
        "reference_pdb": "9TEST",
        "preflight_flags": [],
        "candidate_features": [{
            "type": "Active site",
            "description": "Catalytic residue",
            "ligand": None,
            "positions_uniprot_1based": [1, 3],
            "evidence_codes": ["ECO:0000269"],
            "fully_observed_chains": ["A"],
            "mapped_pdb_residues_by_chain": {"A": {"1": [], "3": []}},
        }],
    }], path)

    content = path.read_text(encoding="utf-8")
    assert "PTEST,9TEST,0,Active site" in content
    assert '"1,3"' in content
    assert "ECO:0000269" in content
