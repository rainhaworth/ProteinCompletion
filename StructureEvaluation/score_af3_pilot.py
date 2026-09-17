"""Score one completed AlphaFold 3 mmCIF against a mapped pilot motif."""

import argparse
import hashlib
import json
from pathlib import Path

from .af3_output import read_single_protein_input, summarize_confidence, validate_prediction_sequence
from .reference_geometry import score_predicted_motif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path, help="AlphaFold 3 model mmCIF file")
    parser.add_argument("--input-json", type=Path, required=True, help="The JSON used to produce this prediction")
    parser.add_argument("--confidence-dir", type=Path, help="Directory containing the top-ranked AF3 confidence files")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference_cif"))
    parser.add_argument("--pilots", type=Path, default=Path("StructureEvaluation/pilot_motifs.json"))
    parser.add_argument("--accession", default="A0A0H2XJL0")
    parser.add_argument("--prediction-chain", default="A")
    parser.add_argument("--output", type=Path, default=Path("data/af3_wt_pilot_score.json"))
    args = parser.parse_args()

    pilot = json.loads(args.pilots.read_text(encoding="utf-8"))
    matches = [case for case in pilot["cases"] if case["accession"] == args.accession]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching pilot case")
    case = matches[0]
    job = read_single_protein_input(args.input_json)
    protein = job["sequences"][0]["protein"]
    if protein["id"] != args.prediction_chain or not job["name"].lower().startswith(args.accession.lower() + "_"):
        raise ValueError("Input job name or chain does not match the selected pilot")
    validation = validate_prediction_sequence(args.prediction, protein["sequence"], args.prediction_chain)
    reference = args.reference_dir / f"{case['reference_pdb']}.cif"
    score = score_predicted_motif(case, reference, args.prediction, args.prediction_chain)
    result = {
        "accession": args.accession,
        "reference_pdb": case["reference_pdb"],
        "uniprot_positions_1based": case["uniprot_positions_1based"],
        "prediction_file": args.prediction.name,
        "prediction_sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input_json.read_bytes()).hexdigest(),
        "prediction_label_chain": args.prediction_chain,
        "prediction_validation": validation,
        **score,
    }
    if args.confidence_dir is not None:
        result["predicted_confidence"] = summarize_confidence(
            args.confidence_dir, job["name"], args.prediction_chain
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
