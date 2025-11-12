from pathlib import Path
import os
import csv
from generated_parser import parse_tsv  # (gen_pct, contiguous, ppl, se, idx, seq)
from structure_evaluator import StructureEvaluator
from esm_figure_generator import generate_figures
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", message="Entity ID not found in metadata")

# --- Config ---
model_id = "esm3-medium-2024-08"
api_key  = "0ZAf8oIXmihpWjfhpQbXh"
tsv_path = Path("esmlike-comp.tsv")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pdb_dir  = Path(f"pdb_outputs_{timestamp}")
out_tsv  = Path(f"results_{timestamp}.tsv")
fig_dir  = Path(f"figs_{timestamp}")  # <-- directory, not .tsv

# Ensure output dirs exist
pdb_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Parse TSV
num_rows_to_keep = 600
rows = parse_tsv(str(tsv_path))[:num_rows_to_keep]
if not rows:
    raise RuntimeError(f"No rows parsed from {tsv_path}")

print(len(rows))
print(f"Contiguous True rows:  {sum(1 for r in rows if r[1])}")
print(f"Contiguous False rows: {sum(1 for r in rows if not r[1])}")

# Load model once
evaluator = StructureEvaluator(model_id, api_key)

records = []

# --- TSV schema ---
fieldnames = [
    "record_id", "gen_pct", "contiguous",
    "ppl", "se", "length", "ptm",
    "mean_gen_plddt", "mean_non_gen_plddt",
    "idx", "seq"
]

# Create header only if the file doesn't exist
if not out_tsv.exists():
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

# Always open in append mode
# Run and stop when credit limit is hit
with out_tsv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

    for i, (gen_pct, contiguous, ppl, se, non_generated_indices, sequence) in enumerate(rows, start=1):
        pdb_out = pdb_dir / f"{tsv_path.stem}_rec{i}.pdb"

        try:
            ptm, mean_generated_plddt, mean_non_generated_plddt = evaluator.generate_structure(
                sequence=sequence,
                non_generated_indices=non_generated_indices,
                pdb_out=str(pdb_out)
            )

        except Exception as e:
            print("\n--- Credit limit reached ---")
            print("Saving progress and generating figures before exit.")
            f.flush()
            os.fsync(f.fileno())
            if records:
                generate_figures(records, outdir=fig_dir)
            break

        # --- Normal record save ---
        writer.writerow({
            "record_id": i,
            "gen_pct": gen_pct,
            "contiguous": contiguous,
            "ppl": ppl,
            "se": se,
            "length": len(sequence),
            "ptm": ptm,
            "mean_gen_plddt": mean_generated_plddt,
            "mean_non_gen_plddt": mean_non_generated_plddt,
            "idx": "[" + " ".join(str(x) for x in non_generated_indices) + "]",
            "seq": sequence,
        })
        f.flush()
        os.fsync(f.fileno())

        records.append({
            "gen_pct": gen_pct,
            "contiguous": contiguous,
            "ptm": ptm,
            "mean_gen_plddt": mean_generated_plddt,
            "mean_non_gen_plddt": mean_non_generated_plddt,
        })

        print(f"\n--- Record {i} ---")
        print("Generated %:", gen_pct)
        print("Contiguous:", contiguous)
        print("PPL:", ppl, "| SE:", se)
        print("Length:", len(sequence))
        print("Structure pTM:", ptm)
        print("Mean Generated pLDDT:", mean_generated_plddt)
        print("Mean Non-Generated pLDDT:", mean_non_generated_plddt)
        print("PDB saved to:", pdb_out)

# --- If we finish without hitting credit limit, still generate figures ---
if records:
    generate_figures(records, outdir=fig_dir)
