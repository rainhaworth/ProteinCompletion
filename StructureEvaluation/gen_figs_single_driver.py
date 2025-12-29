from pathlib import Path
from datetime import datetime

from generated_parser import parse_rich_tsv, parse_original_sequences_tsv
from esm_figure_generator import generate_figures_single

# ----------------------------
# Config
# ----------------------------

# TSV of ONE model's generated-completion results
model_path = Path("esmlike-structure-265-results.tsv")

# TSV of original, non-generated sequences
orig_path = Path("seqs_scored_40.tsv")

# Optional: label/color for plots
model_label = "ESM"
model_color = "red"

# pLDDT scale: 1.0 if in [0,1], 100.0 if in [0,100]
plddt_scale_max = 1.0

# Optional y-limits for the LENGTH plots
length_ymax_ptm = 1.0
length_ymax_plddt = 1.0

# ----------------------------
# Timestamped output directory
# ----------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fig_dir = Path(f"figs_{model_label.lower()}_{timestamp}")
fig_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Parse TSV files
# ----------------------------
model_records = parse_rich_tsv(model_path)
orig_records = parse_original_sequences_tsv(orig_path)

print(f"Parsed {len(model_records)} rows from {model_path}")
print(f"Parsed {len(orig_records)} rows from {orig_path}")

# ----------------------------
# Ensure single-records have length info (needed for length plots)
# ----------------------------
for r in model_records:
    if "length" not in r:
        if "seq" in r and r["seq"] is not None:
            r["length"] = len(str(r["seq"]))

# ----------------------------
# Generate figures for ONE model
# (Includes gen_pct plots + histograms + length×metric plots)
# ----------------------------
generate_figures_single(
    records=model_records,
    original_tsv=orig_path,     # used to compute baseline lines/curves
    outdir=fig_dir,
    model_label=model_label,
    model_color=model_color,
    plddt_scale_max=plddt_scale_max,
    length_ymax_ptm=length_ymax_ptm,
    length_ymax_plddt=length_ymax_plddt,
)

print(f"Saved figures to: {fig_dir.resolve()}")
