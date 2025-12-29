from pathlib import Path
from datetime import datetime

from generated_parser import parse_rich_tsv, parse_original_sequences_tsv
from esm_figure_generator import generate_figures_combined, plot_length_genpct_metric_all

# Input TSVs
bcm_path = Path("BCM-275-results.tsv")
esm_path = Path("esmlike-structure-265-results.tsv")
orig_path = Path("seqs_scored_40.tsv")   # original, non-generated sequences TSV

# Timestamped output directory for figures
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fig_dir = Path(f"figs_{timestamp}")
fig_dir.mkdir(parents=True, exist_ok=True)

# Parse input TSV files
bcm_records = parse_rich_tsv(bcm_path)
esm_records = parse_rich_tsv(esm_path)
orig_records = parse_original_sequences_tsv(orig_path)

print(f"Parsed {len(bcm_records)} rows from {bcm_path}")
print(f"Parsed {len(esm_records)} rows from {esm_path}")
print(f"Parsed {len(orig_records)} rows from {orig_path}")

# Generate and save BCM/ESM comparison figures
generate_figures_combined(
    bcm_records,
    esm_records,
    original_tsv=orig_path,  # path used inside to compute baseline means
    outdir=fig_dir,
)

# Generate length × gen% × metric plots (pTM and pLDDT)
plot_length_genpct_metric_all(
    bcm_records=bcm_records,
    esm_records=esm_records,
    orig_records=orig_records,
    metric_kind="ptm",
    outdir=fig_dir,
    y_max=1.0,
)

plot_length_genpct_metric_all(
    bcm_records=bcm_records,
    esm_records=esm_records,
    orig_records=orig_records,
    metric_kind="plddt",
    outdir=fig_dir,
    y_max=1.0,
)

print(f"Saved figures to: {fig_dir.resolve()}")
