from pathlib import Path
from datetime import datetime
from generated_parser import parse_rich_tsv
from esm_figure_generator import generate_figures_combined

# Input TSVs
bcm_path = Path("BCM-275-results.tsv")
esm_path = Path("esmlike-structure-265-results.tsv")

# Timestamped output directory for figures
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fig_dir = Path(f"figs_{timestamp}")
fig_dir.mkdir(parents=True, exist_ok=True)

# --- Parse input TSV files ---
bcm_records = parse_rich_tsv(bcm_path)
esm_records = parse_rich_tsv(esm_path)

print(f"Parsed {len(bcm_records)} rows from {bcm_path}")
print(f"Parsed {len(esm_records)} rows from {esm_path}")

# --- Generate and save figures ---
generate_figures_combined(bcm_records, esm_records, outdir=fig_dir)
print(f"Saved figures to: {fig_dir.resolve()}")
