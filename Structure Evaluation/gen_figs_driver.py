from pathlib import Path
from datetime import datetime
from generated_parser import parse_rich_tsv
from esm_figure_generator import generate_figures

# Pick the newest results_*.tsv automatically (fallback to 'results.tsv' if none)
tsv_dir = Path("esmlike-structure-265-results.tsv")

# Timestamped figure directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fig_dir = Path(f"figs_{timestamp}")
fig_dir.mkdir(parents=True, exist_ok=True)

# Parse
records = parse_rich_tsv(tsv_path)
if not records:
    raise RuntimeError(f"No rows parsed from {tsv_path}")

print(f"Parsed {len(records)} rows from {tsv_path}")

# Generate figures
generate_figures(records, outdir=fig_dir)
print(f"Saved figures to {fig_dir.resolve()}")
