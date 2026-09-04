"""Generate comparison figures for ATP and ESM structure evaluations."""

import argparse
from datetime import datetime
from pathlib import Path

try:
    from .generated_parser import parse_original_sequences_tsv, parse_rich_tsv
except ImportError:
    from generated_parser import parse_original_sequences_tsv, parse_rich_tsv


def build_parser():
    parser = argparse.ArgumentParser(description='Plot ATP and ESM structure results.')
    parser.add_argument('--atp', required=True, help='ATP structure-results TSV')
    parser.add_argument('--esm', required=True, help='ESM structure-results TSV')
    parser.add_argument('--original', required=True, help='Original-sequence structure TSV')
    parser.add_argument('--output-dir', default='', help='Figure directory')
    parser.add_argument('--plddt-scale-max', type=float, default=1.0)
    return parser


def generate_outputs(
    atp_records,
    esm_records,
    original_records,
    original_path,
    output_dir,
    plddt_scale_max,
    combined_figure_fn=None,
    length_figure_fn=None,
):
    if combined_figure_fn is None or length_figure_fn is None:
        try:
            from .figure_generator import (
                generate_figures_combined,
                plot_length_genpct_metric_all,
            )
        except ImportError:
            from figure_generator import (
                generate_figures_combined,
                plot_length_genpct_metric_all,
            )
        combined_figure_fn = combined_figure_fn or generate_figures_combined
        length_figure_fn = length_figure_fn or plot_length_genpct_metric_all

    combined_figure_fn(
        atp_records,
        esm_records,
        original_tsv=original_path,
        outdir=output_dir,
        plddt_scale_max=plddt_scale_max,
    )
    for metric in ('ptm', 'plddt'):
        length_figure_fn(
            bcm_records=atp_records,
            esm_records=esm_records,
            orig_records=original_records,
            metric_kind=metric,
            outdir=output_dir,
            y_max=plddt_scale_max if metric == 'plddt' else 1.0,
        )


def main(argv=None, combined_figure_fn=None, length_figure_fn=None):
    args = build_parser().parse_args(argv)
    paths = {
        'ATP': Path(args.atp),
        'ESM': Path(args.esm),
        'original': Path(args.original),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{label} TSV not found: {path}')
    if args.plddt_scale_max <= 0:
        raise ValueError('--plddt-scale-max must be positive')

    atp_records = parse_rich_tsv(paths['ATP'])
    esm_records = parse_rich_tsv(paths['ESM'])
    original_records = parse_original_sequences_tsv(paths['original'])
    if not atp_records or not esm_records or not original_records:
        raise ValueError('Each input TSV must contain at least one parsed record')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path(f'figures_combined_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_outputs(
        atp_records,
        esm_records,
        original_records,
        paths['original'],
        output_dir,
        args.plddt_scale_max,
        combined_figure_fn,
        length_figure_fn,
    )
    print('saved figures to', output_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
