"""Generate figures for one model's structure-evaluation results."""

import argparse
from datetime import datetime
from pathlib import Path

try:
    from .generated_parser import parse_original_sequences_tsv, parse_rich_tsv
except ImportError:
    from generated_parser import parse_original_sequences_tsv, parse_rich_tsv


def build_parser():
    parser = argparse.ArgumentParser(description='Plot one model\'s structure results.')
    parser.add_argument('--input', required=True, help='Structure-results TSV')
    parser.add_argument('--original', required=True, help='Original-sequence structure TSV')
    parser.add_argument('--model-label', default='model')
    parser.add_argument('--model-color', default='blue')
    parser.add_argument('--output-dir', default='', help='Figure directory')
    parser.add_argument('--plddt-scale-max', type=float, default=1.0)
    parser.add_argument('--length-ymax-ptm', type=float, default=1.0)
    parser.add_argument('--length-ymax-plddt', type=float, default=1.0)
    return parser


def main(argv=None, figure_fn=None):
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    original_path = Path(args.original)
    for label, path in (('input', input_path), ('original', original_path)):
        if not path.is_file():
            raise FileNotFoundError(f'{label} TSV not found: {path}')
    if min(
        args.plddt_scale_max,
        args.length_ymax_ptm,
        args.length_ymax_plddt,
    ) <= 0:
        raise ValueError('Plot scales must be positive')

    records = parse_rich_tsv(input_path)
    original_records = parse_original_sequences_tsv(original_path)
    if not records or not original_records:
        raise ValueError('Each input TSV must contain at least one parsed record')
    for record in records:
        if record.get('length') is None and record.get('seq'):
            record['length'] = len(record['seq'])

    if figure_fn is None:
        try:
            from .figure_generator import generate_figures_single
        except ImportError:
            from figure_generator import generate_figures_single
        figure_fn = generate_figures_single

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    default_dir = f'figures_{args.model_label.lower()}_{timestamp}'
    output_dir = Path(args.output_dir) if args.output_dir else Path(default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_fn(
        records=records,
        original_tsv=original_path,
        outdir=output_dir,
        model_label=args.model_label,
        model_color=args.model_color,
        plddt_scale_max=args.plddt_scale_max,
        length_ymax_ptm=args.length_ymax_ptm,
        length_ymax_plddt=args.length_ymax_plddt,
    )
    print('saved figures to', output_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
