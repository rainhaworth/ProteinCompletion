"""Run ESM structure prediction for completed protein sequences."""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

try:
    from .generated_parser import parse_tsv
    from .structure_evaluator import StructureEvaluator
except ImportError:
    from generated_parser import parse_tsv
    from structure_evaluator import StructureEvaluator


FIELDNAMES = [
    'record_id',
    'gen_pct',
    'contiguous',
    'ppl',
    'se',
    'length',
    'ptm',
    'mean_gen_plddt',
    'mean_non_gen_plddt',
    'idx',
    'seq',
    'error',
]


def build_parser():
    parser = argparse.ArgumentParser(
        description='Predict structures and score generated protein sequences.'
    )
    parser.add_argument('--input', required=True, help='Completion TSV from eval-completion.py')
    parser.add_argument('--output', default='', help='Output TSV; defaults to a timestamped file')
    parser.add_argument('--pdb-dir', default='', help='PDB output directory')
    parser.add_argument('--model-id', default='esm3-medium-2024-08')
    parser.add_argument('--api-key-env', default='ESM_API_KEY')
    parser.add_argument('--max-records', type=int, default=600)
    parser.add_argument('--fail-fast', action='store_true')
    return parser


def evaluate_rows(rows, evaluator, output_path, pdb_dir, fail_fast=False):
    output_path = Path(output_path)
    pdb_dir = Path(pdb_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter='\t')
        writer.writeheader()

        for record_id, row in enumerate(rows, start=1):
            gen_pct, contiguous, ppl, entropy, known_indices, sequence = row
            result = {
                'record_id': record_id,
                'gen_pct': gen_pct,
                'contiguous': contiguous,
                'ppl': ppl,
                'se': entropy,
                'length': len(sequence),
                'ptm': '',
                'mean_gen_plddt': '',
                'mean_non_gen_plddt': '',
                'idx': '[' + ' '.join(map(str, known_indices)) + ']',
                'seq': sequence,
                'error': '',
            }
            pdb_path = pdb_dir / f'{Path(output_path).stem}_record{record_id}.pdb'
            try:
                ptm, generated_plddt, known_plddt = evaluator.generate_structure(
                    sequence=sequence,
                    non_generated_indices=known_indices,
                    pdb_out=str(pdb_path),
                )
                result.update(
                    ptm=ptm,
                    mean_gen_plddt=generated_plddt,
                    mean_non_gen_plddt=known_plddt,
                )
                successful += 1
                print(f'record {record_id}: pTM={ptm:.4f}')
            except Exception as exc:
                failed += 1
                message = str(exc)
                api_key = getattr(evaluator, 'api_key', None)
                if api_key:
                    message = message.replace(str(api_key), '[redacted]')
                result['error'] = f'{type(exc).__name__}: {message}'
                print(f'record {record_id} failed: {result["error"]}')
                writer.writerow(result)
                handle.flush()
                if fail_fast:
                    raise
                continue

            writer.writerow(result)
            handle.flush()

    return successful, failed


def main(argv=None, evaluator_factory=StructureEvaluator):
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f'Input TSV not found: {input_path}')
    if args.max_records <= 0:
        raise ValueError('--max-records must be positive')

    rows = parse_tsv(input_path)[:args.max_records]
    if not rows:
        raise ValueError(f'No completion records were parsed from {input_path}')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = Path(args.output) if args.output else Path(f'structure_results_{timestamp}.tsv')
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else Path(f'pdb_outputs_{timestamp}')
    api_key = os.getenv(args.api_key_env)
    evaluator = evaluator_factory(args.model_id, api_key)
    successful, failed = evaluate_rows(
        rows, evaluator, output_path, pdb_dir, args.fail_fast
    )
    print(f'completed {successful} records; {failed} failed')
    print('results saved to', output_path.resolve())
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
