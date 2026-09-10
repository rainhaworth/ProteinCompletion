# ProteinCompletion

This repository contains the code for our ISMB 2026 submission, "Adjacent Token Prediction for Protein Sequence Motif Scaffolding." 

## Overview

Relevant executable scripts
- `train.py`: full training run from scratch with checkpointing
- `eval-completion.py`: motif scaffolding experiments
- `generate.py`: generate completions for FASTA sequences or binding-site TSVs
- `eval.py`: measure ATP adjacent-token scoring or ESM masked-residue recovery
- `StructureEvaluation/structure_evaluator_driver.py`: main driver for running structure prediction and evaluation on generated sequences
- `StructureEvaluation/gen_combined_figs_driver.py`: driver for generating combined (multi-model) structure evaluation figures
- `StructureEvaluation/gen_figs_single_driver.py`: driver for figures from one model

Relevant non-executable scripts
- `utils/model_base.py`: defines all model components except prediction head
- `utils/model_bidirectional.py`: novel bidirectional model using ATP; increases size of LM head
- `utils/model_esmlike.py`: masked-language-model baseline using the same transformer backbone
- `utils/data.py`: all PyTorch `Dataset` definitions for preprocessing input data
- `utils/mask.py`: custom causal mask generation for ATP
- `utils/config.py`: defines `BaseConfig`, setting default values; config json files will always override these settings
- `utils/utils.py`: all other utility functions, notably including model and tokenizer loading
- `StructureEvaluation/structure_evaluator.py`: wraps the structure prediction model and computes pTM and pLDDT metrics
- `StructureEvaluation/generated_parser.py`: utilities for parsing generated and baseline sequence TSVs
- `StructureEvaluation/figure_generator.py`: plotting utilities for structure evaluation figures

## Usage

### Training
Training requires a single GPU with at least 48GB of VRAM and consumes a packed binary file rather than FASTA directly. Set the input and output paths near the top of `filter_uniref.py` and `pack_uniref.py`, then run those scripts before training. Create the checkpoint directory before starting.
```
python filter_uniref.py
python pack_uniref.py
mkdir weights
python train.py --data ./data/uniref50-packed.bin --tokenizer ./tokenizer-uniref.json --config config-medium --model_type atp --epochs 1 --save-every 20000 --save ./weights
```
Useful arguments:
- `--ckpt <filepath>`: Specify a checkpoint to load.
- `--model_type <atp|esm>`: Specify whether to train ATP or the masked-language-model baseline.
- `--data <filepath>`: Specify the packed binary training data.
- `--bsz <integer>`: Set the batch size.

A checkpoint restores model, optimizer, scheduler, and saved random-number-generator states. The shuffled data-loader position is not stored, so resumption is not an exact continuation of the previous sample order.

### Experiments
To perform our motif scaffolding experiments, run the command below, specifying your own checkpoint (e.g., `train-step220000`) and model type. Results will be saved to `out.tsv`.
```
python eval-completion.py --weights ./weights/<checkpoint>.pt --model_type <atp|esm> --data ./data/uniprot_sprot.fasta --output ./results/out.tsv --tokenizer ./tokenizer-uniref.json
```

The reported fully-visible reconstruction score is measured after completion and is not a conditional generation metric. Amino-acid composition entropy measures residue diversity in the generated region and is not model uncertainty.

For a small standalone generation run or a direct model-scoring run, use:

```
python generate.py --weights ./weights/<checkpoint>.pt --model_type <atp|esm> --data ./data/uniprot_sprot.fasta --tokenizer ./tokenizer-uniref.json --output ./results/generated.tsv
python eval.py --weights ./weights/<checkpoint>.pt --model_type <atp|esm> --data ./data/uniprot_sprot.fasta --tokenizer ./tokenizer-uniref.json --output ./results/scored.tsv
```

For ESM, `eval.py` hides a reproducible random subset of residues and reports recovery accuracy and conditional cross-entropy only at those positions. For ATP, it reports adjacent-token accuracy and cross-entropy from both prediction heads on visible sequences. These are diagnostic scoring tools; `eval-completion.py` remains the motif-scaffolding experiment.

The legacy ESM structure client requires Python 3.12, the optional structure dependencies, and an `ESM_API_KEY` environment variable. Install `requirements-structure.txt` in a separate environment because its ESM SDK and tokenizer requirements differ from the training environment. Failed records are written to the output with an error message, and successful predictions save their PDB files separately.

```
python -m pip install -r requirements-structure.txt
python StructureEvaluation/structure_evaluator_driver.py --input ./results/out.tsv --output ./results/structures.tsv --pdb-dir ./results/pdb
python StructureEvaluation/gen_figs_single_driver.py --input ./results/structures.tsv --original ./results/original_structures.tsv --model-label ATP --output-dir ./figures/atp
python StructureEvaluation/gen_combined_figs_driver.py --atp ./results/atp_structures.tsv --esm ./results/esm_structures.tsv --original ./results/original_structures.tsv --output-dir ./figures/combined
```

Run any script with `--help` for its complete argument list. Plotting previously generated structure results does not require an API key.

### Tests

```
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

