# ProteinCompletion

This repository contains the code for our ISMB 2026 submission, "Adjacent Token Prediction for Protein Sequence Motif Scaffolding." 

## Overview

Relevant executable scripts
- `train.py`: full training run from scratch with checkpointing
- `eval-completion.py`: motif scaffolding experiments
- `StructureEvaluation/structure_evaluator_driver.py`: main driver for running structure prediction and evaluation on generated sequences
- `StructureEvaluation/gen_combined_figs_driver.py`: driver for generating combined (multi-model) structure evaluation figures

Relevant non-executable scripts
- `utils/model_base.py`: defines all model components except prediction head
- `utils/model_bidirectional.py`: novel bidirectional model using ATP; increases size of LM head
- `utils/model_esmlike.py`: bidirectional model based on ESM3
- `utils/data.py`: all PyTorch `Dataset` definitions for preprocessing input data
- `utils/mask.py`: custom causal mask generation for ATP
- `utils/config.py`: defines `BaseConfig`, setting default values; config json files will always override these settings
- `utils/utils.py`: all other utility functions, notably including model and tokenizer loading
- `StructureEvaluation/structure_evaluator.py`: wraps the structure prediction model and computes pTM and pLDDT metrics
- `StructureEvaluation/generated_parser.py`: utilities for parsing generated and baseline sequence TSVs
- `StructureEvaluation/figure_generator.py`: plotting utilities for structure evaluation figures

## Usage

### Installation

Python 3.11 or 3.12 is required by the pinned dependencies.

```
python -m venv .venv
python -m pip install -r requirements.txt
```

For development and tests, install `requirements-dev.txt`. Structure evaluation uses a separate environment because ESM requires a different tokenizers stack.

```
python -m venv .venv-structure
# Activate .venv-structure, then run
python -m pip install -r requirements-structure.txt
```

### Training

Training the medium configuration requires a single GPU with approximately 48GB of VRAM. Download and decompress the desired FASTA file, then provide it with `--data`. The loader automatically distinguishes UniRef headers from ordinary FASTA headers. This can be overridden with `--data-format uniref` or `--data-format fasta`.

The save directory is created automatically. CUDA training uses automatic mixed precision, while CPU training uses FP32.

```
python train.py --data ./data/uniref50.fasta --data-format uniref --model_type atp --max-samples 600000 --epochs 5 --save-every 20000 --save ./weights/
```

Useful arguments:
- `--ckpt <filepath>`: Specify a checkpoint to load.
- `--model_type <atp|esm>`: Select the ATP or ESM-like model.
- `--data <filepath>`: Specify the training FASTA or TSV.
- `--data-format <auto|fasta|uniref>`: Select FASTA parsing behavior.

### Experiments

To perform motif scaffolding experiments, specify a checkpoint and its model type. Nucleus sampling is used by default. Use `--sample greedy` for greedy decoding or `--p` to change the nucleus threshold.

```
python eval-completion.py --weights ./weights/<checkpoint>.pt --model_type <atp|esm> --data ./data/uniprot_sprot.fasta --output ./results/out.tsv
```

The reported full-sequence perplexity is measured after completion and is not a conditional generation metric. Amino-acid composition entropy measures residue diversity in the generated region and is not model uncertainty.

Structure prediction requires the optional structure dependencies and an `ESM_API_KEY` environment variable. Structure evaluation is implemented in `StructureEvaluation/structure_evaluator_driver.py`, and combined figures are generated with `StructureEvaluation/gen_combined_figs_driver.py`.

### Tests

```
python -m pytest -q
```

