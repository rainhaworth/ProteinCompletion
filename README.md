# ProteinCompletion

## Overview

- `train.py`: full training run from scratch with checkpointing
- `generate.py`: generate sample proteins
- `eval.py`: basic perplexity and prediction accuracy evaluation
- `utils/model_base.py`: defines all model components except prediction head
- `utils/model_bidirectional.py`: novel bidirectional causal masking model; increases size of LM head
- `utils/model_esmlike.py`: bidirectional model based on ESM3
- `utils/data.py`: all PyTorch `Dataset` definitions for preprocessing input data
- `utils/mask.py`: custom causal mask generation
- `utils/config.py`: defines `BaseConfig`, setting default values; config json files will always override these settings
- `utils/utils.py`: all other utility functions, notably including model and tokenizer loading
- 'StructureEvaluation/structure_evaluator_driver.py': 
- 'StructureEvaluation/structure_evaluator.py': 
- 'StructureEvaluation/generated_parser.py': 
- 'StructureEvaluation/figure_generator.py': 
- 'StructureEvaluation/gen_combined_figs_driver.py': 
- 'StructureEvaluation/gen_figs_single_driver.py': 
