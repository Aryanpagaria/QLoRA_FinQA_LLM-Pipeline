# QLoRA  FinQA LLM-Pipeline

> A reproducible, production-oriented QLoRA fine-tuning pipeline for
> financial question answering using **Qwen2.5-3B-Instruct**, 4-bit NF4
> quantization, and LoRA adapters.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging
Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)

## Overview

This project implements an end-to-end parameter-efficient fine-tuning
pipeline for financial question answering.

The system adapts **Qwen/Qwen2.5-3B-Instruct** with QLoRA rather than
fully fine-tuning the base model.

### Core stack

-   Qwen/Qwen2.5-3B-Instruct
-   4-bit NF4 quantization
-   Double quantization
-   FP16 compute
-   LoRA / PEFT
-   PyTorch
-   Hugging Face Transformers
-   YAML-driven configuration

The project is designed as an engineering pipeline rather than a single
training notebook, with dedicated components for tokenization, model
construction, optimization, scheduling, training, checkpointing,
evaluation, and inference.

------------------------------------------------------------------------

## Architecture

``` text
Raw Financial Dataset
        |
        v
Validation / Preprocessing
        |
        v
Tokenization
        |
        v
Qwen2.5-3B-Instruct
4-bit NF4 Base Model
        +
LoRA Adapters
        |
        v
Trainer
  |       |       |
Optimizer Scheduler Checkpoint
        |
        v
Validation
        |
        v
Evaluator
  |      |       |
EM     Norm-EM  Numerical Accuracy
        |
        v
Inference
Load model + adapter once
        |
        v
Interactive financial QA
```

------------------------------------------------------------------------

## Model

### Base model

``` text
Qwen/Qwen2.5-3B-Instruct
```

### Quantization

``` text
4-bit
NF4
Double quantization: enabled
Compute dtype: float16
```

### LoRA

``` text
Rank:       16
Alpha:      32
Dropout:    0.05
Bias:       none
Task type:  CAUSAL_LM
```

Target modules:

``` text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The base model remains frozen while the LoRA parameters provide the
trainable domain adaptation.

------------------------------------------------------------------------

## Dataset

The current preprocessing pipeline produces:

``` text
Train:       6,251 samples
Validation:    883 samples
Test:        1,147 samples
Total:       8,281 samples
```

The model-ready representation contains:

``` text
input_ids
attention_mask
labels
```

The pipeline validates samples before they reach the training loop.

------------------------------------------------------------------------

## Training

Current training configuration:

  Parameter                  Value
  ----------------------- --------
  Epochs                        10
  Batch size                     1
  Gradient accumulation          8
  Learning rate               2e-4
  Weight decay                0.01
  Max gradient norm            1.0
  Warmup ratio                0.03
  Max sequence length          512
  Optimizer                  AdamW
  Scheduler                 Linear
  Precision                   FP16
  Seed                          42

Gradient accumulation provides a larger effective batch while keeping
the per-device batch small enough for constrained GPU memory.

------------------------------------------------------------------------

## Checkpoint engineering

Checkpointing is a first-class subsystem.

The checkpoint implementation has been tested for:

-   Configuration validation
-   Directory creation
-   PEFT model compatibility
-   LoRA-only state extraction
-   Frozen base-model exclusion
-   Optimizer state persistence
-   Scheduler state persistence
-   Python, NumPy, PyTorch and CUDA RNG persistence
-   Deterministic RNG restoration
-   Checkpoint schema validation
-   Atomic checkpoint writes
-   Checkpoint loading
-   LoRA restoration
-   Optimizer restoration
-   Scheduler restoration
-   Resume metadata restoration
-   Resume compatibility validation
-   Latest checkpoint discovery
-   Best checkpoint creation
-   Retention policy

The checkpoint stores the trainable LoRA state rather than unnecessarily
duplicating the frozen quantized base model.

### Resume safety

Training configuration fields that materially affect training are
checked before resume. Examples include:

``` text
batch size
gradient accumulation
maximum sequence length
optimizer
scheduler
precision
random seed
LoRA configuration
quantization configuration
```

Incompatible configurations are rejected instead of silently continuing
from an invalid state.

------------------------------------------------------------------------

## Trainer

`src/training/trainer.py` is responsible for:

-   Dataset/DataLoader construction
-   Batch validation
-   Device transfer
-   Forward and backward passes
-   Gradient accumulation
-   Gradient clipping
-   Optimizer stepping
-   Scheduler stepping
-   Global-step tracking
-   Epoch tracking
-   Validation
-   Best-metric tracking
-   Checkpoint integration
-   Resume integration

The trainer module has passed its module-level validation.

------------------------------------------------------------------------

## Evaluation

Evaluation is implemented in:

``` text
src/evaluation/evaluator.py
```

Supported metrics:

-   Exact Match
-   Normalized Exact Match
-   Numerical Accuracy

The evaluator has been validated for:

-   Configuration
-   Dataset conversion
-   DataLoader construction
-   Metric logic
-   Reproducibility
-   Tokenizer validation
-   Dataset preprocessing

Importantly, evaluation refuses to proceed when a valid trained
checkpoint is unavailable. This prevents meaningless or fabricated
evaluation results.

------------------------------------------------------------------------

## Inference

Inference is deliberately separated from training and evaluation.

The intended production flow is:

``` text
Start process
    |
    v
Load configuration
    |
    v
Load tokenizer
    |
    v
Load Qwen2.5-3B-Instruct
    |
    v
Load trained LoRA adapter
    |
    v
model.eval()
    |
    v
Interactive chat loop
    |
    +--> Question
    +--> Question
    +--> Question
    |
    v
Exit
```

The model is loaded **once per inference process**, not once per
question.

This makes interactive inference substantially more efficient than
repeatedly constructing the model.

------------------------------------------------------------------------

## Project structure

``` text
financial-lora-finetuning/
|
+-- configs/
|   +-- config.yaml
|   +-- checkpoint/checkpoint.yaml
|   +-- data/data.yaml
|   +-- evaluation/evaluation.yaml
|   +-- inference/inference.yaml
|   +-- logging/logging.yaml
|   +-- model/model.yaml
|   +-- training/training.yaml
|
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- reports/
|
+-- artifacts/
|   +-- checkpoints/
|   +-- best_model/
|   +-- lora_adapter/
|   +-- evaluation/
|   +-- logs/
|
+-- src/
|   +-- evaluation/
|   |   +-- evaluator.py
|   |
|   +-- inference/
|   |   +-- inference.py
|   |
|   +-- training/
|   |   +-- model.py
|   |   +-- optimizer.py
|   |   +-- scheduler.py
|   |   +-- trainer.py
|   |   +-- checkpoint.py
|   |   +-- tokenizer.py
|   |
|   +-- utils/
|       +-- config_loader.py
|
+-- tests/
+-- requirements.txt
+-- README.md
+-- LICENSE
```

------------------------------------------------------------------------

## Configuration

The project is configuration-driven. Major settings live in:

``` text
configs/model/model.yaml
configs/training/training.yaml
configs/checkpoint/checkpoint.yaml
configs/evaluation/evaluation.yaml
configs/inference/inference.yaml
configs/logging/logging.yaml
configs/data/data.yaml
```

This keeps experiment parameters out of the implementation and makes
runs easier to reproduce and audit.

------------------------------------------------------------------------

## Validation commands

Validate the implemented modules before running the full experiment:

``` powershell
python -m src.training.trainer
python -m src.training.checkpoint
python -m src.evaluation.evaluator
```

After a trained checkpoint exists:

``` powershell
python -c "from src.evaluation.evaluator import evaluate; print(evaluate())"
```

Inference:

``` powershell
python -m src.inference.inference
```

------------------------------------------------------------------------

## Reproducibility

The configured seed is:

``` text
42
```

Checkpoint state captures:

``` text
Python RNG
NumPy RNG
PyTorch CPU RNG
CUDA RNG when CUDA is available
```

Together with optimizer and scheduler state, this allows the training
process to restore substantially more state than model parameters alone.

------------------------------------------------------------------------

## Engineering principles

### No silent incompatibility

A materially different training configuration must not silently resume
from an existing checkpoint.

### No fake evaluation

Evaluation must fail when the required trained model is unavailable.

### Adapter-focused persistence

The frozen quantized base model should not be unnecessarily serialized
into every training checkpoint.

### Configuration-driven experiments

Important experiment parameters are defined in YAML.

### Separation of concerns

``` text
model.py       -> model construction
tokenizer.py   -> tokenizer configuration
optimizer.py   -> optimizer
scheduler.py   -> learning-rate scheduling
trainer.py     -> training loop
checkpoint.py  -> persistence and resume
evaluator.py   -> evaluation
inference.py   -> generation and chat
```

------------------------------------------------------------------------

## What makes the project interesting?

This is intentionally more than:

``` text
load model -> fine-tune -> save model
```

It implements the complete lifecycle of a parameter-efficient
fine-tuning experiment:

``` text
Data
 |
 v
Validation
 |
 v
Tokenization
 |
 v
QLoRA model construction
 |
 v
Optimizer + Scheduler
 |
 v
Training
 |
 v
Validated Checkpoint
 |
 v
Safe Resume
 |
 v
Evaluation
 |
 v
Inference
```

The engineering emphasis is on **reproducibility, checkpoint integrity,
evaluation correctness, and production-oriented inference**, not just
obtaining a fine-tuned model.



------------------------------------------------------------------------

## Planned experiment

``` text
Base:
Qwen/Qwen2.5-3B-Instruct

Method:
QLoRA

Quantization:
4-bit NF4

LoRA:
r=16
alpha=32
dropout=0.05

Training:
10 epochs
batch size=1
gradient accumulation=8
learning rate=2e-4
max sequence length=512
```

The final experiment should report:

-   Base-model performance
-   Fine-tuned-model performance
-   Exact Match
-   Normalized Exact Match
-   Numerical Accuracy
-   Validation loss
-   Test metrics
-   Training time
-   GPU memory usage
-   Checkpoint size
-   Inference latency
-   Generation throughput

------------------------------------------------------------------------

## Limitations

This is a research and engineering project, not a financial advisory
system.

A strong benchmark score does not guarantee factual correctness on
unseen financial questions or suitability for real-world regulated
financial decision-making.

The trained model should be evaluated against the intended deployment
scenario before use in production.

------------------------------------------------------------------------

## Contributing

Contributions are welcome.

Please preserve:

1.  Separation between model, training, checkpoint, evaluation, and
    inference components.
2.  Configuration validation for new settings.
3.  Checkpoint compatibility guarantees.
4.  Reproducibility controls.
5.  Tests for new stateful behavior.

Do not commit:

-   API keys
-   private datasets
-   credentials
-   large trained model weights
-   generated artifacts that should remain local

------------------------------------------------------------------------

## License

See `LICENSE`.

------------------------------------------------------------------------

## Author

**Aryan Pagaria**

Built as an AI/ML engineering and research project focused on
parameter-efficient LLM fine-tuning, financial question answering,
reproducible training, checkpoint engineering, evaluation, and
inference.
