"""
FinQA benchmark evaluation runner.

Evaluates:
    1. Qwen/Qwen2.5-3B-Instruct
    2. Qwen/Qwen2.5-3B-Instruct + trained LoRA adapter

The benchmark evaluates answer generation rather than FinQA program
generation.

Metrics:
    - Normalized Exact Match
    - Numerical Accuracy
    - Invalid Prediction Rate
    - Average Generation Latency
    - Median Generation Latency
    - Generated Tokens
    - Generation Throughput

Per-example predictions and aggregate metrics are saved under:
    evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generation import generate_response
from src.inference.inference import (
    _load_base_model,
    _load_inference_config,
    _load_lora_adapter,
    _load_tokenizer,
    _set_reproducibility_seed,
)
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleResult:
    """Evaluation result for one FinQA example."""

    example_id: str
    question: str
    gold_answer: str
    prediction: str

    normalized_gold: str
    normalized_prediction: str

    gold_number: float | None
    predicted_number: float | None

    exact_match: bool
    numerical_accuracy: bool
    invalid_prediction: bool

    input_tokens: int
    generated_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate benchmark result."""

    benchmark: str
    split: str
    model_type: str
    base_model: str
    adapter_used: bool

    evaluated_examples: int

    exact_match: float
    numerical_accuracy: float
    invalid_prediction_rate: float

    average_latency_seconds: float
    median_latency_seconds: float

    total_generated_tokens: int
    average_generated_tokens: float
    generation_tokens_per_second: float

    evaluation_timestamp_utc: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/QLoRA on the FinQA benchmark."
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "dev", "test"),
        help="FinQA split to evaluate.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional number of examples for smoke tests.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--model-type",
        default="local-lora",
        choices=("base", "local-lora"),
        help=(
            "Evaluate either the base Qwen model or "
            "the trained local LoRA model."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for result files.",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""

    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError(
            "--max-examples must be a positive integer."
        )

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be a positive integer."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run model evaluation in the Colab GPU environment."
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_finqa_split(
    split: str,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """
    Load an official FinQA JSON split.

    Files expected:
        evaluation/datasets/finqa/train.json
        evaluation/datasets/finqa/dev.json
        evaluation/datasets/finqa/test.json
    """

    if split not in {"train", "dev", "test"}:
        raise ValueError(
            "split must be one of: train, dev, test."
        )

    dataset_file = (
        PROJECT_ROOT
        / "evaluation"
        / "datasets"
        / "finqa"
        / f"{split}.json"
    )

    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"FinQA dataset file was not found: {dataset_file}"
        )

    try:
        with dataset_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {dataset_file}"
        ) from exc

    if not isinstance(records, list):
        raise RuntimeError(
            f"FinQA file must contain a JSON list: {dataset_file}"
        )

    if max_examples is not None:
        records = records[:max_examples]

    if not records:
        raise RuntimeError(
            f"FinQA split '{split}' contains no examples."
        )

    return records


# ---------------------------------------------------------------------------
# Answer handling
# ---------------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """
    Normalize an answer for diagnostic exact-match comparison.

    This is intentionally a lightweight diagnostic metric and is not
    claimed to be the official FinQA program/execution metric.
    """

    if text is None:
        return ""

    normalized = str(text).lower().strip()

    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"[.!?]+$",
        "",
        normalized,
    )

    return normalized.strip()


def parse_number(
    text: Any,
) -> float | None:
    """
    Extract the first numeric value from a prediction.

    Supports:
        - integers
        - decimals
        - negative numbers
        - comma-separated numbers
        - percentages
        - scientific notation
    """

    if text is None:
        return None

    cleaned = str(text).strip()

    if not cleaned:
        return None

    percentage_match = re.search(
        r"[-+]?\d[\d,]*(?:\.\d+)?\s*%",
        cleaned,
    )

    if percentage_match:
        value = (
            percentage_match.group(0)
            .replace("%", "")
            .replace(",", "")
            .strip()
        )

        try:
            return float(value) / 100.0
        except ValueError:
            return None

    number_match = re.search(
        r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if number_match is None:
        return None

    value = number_match.group(0).replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def numbers_are_equivalent(
    predicted: float | None,
    gold: float | None,
) -> bool:
    """Compare numeric answers with a small tolerance."""

    if predicted is None or gold is None:
        return False

    if not math.isfinite(predicted) or not math.isfinite(gold):
        return False

    return math.isclose(
        predicted,
        gold,
        rel_tol=1e-4,
        abs_tol=1e-6,
    )


# ---------------------------------------------------------------------------
# FinQA example extraction
# ---------------------------------------------------------------------------


def extract_question(
    example: dict[str, Any],
) -> str:
    """Extract the FinQA question."""

    qa = example.get("qa")

    if not isinstance(qa, dict):
        raise ValueError(
            "FinQA example does not contain a valid qa object."
        )

    question = qa.get("question")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "FinQA example does not contain a valid question."
        )

    return question.strip()


def extract_gold_answer(
    example: dict[str, Any],
) -> str:
    """
    Extract the human-readable FinQA answer.

    IMPORTANT:
    The current QLoRA model was trained to generate qa.answer,
    not qa.program. Therefore the answer-generation benchmark
    uses qa.answer as its primary target.

    qa.exe_ans remains part of the source dataset and is useful
    for future program/execution evaluation.
    """

    qa = example.get("qa")

    if not isinstance(qa, dict):
        raise ValueError(
            "FinQA example does not contain a valid qa object."
        )

    answer = qa.get("answer")

    if answer is None:
        raise ValueError(
            "FinQA example does not contain qa.answer."
        )

    answer = str(answer).strip()

    if not answer:
        raise ValueError(
            "FinQA example contains an empty qa.answer."
        )

    return answer


def extract_example_id(
    example: dict[str, Any],
    index: int,
) -> str:
    """Return a stable example identifier."""

    for key in (
        "id",
        "example_id",
        "uid",
    ):
        value = example.get(key)

        if value is not None:
            return str(value)

    return f"finqa-{index:06d}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _coerce_context_fields(
    example: dict[str, Any],
) -> dict[str, str]:
    """
    Convert raw FinQA JSON fields into the same string representation
    expected by the training preprocessing pipeline.

    Training cleaning converts:
        pre_text  -> paragraphs joined by '\\n\\n'
        post_text -> paragraphs joined by '\\n\\n'

    Raw FinQA JSON stores these fields as lists.
    """

    pre_text = example.get("pre_text", "")
    post_text = example.get("post_text", "")
    table = example.get("table", "")

    if isinstance(pre_text, list):
        pre_text = "\n\n".join(
            str(item)
            for item in pre_text
        )
    else:
        pre_text = str(pre_text)

    if isinstance(post_text, list):
        post_text = "\n\n".join(
            str(item)
            for item in post_text
        )
    else:
        post_text = str(post_text)

    if isinstance(table, list):
        rows: list[str] = []

        for row in table:
            if isinstance(row, list):
                rows.append(
                    " | ".join(
                        str(cell)
                        for cell in row
                    )
                )
            else:
                rows.append(str(row))

        table = "\n".join(rows)

    else:
        table = str(table)

    return {
        "pre_text": pre_text,
        "table": table,
        "post_text": post_text,
    }

def _truncate_text(
    text: str,
    max_tokens: int,
    tokenizer: Any,
) -> str:
    """
    Truncate text using the same Qwen tokenizer mechanism
    used by the training preprocessing pipeline.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip()

    if not text:
        return ""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_ids = token_ids[:max_tokens]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

def build_prompt(
    example: dict[str, Any],
    tokenizer: Any,
) -> str:
    """
    Build the exact training-style FinQA prompt.

    The model receives:
        system instruction
        financial context
        question

    The following are NEVER inserted:
        qa.answer
        qa.exe_ans
        qa.program
    """

    question = extract_question(example)

    fields = _coerce_context_fields(example)

    background = _truncate_text(
        fields["pre_text"],
        150,
        tokenizer,
    )

    financial_table = _truncate_text(
        fields["table"],
        180,
        tokenizer,
    )

    additional_context = _truncate_text(
        fields["post_text"],
        40,
        tokenizer,
    )

    sections: list[str] = []

    if background:
        sections.append(
            "Background:\n"
            f"{background}"
        )

    if financial_table:
        sections.append(
            "Financial Table:\n"
            f"{financial_table}"
        )

    if additional_context:
        sections.append(
            "Additional Context:\n"
            f"{additional_context}"
        )

    context = "\n\n".join(sections)

    system_prompt = (
        "You are a highly accurate financial AI assistant.\n\n"
        "Answer financial questions using ONLY the provided "
        "financial context.\n\n"
        "If the context does not contain enough information, "
        "say that the answer cannot be determined from the "
        "provided information.\n\n"
        "Do not fabricate facts."
    )

    user_prompt = (
        "Financial Context:\n\n"
        f"{context}\n\n"
        "Question:\n\n"
        f"{question}\n\n"
        "Answer the question using only the provided financial context."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------


def evaluate_example(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: dict[str, Any],
    index: int,
    max_new_tokens: int,
) -> ExampleResult:
    """Generate and evaluate one FinQA example."""

    question = extract_question(example)
    gold_answer = extract_gold_answer(example)

    prompt = build_prompt(
        example,
        tokenizer,
    )

    start_time = time.perf_counter()

    generation = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    latency_seconds = (
        time.perf_counter()
        - start_time
    )

    prediction = generation.text

    normalized_gold = normalize_text(
        gold_answer
    )

    normalized_prediction = normalize_text(
        prediction
    )

    gold_number = parse_number(
        gold_answer
    )

    predicted_number = parse_number(
        prediction
    )

    exact_match = (
        normalized_prediction
        == normalized_gold
    )

    numerical_accuracy = numbers_are_equivalent(
        predicted=predicted_number,
        gold=gold_number,
    )

    invalid_prediction = (
        not exact_match
        and not numerical_accuracy
        and predicted_number is None
    )

    return ExampleResult(
        example_id=extract_example_id(
            example=example,
            index=index,
        ),
        question=question,
        gold_answer=gold_answer,
        prediction=prediction,
        normalized_gold=normalized_gold,
        normalized_prediction=normalized_prediction,
        gold_number=gold_number,
        predicted_number=predicted_number,
        exact_match=exact_match,
        numerical_accuracy=numerical_accuracy,
        invalid_prediction=invalid_prediction,
        input_tokens=generation.input_tokens,
        generated_tokens=generation.generated_tokens,
        latency_seconds=latency_seconds,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def calculate_benchmark_metrics(
    results: list[ExampleResult],
    model_type: str,
    split: str,
    base_model: str,
    adapter_used: bool,
) -> BenchmarkResult:
    """Calculate aggregate benchmark metrics."""

    if not results:
        raise RuntimeError(
            "Cannot calculate metrics from zero results."
        )

    count = len(results)

    exact_match_count = sum(
        result.exact_match
        for result in results
    )

    numerical_accuracy_count = sum(
        result.numerical_accuracy
        for result in results
    )

    invalid_prediction_count = sum(
        result.invalid_prediction
        for result in results
    )

    latencies = [
        result.latency_seconds
        for result in results
    ]

    total_generated_tokens = sum(
        result.generated_tokens
        for result in results
    )

    total_latency = sum(latencies)

    return BenchmarkResult(
        benchmark="FinQA",
        split=split,
        model_type=model_type,
        base_model=base_model,
        adapter_used=adapter_used,
        evaluated_examples=count,
        exact_match=(
            exact_match_count / count
        ),
        numerical_accuracy=(
            numerical_accuracy_count / count
        ),
        invalid_prediction_rate=(
            invalid_prediction_count / count
        ),
        average_latency_seconds=(
            total_latency / count
        ),
        median_latency_seconds=statistics.median(
            latencies
        ),
        total_generated_tokens=(
            total_generated_tokens
        ),
        average_generated_tokens=(
            total_generated_tokens / count
        ),
        generation_tokens_per_second=(
            total_generated_tokens / total_latency
            if total_latency > 0
            else 0.0
        ),
        evaluation_timestamp_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_results(
    example_results: list[ExampleResult],
    benchmark_result: BenchmarkResult,
    output_prefix: str,
) -> tuple[Path, Path]:
    """Save predictions and metrics."""

    results_directory = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        results_directory
        / f"{output_prefix}_predictions.jsonl"
    )

    metrics_path = (
        results_directory
        / f"{output_prefix}_metrics.json"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in example_results:
            file.write(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                )
                + "\n"
            )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(benchmark_result),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        predictions_path,
        metrics_path,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(
    result: BenchmarkResult,
) -> None:
    """Print benchmark summary."""

    print()
    print("=" * 80)
    print("FINQA BENCHMARK RESULTS")
    print("=" * 80)

    print(
        f"Model Type             : "
        f"{result.model_type}"
    )

    print(
        f"Base Model             : "
        f"{result.base_model}"
    )

    print(
        f"LoRA Adapter Used      : "
        f"{result.adapter_used}"
    )

    print(
        f"Split                  : "
        f"{result.split}"
    )

    print(
        f"Examples               : "
        f"{result.evaluated_examples}"
    )

    print(
        f"Exact Match            : "
        f"{result.exact_match:.4%}"
    )

    print(
        f"Numerical Accuracy     : "
        f"{result.numerical_accuracy:.4%}"
    )

    print(
        f"Invalid Prediction     : "
        f"{result.invalid_prediction_rate:.4%}"
    )

    print(
        f"Average Latency        : "
        f"{result.average_latency_seconds:.4f}s"
    )

    print(
        f"Median Latency         : "
        f"{result.median_latency_seconds:.4f}s"
    )

    print(
        f"Average Generated      : "
        f"{result.average_generated_tokens:.2f} tokens"
    )

    print(
        f"Generation Throughput  : "
        f"{result.generation_tokens_per_second:.2f} tokens/s"
    )

    print("=" * 80)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_evaluation(
    model_type: str,
) -> tuple[Any, Any, torch.device, str, bool]:
    """
    Load either:

        base
            Qwen/Qwen2.5-3B-Instruct

        local-lora
            Qwen/Qwen2.5-3B-Instruct + exported LoRA adapter
    """

    if model_type not in {
        "base",
        "local-lora",
    }:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    config = _load_inference_config()

    if not config["inference"]["enabled"]:
        raise RuntimeError(
            "Inference is disabled in the inference configuration."
        )

    _set_reproducibility_seed(
        config["inference"]["reproducibility"]["seed"]
    )

    tokenizer = _load_tokenizer(
        config
    )

    model = _load_base_model(
        config
    )

    adapter_used = False

    if model_type == "local-lora":
        model = _load_lora_adapter(
            model=model,
            config=config,
        )
        adapter_used = True

    device = next(
        model.parameters()
    ).device

    base_model_name = config[
        "model"
    ][
        "name"
    ]

    # Safety checks.
    if model_type == "base":

        if hasattr(
            model,
            "peft_config",
        ):
            raise RuntimeError(
                "Base evaluation model unexpectedly contains "
                "a PEFT configuration."
            )

    else:

        if not hasattr(
            model,
            "peft_config",
        ):
            raise RuntimeError(
                "LoRA evaluation model does not contain "
                "a PEFT configuration."
            )

    print("=" * 80)
    print("EVALUATION MODEL READY")
    print("=" * 80)

    print(
        f"Model Type      : {model_type}"
    )

    print(
        f"Base Model      : {base_model_name}"
    )

    print(
        f"LoRA Adapter    : {adapter_used}"
    )

    print(
        f"Device          : {device}"
    )

    print("=" * 80)

    return (
        model,
        tokenizer,
        device,
        base_model_name,
        adapter_used,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the FinQA benchmark."""

    args = parse_arguments()

    validate_arguments(args)

    set_seed()

    print("=" * 80)
    print("FINQA EVALUATION")
    print("=" * 80)

    print(
        f"Model Type      : {args.model_type}"
    )

    print(
        f"Split           : {args.split}"
    )

    if args.max_examples is not None:
        print(
            f"Max Examples    : {args.max_examples}"
        )

    print("=" * 80)

    dataset = load_finqa_split(
        split=args.split,
        max_examples=args.max_examples,
    )

    print(
        f"FinQA examples  : {len(dataset)}"
    )

    (
        model,
        tokenizer,
        device,
        base_model_name,
        adapter_used,
    ) = load_model_for_evaluation(
        model_type=args.model_type
    )

    example_results: list[ExampleResult] = []

    for index, example in enumerate(dataset):

        result = evaluate_example(
            model=model,
            tokenizer=tokenizer,
            device=device,
            example=example,
            index=index,
            max_new_tokens=args.max_new_tokens,
        )

        example_results.append(result)

        current_count = len(
            example_results
        )

        current_exact_match = (
            sum(
                item.exact_match
                for item in example_results
            )
            / current_count
        )

        current_numerical_accuracy = (
            sum(
                item.numerical_accuracy
                for item in example_results
            )
            / current_count
        )

        print(
            f"[{current_count}/{len(dataset)}] "
            f"EM={current_exact_match:.4%} "
            f"NumAcc={current_numerical_accuracy:.4%} "
            f"Latency={result.latency_seconds:.2f}s"
        )

    benchmark_result = calculate_benchmark_metrics(
        results=example_results,
        model_type=args.model_type,
        split=args.split,
        base_model=base_model_name,
        adapter_used=adapter_used,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_prefix = (
        args.output_prefix
        or (
            f"finqa_"
            f"{args.model_type}_"
            f"{args.split}_"
            f"{timestamp}"
        )
    )

    predictions_path, metrics_path = save_results(
        example_results=example_results,
        benchmark_result=benchmark_result,
        output_prefix=output_prefix,
    )

    print_summary(
        benchmark_result
    )

    print(
        f"Predictions saved : {predictions_path}"
    )

    print(
        f"Metrics saved     : {metrics_path}"
    )


if __name__ == "__main__":
    main()