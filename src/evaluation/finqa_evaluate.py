"""
FinQA benchmark evaluation runner.

Evaluates a local causal language model on the FinQA test split and records:
- Exact Match
- Numerical Accuracy
- Invalid Prediction Rate
- Average Generation Latency
- Generated Tokens
- Generation Throughput

The runner is deterministic and stores both per-example predictions and
aggregate benchmark results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generation import generate_response
from src.inference.inference import load_inference_stack
from src.utils.seed import set_seed


@dataclass(frozen=True)
class ExampleResult:
    """
    Evaluation result for a single FinQA example.
    """

    example_id: str
    question: str
    gold_answer: str
    prediction: str
    normalized_gold: str
    normalized_prediction: str
    predicted_number: float | None
    gold_number: float | None
    exact_match: bool
    numerical_accuracy: bool
    invalid_prediction: bool
    input_tokens: int
    generated_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Aggregate FinQA benchmark result.
    """

    benchmark: str
    split: str
    model_type: str
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate a local model on the FinQA test benchmark."
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="FinQA dataset split to evaluate.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for development runs.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per example.",
    )

    parser.add_argument(
        "--model-type",
        default="local-lora",
        choices=("base", "local-lora"),
        help="Model configuration to evaluate.",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output filename prefix.",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Validate command-line arguments.
    """

    if (
        args.max_examples is not None
        and args.max_examples <= 0
    ):
        raise ValueError(
            "--max-examples must be a positive integer."
        )

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be a positive integer."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run the full FinQA benchmark "
            "on the Colab GPU environment rather than the CPU-only "
            "Windows development environment."
        )


def load_finqa_split(
    split: str,
    max_examples: int | None,
) -> Any:
    """
    Load FinQA directly from the official dataset files.

    The loader avoids the deprecated Hugging Face dataset-script
    mechanism and reads the JSON split files directly.
    """

    if split not in {
        "train",
        "dev",
        "test",
    }:
        raise ValueError(
            "split must be one of: train, dev, test."
        )

    dataset_directory = (
        Path("evaluation")
        / "datasets"
        / "finqa"
    )

    dataset_file = (
        dataset_directory
        / f"{split}.json"
    )

    if not dataset_file.exists():
        raise FileNotFoundError(
            "FinQA dataset file was not found: "
            f"{dataset_file}. "
            "Download the official FinQA JSON files into "
            f"{dataset_directory}."
        )

    try:
        with dataset_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in FinQA dataset file: {dataset_file}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read FinQA dataset file: {dataset_file}"
        ) from exc

    if not isinstance(records, list):
        raise RuntimeError(
            f"FinQA dataset file must contain a JSON list: {dataset_file}"
        )

    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError(
                "max_examples must be greater than zero."
            )

        records = records[
            :min(
                max_examples,
                len(records),
            )
        ]

    if not records:
        raise RuntimeError(
            f"FinQA split '{split}' contains no examples."
        )

    return records


def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for exact-match comparison.
    """

    if not isinstance(text, str):
        return ""

    normalized = text.lower().strip()

    normalized = normalized.replace(
        "$",
        "",
    )

    normalized = normalized.replace(
        ",",
        "",
    )

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
    text: str,
) -> float | None:
    """
    Extract a numeric answer from generated text.

    Handles:
    - integers
    - decimals
    - percentages
    - negative values
    - comma-separated values
    - simple scientific notation
    """

    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    if not cleaned:
        return None

    percentage_match = re.search(
        r"[-+]?\d[\d,]*(?:\.\d+)?\s*%",
        cleaned,
    )

    if percentage_match:
        value = percentage_match.group(0)
        value = value.replace(
            "%",
            "",
        ).replace(
            ",",
            "",
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

    value = number_match.group(0).replace(
        ",",
        "",
    )

    try:
        return float(value)
    except ValueError:
        return None


def numbers_are_equivalent(
    predicted: float | None,
    gold: float | None,
) -> bool:
    """
    Compare numeric values using a relative and absolute tolerance.
    """

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


def extract_question(
    example: dict[str, Any],
) -> str:
    """
    Extract the natural-language question from a FinQA example.
    """

    question = example.get("qa", {}).get(
        "question",
        "",
    )

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "FinQA example does not contain a valid QA question."
        )

    return question.strip()


def extract_gold_answer(
    example: dict[str, Any],
) -> str:
    """
    Extract the gold answer from a FinQA example.
    """

    answer = example.get("qa", {}).get(
        "answer",
        "",
    )

    if isinstance(answer, (int, float)):
        return str(answer)

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "FinQA example does not contain a valid gold answer."
        )

    return answer.strip()


def extract_example_id(
    example: dict[str, Any],
    index: int,
) -> str:
    """
    Create a stable identifier for a benchmark example.
    """

    for key in (
        "id",
        "example_id",
        "uid",
    ):
        value = example.get(key)

        if value is not None:
            return str(value)

    return f"finqa-{index:06d}"


def build_prompt(
    example: dict[str, Any],
) -> str:
    """
    Build a concise instruction prompt for FinQA.

    The benchmark evaluates answer generation. It does not inject the gold
    answer or gold program into the prompt.
    """

    question = extract_question(
        example
    )

    return (
        "You are a financial question answering assistant.\n\n"
        "Answer the following question using the provided financial "
        "context when available.\n\n"
        f"Question: {question}\n\n"
        "Give the final answer clearly. "
        "If the answer is numerical, provide the numerical result "
        "and its unit or percentage when applicable."
    )


def calculate_exact_match(
    prediction: str,
    gold_answer: str,
) -> bool:
    """
    Calculate normalized exact-match accuracy.
    """

    return (
        normalize_text(prediction)
        == normalize_text(gold_answer)
    )


def evaluate_example(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: dict[str, Any],
    index: int,
    max_new_tokens: int,
) -> ExampleResult:
    """
    Generate and evaluate one FinQA example.
    """

    question = extract_question(
        example
    )

    gold_answer = extract_gold_answer(
        example
    )

    prompt = build_prompt(
        example
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

    exact_match = calculate_exact_match(
        prediction=prediction,
        gold_answer=gold_answer,
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
        predicted_number=predicted_number,
        gold_number=gold_number,
        exact_match=exact_match,
        numerical_accuracy=numerical_accuracy,
        invalid_prediction=invalid_prediction,
        input_tokens=generation.input_tokens,
        generated_tokens=generation.generated_tokens,
        latency_seconds=latency_seconds,
    )


def calculate_benchmark_metrics(
    results: list[ExampleResult],
    model_type: str,
    split: str,
) -> BenchmarkResult:
    """
    Calculate aggregate benchmark metrics.
    """

    if not results:
        raise RuntimeError(
            "Cannot calculate benchmark metrics from zero results."
        )

    example_count = len(
        results
    )

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

    total_latency = sum(
        latencies
    )

    average_latency = (
        total_latency
        / example_count
    )

    median_latency = statistics.median(
        latencies
    )

    average_generated_tokens = (
        total_generated_tokens
        / example_count
    )

    generation_tokens_per_second = (
        total_generated_tokens / total_latency
        if total_latency > 0
        else 0.0
    )

    return BenchmarkResult(
        benchmark="FinQA",
        split=split,
        model_type=model_type,
        evaluated_examples=example_count,
        exact_match=exact_match_count / example_count,
        numerical_accuracy=(
            numerical_accuracy_count
            / example_count
        ),
        invalid_prediction_rate=(
            invalid_prediction_count
            / example_count
        ),
        average_latency_seconds=average_latency,
        median_latency_seconds=median_latency,
        total_generated_tokens=total_generated_tokens,
        average_generated_tokens=average_generated_tokens,
        generation_tokens_per_second=(
            generation_tokens_per_second
        ),
        evaluation_timestamp_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


def save_results(
    example_results: list[ExampleResult],
    benchmark_result: BenchmarkResult,
    output_prefix: str,
) -> tuple[Path, Path]:
    """
    Save per-example results and aggregate metrics.
    """

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

    return predictions_path, metrics_path


def print_summary(
    result: BenchmarkResult,
) -> None:
    """
    Print aggregate benchmark metrics.
    """

    print()
    print("=" * 80)
    print("FINQA BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"Model                  : {result.model_type}"
    )
    print(
        f"Split                  : {result.split}"
    )
    print(
        f"Examples               : {result.evaluated_examples}"
    )
    print(
        f"Exact Match            : {result.exact_match:.4%}"
    )
    print(
        f"Numerical Accuracy     : {result.numerical_accuracy:.4%}"
    )
    print(
        f"Invalid Prediction     : {result.invalid_prediction_rate:.4%}"
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


def load_model_for_evaluation(
    model_type: str,
) -> tuple[Any, Any, torch.device]:
    """
    Load the requested model configuration.

    The base/LoRA distinction is currently controlled by the inference
    configuration. The local LoRA stack is loaded through the project's
    existing inference entry point.
    """

    if model_type not in {
        "base",
        "local-lora",
    }:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    config, tokenizer, model = (
        load_inference_stack()
    )

    device = next(
        model.parameters()
    ).device

    return (
        model,
        tokenizer,
        device,
    )


def main() -> None:
    """
    Execute the FinQA benchmark.
    """

    args = parse_arguments()

    validate_arguments(
        args
    )

    set_seed()

    print(
        "Loading FinQA dataset..."
    )

    dataset = load_finqa_split(
        split=args.split,
        max_examples=args.max_examples,
    )

    print(
        f"FinQA examples: {len(dataset)}"
    )

    print(
        f"Loading model: {args.model_type}"
    )

    model, tokenizer, device = (
        load_model_for_evaluation(
            model_type=args.model_type
        )
    )

    print(
        f"Evaluation device: {device}"
    )

    example_results: list[ExampleResult] = []

    for index, example in enumerate(
        dataset
    ):
        result = evaluate_example(
            model=model,
            tokenizer=tokenizer,
            device=device,
            example=example,
            index=index,
            max_new_tokens=args.max_new_tokens,
        )

        example_results.append(
            result
        )

        if (
            (index + 1) % 25 == 0
            or index + 1 == len(dataset)
        ):
            current_exact_match = (
                sum(
                    item.exact_match
                    for item in example_results
                )
                / len(example_results)
            )

            current_numerical_accuracy = (
                sum(
                    item.numerical_accuracy
                    for item in example_results
                )
                / len(example_results)
            )

            print(
                f"[{index + 1}/{len(dataset)}] "
                f"EM={current_exact_match:.4%} "
                f"NumAcc={current_numerical_accuracy:.4%}"
            )

    benchmark_result = calculate_benchmark_metrics(
        results=example_results,
        model_type=args.model_type,
        split=args.split,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_prefix = (
        args.output_prefix
        or f"finqa_{args.model_type}_{timestamp}"
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
        f"Predictions saved to: {predictions_path}"
    )

    print(
        f"Metrics saved to: {metrics_path}"
    )


if __name__ == "__main__":
    main()