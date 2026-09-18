"""
Deterministic text generation utilities for benchmark evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationOutput:
    """
    Structured result produced by the generation engine.
    """

    text: str
    input_tokens: int
    generated_tokens: int
    total_tokens: int


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> GenerationOutput:
    """
    Generate a deterministic response from a causal language model.

    The returned text contains only newly generated tokens and does not
    include the original prompt.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")

    model.eval()

    encoded_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    encoded_inputs = {
        key: value.to(device)
        for key, value in encoded_inputs.items()
    }

    input_token_count = int(
        encoded_inputs["input_ids"].shape[-1]
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        raise RuntimeError(
            "Tokenizer must define either pad_token_id or eos_token_id."
        )

    with torch.inference_mode():
        generated = model.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = generated[
        0,
        input_token_count:,
    ]

    generated_token_count = int(
        generated_token_ids.shape[-1]
    )

    total_token_count = (
        input_token_count
        + generated_token_count
    )

    text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return GenerationOutput(
        text=text,
        input_tokens=input_token_count,
        generated_tokens=generated_token_count,
        total_tokens=total_token_count,
    )