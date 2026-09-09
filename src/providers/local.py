"""
Local model provider.

This provider connects the application layer to the locally loaded
Qwen2.5-3B-Instruct model with the trained QLoRA adapter.

The provider loads the model once when it is created and reuses that
model for subsequent requests.
"""

from typing import Any

import torch

from src.inference.inference import (
    InferenceConfig,
    _build_prompt,
    generate_response,
    load_inference_stack,
)
from src.providers.base import BaseProvider


class LocalProvider(BaseProvider):
    """
    Provider for the locally fine-tuned Qwen model.

    The model, tokenizer, and inference configuration are loaded once
    during initialization. Each call to generate() then reuses them.
    """

    def __init__(self) -> None:
        """
        Initialize the local inference provider.

        Loads:
        - inference configuration
        - tokenizer
        - Qwen base model
        - trained LoRA adapter
        """

        self.config: InferenceConfig
        self.tokenizer: Any
        self.model: torch.nn.Module

        (
            self.config,
            self.tokenizer,
            self.model,
        ) = load_inference_stack()

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate an assistant response from the local model.

        Parameters
        ----------
        messages:
            Conversation history in standard chat-message format.

        Returns
        -------
        str
            The generated assistant response.

        Raises
        ------
        ValueError
            If messages are empty or malformed.
        RuntimeError
            If model generation fails.
        """

        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        if not messages:
            raise ValueError("messages cannot be empty.")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "Each message must be a dictionary."
                )

            if "role" not in message or "content" not in message:
                raise ValueError(
                    "Each message must contain 'role' and 'content'."
                )

            if not isinstance(message["role"], str):
                raise ValueError(
                    "Message 'role' must be a string."
                )

            if not isinstance(message["content"], str):
                raise ValueError(
                    "Message 'content' must be a string."
                )

        try:
            prompt = _build_prompt(
                tokenizer=self.tokenizer,
                messages=messages,
            )

            result = generate_response(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                config=self.config,
            )

            return result.response

        except Exception as exc:
            raise RuntimeError(
                "Local model generation failed."
            ) from exc

    def name(self) -> str:
        """
        Return the human-readable provider name.
        """

        return "Local QLoRA"

    def health_check(self) -> bool:
        """
        Check whether the local inference stack is loaded.

        Returns
        -------
        bool
            True when the model and tokenizer are available.
        """

        return (
            self.model is not None
            and self.tokenizer is not None
        )