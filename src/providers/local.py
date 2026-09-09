"""
Local model provider.

This provider wraps the existing inference engine and exposes it through
the common BaseProvider interface.

The local backend uses:
    - Qwen/Qwen2.5-3B-Instruct
    - 4-bit quantization
    - the exported QLoRA adapter

The model is loaded once when LocalProvider is created and reused for
all subsequent generation requests.
"""

from typing import Any

from src.inference.inference import (
    GenerationResult,
    _build_prompt,
    _load_base_model,
    _load_inference_config,
    _load_lora_adapter,
    _load_tokenizer,
    _set_reproducibility_seed,
    generate_response,
)

from .base import BaseProvider


class LocalProvider(BaseProvider):
    """
    Provider for the locally hosted Qwen + QLoRA model.

    The model and tokenizer are loaded once during initialization.
    Each call to generate() reuses the same loaded model.
    """

    def __init__(self) -> None:
        """
        Initialize the local provider.

        This loads:
            1. Inference configuration
            2. Reproducibility seed
            3. Tokenizer
            4. Base Qwen model
            5. QLoRA adapter

        Model loading is intentionally done here rather than inside
        generate(), so a multi-turn conversation does not reload the
        model for every user message.
        """

        self.config: dict[str, Any] = _load_inference_config()

        if not self.config["inference"]["enabled"]:
            raise RuntimeError(
                "Inference is disabled in the inference configuration."
            )

        # Make generation reproducible according to the configured seed.
        _set_reproducibility_seed(
            self.config["inference"]["reproducibility"]["seed"]
        )

        # Load tokenizer once.
        self.tokenizer = _load_tokenizer(self.config)

        # Load the quantized base model once.
        self.model = _load_base_model(
            self.config,
        )

        # Attach the exported LoRA adapter once.
        self.model = _load_lora_adapter(
            self.model,
            self.config,
        )

        # Used by health_check() to confirm initialization completed.
        self._initialized = True

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate an assistant response from a conversation.

        Parameters
        ----------
        messages:
            Conversation history in chat-message format.

            Example:
                [
                    {
                        "role": "system",
                        "content": "You are a financial assistant."
                    },
                    {
                        "role": "user",
                        "content": "What is revenue?"
                    }
                ]

        Returns
        -------
        str
            The generated assistant response.
        """

        if not self._initialized:
            raise RuntimeError(
                "LocalProvider has not been initialized."
            )

        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")

        if not messages:
            raise ValueError("messages cannot be empty.")

        # Build the model-ready chat prompt using the tokenizer's
        # Qwen chat template.
        prompt = _build_prompt(
            self.tokenizer,
            messages,
        )

        # Run generation through the existing inference engine.
        result: GenerationResult = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            self.config,
        )

        response = result.get("response")

        if not isinstance(response, str):
            raise RuntimeError(
                "Inference engine returned an invalid response."
            )

        return response

    def name(self) -> str:
        """
        Return the human-readable provider name.
        """

        return "Local QLoRA"

    def health_check(self) -> bool:
        """
        Check whether the local provider is initialized and ready.
        """

        return (
            self._initialized
            and self.model is not None
            and self.tokenizer is not None
        )