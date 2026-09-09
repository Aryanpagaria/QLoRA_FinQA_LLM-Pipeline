"""
Gemini model provider.

Connects the application layer to Google's Gemini API through the
official Google GenAI SDK and the Interactions API.
"""

import os
from typing import Any

from google import genai

from src.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Provider for Google's Gemini API.
    """

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
    ) -> None:
        """
        Initialize the Gemini provider.
        """

        self.model = model

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate a response from Gemini.

        The application uses the common message format:

            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "..."}
            {"role": "system", "content": "..."}

        These messages are converted into the Interactions API's
        stateless conversation format.
        """

        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        if not messages:
            raise ValueError("messages cannot be empty.")

        system_instruction: str | None = None

        interaction_input: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "Each message must be a dictionary."
                )

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                raise ValueError(
                    "Message role must be a string."
                )

            if not isinstance(content, str):
                raise ValueError(
                    "Message content must be a string."
                )

            content = content.strip()

            if not content:
                raise ValueError(
                    "Message content cannot be empty."
                )

            if role == "system":
                system_instruction = content

            elif role == "user":
                interaction_input.append(
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                            }
                        ],
                    }
                )

            elif role == "assistant":
                interaction_input.append(
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                            }
                        ],
                    }
                )

            else:
                raise ValueError(
                    f"Unsupported message role: {role}"
                )

        if not interaction_input:
            raise ValueError(
                "Conversation must contain at least one "
                "user or assistant message."
            )

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": interaction_input,
        }

        if system_instruction:
            request_kwargs["system_instruction"] = (
                system_instruction
            )

        try:
            interaction = self.client.interactions.create(
                **request_kwargs,
            )

        except Exception as exc:
            raise RuntimeError(
                "Gemini generation failed."
            ) from exc

        text = getattr(
            interaction,
            "output_text",
            None,
        )

        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text.strip()

    def name(self) -> str:
        """
        Return the provider name.
        """

        return f"Gemini ({self.model})"

    def health_check(self) -> bool:
        """
        Check whether the Gemini API key is configured.

        This does not make an API request.
        """

        return bool(
            os.getenv("GEMINI_API_KEY")
        )