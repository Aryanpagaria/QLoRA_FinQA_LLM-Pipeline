"""
Base interface for model providers.

A provider is responsible for generating an answer from a conversation.
Concrete implementations can use different model backends, such as:

- LocalProvider: Qwen2.5-3B + QLoRA adapter
- GeminiProvider: Google Gemini API (added later)

The application layer depends on this interface rather than directly
depending on a specific model implementation.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """
    Abstract interface that every model provider must implement.

    The rest of the application can interact with any provider through
    this common interface without knowing how the underlying model works.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate an assistant response for a conversation.

        Parameters
        ----------
        messages:
            Conversation history in chat-message format.

            Example:
                [
                    {
                        "role": "user",
                        "content": "What is revenue?"
                    },
                    {
                        "role": "assistant",
                        "content": "Revenue is total income..."
                    },
                    {
                        "role": "user",
                        "content": "How is it different from profit?"
                    }
                ]

        Returns
        -------
        str
            The generated assistant response.

        Raises
        ------
        NotImplementedError
            This method must be implemented by every concrete provider.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Return a human-readable provider name.

        Concrete providers can override this when they want to expose
        their own name in the terminal or application logs.

        Returns
        -------
        str
            Provider name.
        """
        return self.__class__.__name__

    def health_check(self) -> bool:
        """
        Check whether the provider is available.

        This provides a simple common interface for future application
        startup checks.

        The default implementation assumes the provider is available.
        Providers that require external resources can override it.

        Returns
        -------
        bool
            True when the provider is available.
        """
        return True