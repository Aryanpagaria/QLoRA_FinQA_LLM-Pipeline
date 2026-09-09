"""
Application-level chat service.

This module coordinates conversation sessions and model providers.

It does not know how the underlying model works. The provider abstraction
allows the same service to work with the local QLoRA model, Gemini, or
another provider added later.
"""

from src.application.session import ChatSession
from src.providers.base import BaseProvider


class ChatService:
    """
    Coordinate user messages, conversation history, and model responses.

    Responsibilities:
        1. Receive a user question.
        2. Add it to the current session.
        3. Send the conversation to the selected provider.
        4. Store the assistant response.
        5. Return the response to the interface.
    """

    def __init__(
        self,
        provider: BaseProvider,
        session: ChatSession | None = None,
    ) -> None:
        """
        Initialize the chat service.

        Parameters
        ----------
        provider:
            Model provider used to generate responses.

        session:
            Optional existing conversation session.
            A new session is created when one is not provided.
        """

        if provider is None:
            raise ValueError("provider cannot be None.")

        self.provider = provider

        self.session = (
            session
            if session is not None
            else ChatSession()
        )

    def ask(self, question: str) -> str:
        """
        Send a user question through the conversation pipeline.

        Parameters
        ----------
        question:
            User's message.

        Returns
        -------
        str
            Assistant's generated response.
        """

        if not isinstance(question, str):
            raise ValueError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        # Add the user's message before generation so the model
        # receives the complete conversation.
        self.session.add_user_message(question)

        try:
            response = self.provider.generate(
                self.session.messages()
            )
        except Exception:
            # If generation fails, remove the user message that was
            # just added so the session remains consistent.
            self._remove_last_user_message()
            raise

        if not isinstance(response, str):
            self._remove_last_user_message()
            raise RuntimeError(
                "Provider returned a non-string response."
            )

        response = response.strip()

        if not response:
            self._remove_last_user_message()
            raise RuntimeError(
                "Provider returned an empty response."
            )

        # Store the assistant response only after successful generation.
        self.session.add_assistant_message(response)

        return response

    def history(self) -> list[dict[str, str]]:
        """
        Return the current conversation history.
        """

        return self.session.messages()

    def clear_history(self) -> None:
        """
        Clear the current conversation while preserving the
        session's system message.
        """

        self.session.clear()

    def provider_name(self) -> str:
        """
        Return the name of the currently configured provider.
        """

        return self.provider.name()

    def health_check(self) -> bool:
        """
        Check whether the configured provider is available.
        """

        return self.provider.health_check()

    def _remove_last_user_message(self) -> None:
        """
        Remove the most recently added user message.

        This is used when provider generation fails, preventing a failed
        request from leaving an incomplete message in the conversation.
        """

        messages = self.session.messages()

        if not messages:
            return

        if messages[-1]["role"] != "user":
            return

        messages.pop()

        self.session.clear()

        for message in messages:
            if message["role"] == "user":
                self.session.add_user_message(
                    message["content"]
                )
            elif message["role"] == "assistant":
                self.session.add_assistant_message(
                    message["content"]
                )