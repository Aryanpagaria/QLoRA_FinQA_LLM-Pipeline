"""
Application service for conversational chat.

Coordinates the conversation session and model provider.
The interface layer should communicate with ChatService instead of
directly interacting with the model.
"""

from .session import Session
from src.providers.base import BaseProvider


class ChatService:
    """
    Main application service responsible for processing chat messages.
    """

    def __init__(
        self,
        provider: BaseProvider,
        session: Session | None = None,
    ) -> None:
        """
        Initialize the chat service.

        Parameters
        ----------
        provider:
            Model provider responsible for generating responses.

        session:
            Conversation session. If omitted, a new session is created.
        """

        if not isinstance(provider, BaseProvider):
            raise TypeError(
                "provider must be an instance of BaseProvider."
            )

        self.provider = provider
        self.session = session or Session()

    def chat(self, user_message: str) -> str:
        """
        Process one user message and return the assistant response.

        The message is added to the session, sent to the provider,
        and the generated response is added back to the session.
        """

        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string.")

        user_message = user_message.strip()

        if not user_message:
            raise ValueError("user_message cannot be empty.")

        self.session.add_user_message(user_message)

        try:
            response = self.provider.generate(
                self.session.get_messages()
            )
        except Exception:
            # Remove the user message if generation fails so the
            # session does not retain an unanswered request.
            self._remove_last_user_message()
            raise

        if not isinstance(response, str):
            raise RuntimeError(
                "Provider returned an invalid response."
            )

        response = response.strip()

        if not response:
            raise RuntimeError(
                "Provider returned an empty response."
            )

        self.session.add_assistant_message(response)

        return response

    def reset(self) -> None:
        """Clear the current conversation history."""

        self.session.clear()

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""

        return self.session.get_messages()

    def provider_name(self) -> str:
        """Return the name of the active model provider."""

        return self.provider.name()

    def health_check(self) -> bool:
        """Return whether the active provider is ready."""

        return self.provider.health_check()

    def _remove_last_user_message(self) -> None:
        """Remove the latest user message after a failed generation."""

        messages = self.session.messages

        if messages and messages[-1]["role"] == "user":
            messages.pop()