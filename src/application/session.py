"""
Conversation session management.

Keeps track of the messages exchanged during a chat session.
"""

from typing import Any


class Session:
    """
    Stores and manages conversation history for a single chat session.
    """

    def __init__(
        self,
        system_message: str | None = None,
        max_history: int = 10,
    ) -> None:
        """
        Initialize a conversation session.

        Parameters
        ----------
        system_message:
            Optional system instruction added at the beginning
            of the conversation.

        max_history:
            Maximum number of non-system messages retained.
        """

        if max_history < 1:
            raise ValueError("max_history must be at least 1.")

        self.max_history = max_history
        self.messages: list[dict[str, str]] = []

        if system_message is not None:
            if not isinstance(system_message, str):
                raise TypeError("system_message must be a string.")

            if not system_message.strip():
                raise ValueError("system_message cannot be empty.")

            self.messages.append(
                {
                    "role": "system",
                    "content": system_message.strip(),
                }
            )

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""

        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""

        self._add_message("assistant", content)

    def _add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Add a validated message and trim old history."""

        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Unsupported message role: {role}"
            )

        if not isinstance(content, str):
            raise TypeError("Message content must be a string.")

        content = content.strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        self._trim_history()

    def _trim_history(self) -> None:
        """Keep the system message and the newest conversation messages."""

        system_message: list[dict[str, str]] = []
        conversation_messages: list[dict[str, str]] = []

        for message in self.messages:
            if message["role"] == "system":
                system_message.append(message)
            else:
                conversation_messages.append(message)

        conversation_messages = conversation_messages[
            -self.max_history :
        ]

        self.messages = system_message + conversation_messages

    def get_messages(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""

        return [message.copy() for message in self.messages]

    def clear(self) -> None:
        """Clear the conversation while preserving the system message."""

        system_messages = [
            message
            for message in self.messages
            if message["role"] == "system"
        ]

        self.messages = system_messages

    def is_empty(self) -> bool:
        """Return True when there are no user/assistant messages."""

        return not any(
            message["role"] != "system"
            for message in self.messages
        )

    def message_count(self) -> int:
        """Return the number of user/assistant messages."""

        return sum(
            1
            for message in self.messages
            if message["role"] != "system"
        )

    def last_message(self) -> dict[str, str] | None:
        """Return the latest message, or None if the session is empty."""

        if not self.messages:
            return None

        return self.messages[-1].copy()