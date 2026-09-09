"""
Conversation session management.

This module owns the conversation state for a chat session.

It does not know anything about:
- the model
- QLoRA
- Gemini
- terminal input/output
- HTTP
- FastAPI

It only manages the messages that belong to the conversation.
"""

from copy import deepcopy


class ChatSession:
    """
    Store and manage the messages in one conversation.

    A session contains:
        system message
        user messages
        assistant responses

    The session can be passed to any provider that understands the
    standard chat-message format.
    """

    def __init__(
        self,
        system_message: str | None = None,
        max_history: int = 10,
    ) -> None:
        """
        Create a new conversation session.

        Parameters
        ----------
        system_message:
            Optional instruction that defines the assistant's behavior.

        max_history:
            Maximum number of non-system messages retained in the session.
        """

        if max_history < 1:
            raise ValueError("max_history must be at least 1.")

        if system_message is not None:
            if not isinstance(system_message, str):
                raise ValueError(
                    "system_message must be a string or None."
                )

            if not system_message.strip():
                raise ValueError(
                    "system_message cannot be empty."
                )

        self.max_history = max_history
        self._messages: list[dict[str, str]] = []

        if system_message is not None:
            self._messages.append(
                {
                    "role": "system",
                    "content": system_message.strip(),
                }
            )

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        """

        self._add_message(
            role="user",
            content=content,
        )

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant response to the conversation.
        """

        self._add_message(
            role="assistant",
            content=content,
        )

    def _add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Add a validated message and enforce the history limit.
        """

        if role not in {"user", "assistant"}:
            raise ValueError(
                "Only 'user' and 'assistant' messages can be added "
                "through this method."
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

        self._messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        self._trim_history()

    def _trim_history(self) -> None:
        """
        Keep the system message and the newest conversation messages.

        max_history applies to non-system messages.
        """

        system_messages = [
            message
            for message in self._messages
            if message["role"] == "system"
        ]

        conversation_messages = [
            message
            for message in self._messages
            if message["role"] != "system"
        ]

        conversation_messages = conversation_messages[
            -self.max_history:
        ]

        self._messages = (
            system_messages[:1]
            + conversation_messages
        )

    def messages(self) -> list[dict[str, str]]:
        """
        Return a copy of the current conversation.

        A deep copy prevents callers from accidentally modifying the
        session's internal state.
        """

        return deepcopy(self._messages)

    def clear(self) -> None:
        """
        Clear the conversation while preserving the system message.
        """

        system_messages = [
            message
            for message in self._messages
            if message["role"] == "system"
        ]

        self._messages = system_messages[:1]

    def is_empty(self) -> bool:
        """
        Return True if the session contains no user or assistant messages.
        """

        return not any(
            message["role"] != "system"
            for message in self._messages
        )

    def message_count(self) -> int:
        """
        Return the number of non-system messages in the session.
        """

        return sum(
            1
            for message in self._messages
            if message["role"] != "system"
        )

    def last_message(self) -> dict[str, str] | None:
        """
        Return the most recent message.

        Returns None when the conversation has no messages.
        """

        if not self._messages:
            return None

        return deepcopy(self._messages[-1])