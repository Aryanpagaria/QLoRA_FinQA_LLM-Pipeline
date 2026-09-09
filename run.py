"""
Application entry point.

Builds the provider, session, chat service, and terminal interface,
then starts the financial assistant.
"""

from src.application.chat_service import ChatService
from src.application.session import Session
from src.interfaces.terminal import TerminalInterface
from src.providers.local import LocalProvider


def main() -> None:
    """Initialize the application and start the terminal interface."""

    print("Loading local model...")

    provider = LocalProvider()

    session = Session(
        max_history=10,
    )

    chat_service = ChatService(
        provider=provider,
        session=session,
    )

    terminal = TerminalInterface(
        chat_service=chat_service,
    )

    terminal.run()


if __name__ == "__main__":
    main()