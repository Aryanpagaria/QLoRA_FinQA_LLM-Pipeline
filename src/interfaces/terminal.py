"""
Terminal interface for the financial assistant.

This module is responsible only for user interaction through the
command line. Application logic and model inference remain in the
application/provider layers.
"""

from src.application.chat_service import ChatService


class TerminalInterface:
    """
    Interactive terminal interface for ChatService.
    """

    def __init__(self, chat_service: ChatService) -> None:
        """
        Initialize the terminal interface.

        Parameters
        ----------
        chat_service:
            Application service responsible for processing messages.
        """

        if not isinstance(chat_service, ChatService):
            raise TypeError(
                "chat_service must be an instance of ChatService."
            )

        self.chat_service = chat_service

    def run(self) -> None:
        """
        Start the interactive terminal chat loop.
        """

        self._print_welcome()

        while True:
            try:
                user_input = input("\nYou: ").strip()

            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            if command in {"clear", "reset"}:
                self.chat_service.reset()
                print("Conversation cleared.")
                continue

            if command == "help":
                self._print_help()
                continue

            try:
                response = self.chat_service.chat(user_input)
                print(f"\nAssistant: {response}")

            except Exception as exc:
                print(
                    f"\nError: {exc}"
                )

    def _print_welcome(self) -> None:
        """Print the terminal application's welcome message."""

        print("=" * 60)
        print("Financial QA Assistant")
        print("=" * 60)
        print(
            f"Provider: {self.chat_service.provider_name()}"
        )
        print("\nType 'help' for commands.")
        print("Type 'exit', 'quit', or 'q' to leave.")

    def _print_help(self) -> None:
        """Print available terminal commands."""

        print("\nAvailable commands:")
        print("  help   - Show this help message")
        print("  clear  - Clear the current conversation")
        print("  reset  - Clear the current conversation")
        print("  exit   - Exit the application")
        print("  quit   - Exit the application")
        print("  q      - Exit the application")