"""
Terminal interface for the financial assistant.

This module is responsible only for:
- reading user input
- displaying responses
- handling terminal commands

Application logic belongs to ChatService.
Model logic belongs to providers/inference.
"""

from src.application.chat_service import ChatService


class TerminalInterface:
    """
    Command-line interface for the financial assistant.
    """

    def __init__(
        self,
        chat_service: ChatService,
    ) -> None:
        """
        Initialize the terminal interface.
        """

        if chat_service is None:
            raise ValueError(
                "chat_service cannot be None."
            )

        self.chat_service = chat_service

    def run(self) -> None:
        """
        Start the interactive terminal chat.
        """

        self._print_welcome()

        while True:
            try:
                question = input("\nYou: ").strip()

            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break

            if not question:
                continue

            if self._is_exit_command(question):
                print("\nGoodbye!")
                break

            if question.lower() == "clear":
                self.chat_service.clear_history()
                print("\nConversation history cleared.")
                continue

            if question.lower() == "history":
                self._print_history()
                continue

            try:
                print("\nAssistant: ", end="", flush=True)

                response = self.chat_service.ask(question)

                print(response)

            except Exception as exc:
                print(
                    "\nError: "
                    f"{exc}"
                )

    def _is_exit_command(
        self,
        question: str,
    ) -> bool:
        """
        Check whether the user requested to exit.
        """

        exit_commands = {
            "exit",
            "quit",
            "q",
        }

        configured_commands = getattr(
            self.chat_service.provider,
            "config",
            None,
        )

        if configured_commands is not None:
            try:
                configured_commands = (
                    configured_commands.inference
                    .chat
                    .exit_commands
                )

                exit_commands.update(
                    command.lower()
                    for command in configured_commands
                )

            except AttributeError:
                pass

        return question.lower() in exit_commands

    def _print_welcome(self) -> None:
        """
        Display the terminal startup message.
        """

        print("=" * 60)
        print("Financial QA Assistant")
        print("=" * 60)
        print(
            f"Provider: "
            f"{self.chat_service.provider_name()}"
        )
        print()
        print("Ask a financial question to begin.")
        print("Commands:")
        print("  clear   - Clear conversation history")
        print("  history - Show conversation history")
        print("  exit    - Exit the application")
        print("=" * 60)

    def _print_history(self) -> None:
        """
        Display the current conversation history.
        """

        history = self.chat_service.history()

        if not history:
            print("\nNo conversation history.")
            return

        print("\nConversation history:")
        print("-" * 60)

        for message in history:
            role = message["role"].capitalize()
            content = message["content"]

            print(f"{role}: {content}")

        print("-" * 60)