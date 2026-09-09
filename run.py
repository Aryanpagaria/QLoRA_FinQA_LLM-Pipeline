"""
Application entry point.

This module wires together:
    TerminalInterface
        ↓
    ChatService
        ↓
    ChatSession + Provider
        ↓
    LocalProvider / GeminiProvider
"""

import argparse

from src.application.chat_service import ChatService
from src.application.session import ChatSession
from src.interfaces.terminal import TerminalInterface
from src.providers.base import BaseProvider
from src.providers.gemini import GeminiProvider
from src.providers.local import LocalProvider


def create_provider(provider_name: str) -> BaseProvider:
    """
    Create the requested model provider.

    Parameters
    ----------
    provider_name:
        Provider identifier:
            local
            gemini

    Returns
    -------
    BaseProvider
        Initialized model provider.
    """

    if provider_name == "local":
        return LocalProvider()

    if provider_name == "gemini":
        return GeminiProvider()

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )


def build_application(
    provider_name: str,
) -> TerminalInterface:
    """
    Build the complete terminal application.

    Parameters
    ----------
    provider_name:
        Model provider to use.

    Returns
    -------
    TerminalInterface
        Fully configured terminal application.
    """

    provider = create_provider(provider_name)

    session = ChatSession(
        max_history=10,
    )

    chat_service = ChatService(
        provider=provider,
        session=session,
    )

    return TerminalInterface(
        chat_service=chat_service,
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Financial QA Assistant",
    )

    parser.add_argument(
        "--provider",
        choices=["local", "gemini"],
        default="gemini",
        help=(
            "Model provider to use. "
            "Default: gemini"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Start the application.
    """

    args = parse_arguments()

    application = build_application(
        provider_name=args.provider,
    )

    application.run()


if __name__ == "__main__":
    main()