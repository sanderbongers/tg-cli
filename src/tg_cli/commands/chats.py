"""``tg chats`` — list personal chats, bots, groups and channels."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from ..client import make_client
from ..errors import AuthError, handle_errors
from ..models import Chat
from ._peer import (
    is_user_entity,
    marked_peer_id,
    public_username,
    user_display_name,
)
from .groups import _dialog_to_group

_TYPE_CHOICES = ("personal", "bot", "group", "channel", "all")


@click.command()
@click.option(
    "--type",
    "type_filter",
    type=click.Choice(_TYPE_CHOICES),
    default="all",
    show_default=True,
    help="Filter by dialog kind; personal includes Saved Messages.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of dialogs to return.",
)
@click.option("--pretty", is_flag=True, help="Indent JSON output.")
@handle_errors
def chats(type_filter: str, limit: int | None, pretty: bool) -> None:
    """List your personal chats, bots, groups and channels as JSON."""
    result = asyncio.run(_collect_chats(type_filter, limit))
    click.echo(
        json.dumps(
            [chat.to_dict() for chat in result], indent=2 if pretty else None
        )
    )


async def _collect_chats(type_filter: str, limit: int | None) -> list[Chat]:
    client = make_client()
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise AuthError()
        collected: list[Chat] = []
        async for dialog in client.iter_dialogs():
            chat = _dialog_to_chat(dialog)
            if chat is None:
                continue
            if type_filter == "group":
                matches = chat.type in ("group", "supergroup")
            else:
                matches = type_filter in ("all", chat.type)
            if not matches:
                continue
            collected.append(chat)
            if limit is not None and len(collected) >= limit:
                break
        return collected
    finally:
        await client.disconnect()


def _dialog_to_chat(dialog: Any) -> Chat | None:
    entity = dialog.entity
    if not is_user_entity(entity):
        return _dialog_to_group(dialog)
    return Chat(
        id=marked_peer_id(entity),
        title=user_display_name(entity),
        type="bot" if entity.bot else "personal",
        username=public_username(entity),
        member_count=None,
    )
