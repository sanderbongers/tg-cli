"""Personal chat support and compatibility with the group interface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner, Result
from telethon.errors import FloodWaitError
from telethon.tl import types
from telethon.utils import get_peer_id

from tg_cli import cli
from tg_cli.commands._peer import marked_peer_id, message_link_base
from tg_cli.commands._resolve import (
    AmbiguousGroupError,
    GroupNotFoundError,
    resolve,
)
from tg_cli.config import ConfigError


async def _iterate(items: Iterable[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _client(
    entities: Iterable[Any] = (), *, entity: Any = None, authorized: bool = True
) -> MagicMock:
    client = MagicMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.is_user_authorized = AsyncMock(return_value=authorized)
    client.get_entity = AsyncMock(return_value=entity)
    client.iter_dialogs.side_effect = lambda: _iterate(
        [SimpleNamespace(entity=item) for item in entities]
    )
    return client


def _invoke(command: str, client: MagicMock, *args: str) -> Result:
    with patch(f"tg_cli.commands.{command}.make_client", return_value=client):
        return CliRunner().invoke(cli.main, [command, *args])


@pytest.fixture
def alice() -> types.User:
    return types.User(
        id=42, first_name="Alice", last_name="Smith", username="alice"
    )


@pytest.fixture
def dialogs(alice: types.User) -> list[Any]:
    return [
        alice,
        types.User(id=43, first_name="Helper", bot=True),
        types.User(id=44, first_name="Me", is_self=True),
        types.Chat(
            id=42,
            title="Alice's group",
            photo=None,
            participants_count=3,
            date=None,
            version=1,
        ),
        types.Channel(
            id=45, title="Dev", photo=None, date=None, megagroup=True
        ),
        types.Channel(
            id=46, title="News", photo=None, date=None, broadcast=True
        ),
        types.Chat(
            id=47,
            title="Old Dev",
            photo=None,
            participants_count=3,
            date=None,
            version=1,
            migrated_to=types.InputChannel(45, 123),
        ),
    ]


def test_chats_lists_users_and_preserves_group_records(dialogs) -> None:
    client = _client(dialogs)
    result = _invoke("chats", client)
    assert result.exit_code == 0, result.output
    chats = json.loads(result.stdout)
    assert [chat["id"] for chat in chats] == [
        42,
        43,
        44,
        -42,
        -1000000000045,
        -1000000000046,
    ]
    assert chats[0] == {
        "id": 42,
        "title": "Alice Smith",
        "type": "personal",
        "username": "alice",
        "member_count": None,
    }
    assert chats[1]["type"] == "bot"
    assert chats[2]["title"] == "Saved Messages"
    groups_result = _invoke("groups", client)
    assert groups_result.exit_code == 0, groups_result.output
    assert chats[3:] == json.loads(groups_result.stdout)
    assert client.disconnect.await_count == 2


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("personal", [42, 44]),
        ("bot", [43]),
        ("group", [-42, -1000000000045]),
        ("channel", [-1000000000046]),
    ],
)
def test_chats_filters_dialog_types(dialogs, kind, expected) -> None:
    result = _invoke("chats", _client(dialogs), "--type", kind)
    assert result.exit_code == 0, result.output
    assert [chat["id"] for chat in json.loads(result.stdout)] == expected


def test_chats_applies_limit_after_filtering_and_pretty(dialogs) -> None:
    result = _invoke(
        "chats", _client(dialogs), "--type", "group", "--limit", "1", "--pretty"
    )
    assert result.exit_code == 0, result.output
    assert [chat["id"] for chat in json.loads(result.stdout)] == [-42]
    assert "\n  {" in result.stdout


def test_chats_empty_list() -> None:
    result = _invoke("chats", _client())
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_chats_requires_authentication_and_disconnects() -> None:
    client = _client(authorized=False)
    result = _invoke("chats", client)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["type"] == "AuthError"
    client.iter_dialogs.assert_not_called()
    client.disconnect.assert_awaited_once()


def test_chats_reports_configuration_errors() -> None:
    with patch(
        "tg_cli.commands.chats.make_client",
        side_effect=ConfigError("Missing configuration"),
    ):
        result = CliRunner().invoke(cli.main, ["chats"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["type"] == "ConfigError"


@pytest.mark.parametrize("args", [["--limit", "0"], ["--type", "user"]])
def test_chats_invalid_options_are_json_usage_errors(args) -> None:
    with patch("tg_cli.commands.chats.make_client") as make_client:
        result = CliRunner().invoke(cli.main, ["chats", *args])
    assert result.exit_code == 2
    assert json.loads(result.stderr)["type"] == "UsageError"
    make_client.assert_not_called()


@pytest.mark.parametrize("reference", ["42", "@alice"])
def test_resolve_personal_chat_by_exact_reference(alice, reference) -> None:
    client = _client(entity=alice)
    assert asyncio.run(resolve(client, reference, include_users=True)) is alice
    client.iter_dialogs.assert_not_called()
    key = 42 if reference == "42" else "@alice"
    client.get_entity.assert_awaited_once_with(key)


def test_personal_chat_ids_round_trip_from_listing(alice) -> None:
    result = _invoke("chats", _client([alice]))
    assert result.exit_code == 0
    peer_id = json.loads(result.stdout)[0]["id"]
    client = _client(entity=alice)
    assert (
        asyncio.run(resolve(client, str(peer_id), include_users=True)) is alice
    )
    assert marked_peer_id(alice) == get_peer_id(alice) == peer_id


def test_resolve_personal_chat_by_case_insensitive_name(alice) -> None:
    client = _client([alice])
    assert (
        asyncio.run(resolve(client, "  SMITH  ", include_users=True)) is alice
    )


def test_group_title_retains_priority_over_personal_name(dialogs) -> None:
    client = _client(dialogs)
    assert (
        asyncio.run(resolve(client, "alice", include_users=True)) is dialogs[3]
    )


def test_ambiguous_group_matches_do_not_fall_back_to_person(alice) -> None:
    client = _client(
        [
            alice,
            SimpleNamespace(id=1, title="Alice Dev"),
            SimpleNamespace(id=2, title="Alice News"),
        ]
    )
    with pytest.raises(AmbiguousGroupError) as error:
        asyncio.run(resolve(client, "alice", include_users=True))
    assert error.value.matches == ["Alice Dev", "Alice News"]


def test_ambiguous_personal_names_report_distinct_ids(alice) -> None:
    other = types.User(id=99, first_name="Alice", last_name="Smith")
    with pytest.raises(AmbiguousGroupError) as error:
        asyncio.run(
            resolve(_client([alice, other]), "alice", include_users=True)
        )
    assert "ambiguous chat" in str(error.value)
    assert "id=42" in str(error.value)
    assert "id=99" in str(error.value)


def test_repeated_user_dialog_is_not_ambiguous(alice) -> None:
    assert (
        asyncio.run(
            resolve(_client([alice, alice]), "alice", include_users=True)
        )
        is alice
    )


@pytest.mark.parametrize(
    "user",
    [
        types.User(id=43, first_name="Helper", bot=True),
        types.User(id=44, first_name="Me", is_self=True),
        types.User(id=45, deleted=True),
        types.User(id=46, username="onlyusername"),
    ],
)
def test_user_listing_names_resolve_back_to_the_same_peer(user) -> None:
    result = _invoke("chats", _client([user]))
    assert result.exit_code == 0
    title = json.loads(result.stdout)[0]["title"]
    assert (
        asyncio.run(resolve(_client([user]), title, include_users=True)) is user
    )
    assert marked_peer_id(user) == get_peer_id(user)
    assert message_link_base(user) is None


def test_resolver_rejects_unsupported_entities() -> None:
    client = _client(entity=types.UserEmpty(id=42))
    with pytest.raises(GroupNotFoundError):
        asyncio.run(resolve(client, "42", include_users=True))


def test_migrated_group_cannot_redirect_to_a_personal_chat(
    alice, dialogs
) -> None:
    client = _client()
    client.get_entity.side_effect = [dialogs[-1], alice]
    with pytest.raises(GroupNotFoundError):
        asyncio.run(resolve(client, "-47", include_users=True))


def _message(message_id: int, user: types.User) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        date=datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc),
        message=f"hello {message_id}",
        sender=user,
        sender_id=user.id,
    )


@pytest.mark.parametrize("reference", ["42", "@alice", "smith"])
@pytest.mark.parametrize("command", ["messages", "search", "get"])
def test_message_commands_accept_personal_chats(
    alice, command, reference
) -> None:
    client = _client([alice], entity=alice)
    newest = [_message(2, alice), _message(1, alice)]
    client.iter_messages.side_effect = lambda *args, **kwargs: _iterate(newest)
    client.get_messages = AsyncMock(return_value=newest)
    extra = {"messages": [], "search": ["hello"], "get": ["1", "2"]}
    result = _invoke(command, client, reference, *extra[command])
    assert result.exit_code == 0, result.output
    messages = json.loads(result.stdout)
    assert [message["id"] for message in messages] == (
        [2, 1] if command == "search" else [1, 2]
    )
    assert all(message["group_id"] == 42 for message in messages)
    assert all(message["link"] is None for message in messages)
    assert all(message["sender_username"] == "alice" for message in messages)
    if command == "get":
        client.get_messages.assert_awaited_once_with(alice, ids=[1, 2])
    elif command == "search":
        client.iter_messages.assert_called_once_with(
            alice, limit=100, search="hello"
        )
    else:
        client.iter_messages.assert_called_once_with(alice, limit=100)
    client.disconnect.assert_awaited_once()


def test_personal_chat_id_paging_uses_the_same_bounds(alice) -> None:
    client = _client(entity=alice)
    client.iter_messages.return_value = _iterate([_message(2, alice)])
    result = _invoke(
        "messages", client, "42", "--after-id", "1", "--through-id", "2"
    )
    assert result.exit_code == 0, result.output
    assert [item["id"] for item in json.loads(result.stdout)] == [2]
    client.iter_messages.assert_called_once_with(
        alice, limit=100, reverse=True, min_id=1, max_id=3
    )


def test_thread_still_rejects_personal_chats_before_fetching(alice) -> None:
    client = _client(entity=alice)
    client.get_messages = AsyncMock()
    result = _invoke("thread", client, "@alice", "1")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["type"] == "GroupNotFoundError"
    client.get_messages.assert_not_awaited()
    client.disconnect.assert_awaited_once()


def test_chats_reports_flood_wait_and_disconnects() -> None:
    client = _client()
    client.iter_dialogs.side_effect = FloodWaitError(request=None, capture=19)
    result = _invoke("chats", client)
    assert result.exit_code == 1
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["type"] == "FloodWaitError"
    assert "retry after 19 seconds" in error["error"]
    client.disconnect.assert_awaited_once()
