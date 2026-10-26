"""MCP client: starts an MCP server as a child process and calls its tools over stdio.

One background task owns each connection from start to finish. The MCP SDK uses
anyio cancel scopes, which must be entered and left by the same task, and
LangGraph runs graph nodes in different tasks. So callers never touch the
session directly: they put a request on a queue and wait for the answer.

Child processes get a minimal environment (PATH, the import path, and
``INSURITHM_*`` settings). In particular they do not inherit ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

POLICY_LIBRARY_MODULE = "insurithm.mcp_servers.policy_library"
CLAIMS_TRACKER_MODULE = "insurithm.mcp_servers.claims_tracker"


class ToolError(Exception):
    """A tool refused or failed. `code` is the tool's own error code when it gave one."""

    def __init__(self, tool: str, code: str, message: str):
        super().__init__(f"{tool}: [{code}] {message}")
        self.tool, self.code, self.message = tool, code, message


def server_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for an MCP child process. Secrets such as the Gemini key are left out."""
    import insurithm

    package_parent = str(Path(insurithm.__file__).resolve().parents[1])
    env = {
        "PYTHONPATH": os.pathsep.join(p for p in [package_parent, os.environ.get("PYTHONPATH", "")] if p),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    for key, value in os.environ.items():
        if key.startswith("INSURITHM_"):
            env[key] = value
    env.update(extra or {})
    return env


def parse_tool_result(tool: str, result: CallToolResult) -> dict[str, Any]:
    """Turn an MCP result into the tool's payload, or raise ToolError."""
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    if result.isError:
        raise ToolError(tool, "invalid_call", " ".join(texts)[:500] or "tool reported an error")

    data: Any = result.structuredContent
    if isinstance(data, dict) and set(data) == {"result"} and isinstance(data["result"], dict):
        data = data["result"]
    if not isinstance(data, dict):
        try:
            data = json.loads(texts[0]) if texts else {}
        except json.JSONDecodeError:
            raise ToolError(tool, "bad_response", "tool did not return JSON") from None
    if not isinstance(data, dict):
        raise ToolError(tool, "bad_response", "tool did not return a JSON object")

    if data.get("ok") is False:
        error = data.get("error") or {}
        raise ToolError(tool, str(error.get("code", "error")), str(error.get("message", "tool failed")))
    return data


def _errlog():
    """Where a child's stderr goes: ours if it is a real file, otherwise nowhere."""
    try:
        sys.stderr.fileno()
        return sys.stderr
    except (AttributeError, OSError, ValueError):
        return open(os.devnull, "w")


class MCPConnection:
    def __init__(self, name: str, module: str, env: dict[str, str] | None = None, python: str | None = None):
        self.name, self.module = name, module
        self._env = server_env(env)
        self._python = python or sys.executable
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> "MCPConnection":
        if self._task is not None and not self._task.done():
            return self
        self._queue = asyncio.Queue()
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")
        await self._ready
        return self

    async def stop(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            await self._queue.put(None)
        try:
            await self._task
        except Exception:
            pass
        self._task = None

    async def __aenter__(self) -> "MCPConnection":
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def _run(self) -> None:
        params = StdioServerParameters(command=self._python, args=["-m", self.module], env=self._env)
        try:
            async with stdio_client(params, errlog=_errlog()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    assert self._ready is not None
                    self._ready.set_result(None)
                    while True:
                        item = await self._queue.get()
                        if item is None:
                            return
                        operation, future = item
                        try:
                            future.set_result(await operation(session))
                        except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream) as exc:
                            # The pipe to the server is gone: report it and let the next call restart us.
                            future.set_exception(exc)
                            return
                        except Exception as exc:
                            future.set_exception(exc)
        except BaseException as exc:
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(ToolError(self.name, "server_unavailable", f"could not start: {exc!r}"))
            # Anything still waiting gets an error instead of hanging forever.
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None and not item[1].done():
                    item[1].set_exception(ToolError(self.name, "server_unavailable", "server stopped"))
            if not isinstance(exc, Exception):
                raise

    async def _submit(self, operation: Any) -> Any:
        if self._task is None or self._task.done():
            await self.start()  # the server died or was never started: bring it back
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put((operation, future))
        return await future

    # -- calls -------------------------------------------------------------
    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """Call a tool. Returns its payload, or raises ToolError."""
        arguments = {k: v for k, v in arguments.items() if v is not None}

        async def operation(session: ClientSession) -> CallToolResult:
            return await session.call_tool(tool, arguments)

        try:
            result = await self._submit(operation)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(tool, "server_unavailable", f"{type(exc).__name__}: {exc}") from exc
        return parse_tool_result(tool, result)

    async def list_tools(self) -> list[dict[str, Any]]:
        async def operation(session: ClientSession) -> Any:
            return await session.list_tools()

        listing = await self._submit(operation)
        return [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in listing.tools]

    async def read_resource(self, uri: str) -> str:
        async def operation(session: ClientSession) -> Any:
            return await session.read_resource(uri)  # type: ignore[arg-type]

        result = await self._submit(operation)
        return "".join(getattr(c, "text", "") for c in result.contents)

    async def get_prompt(self, name: str) -> str:
        async def operation(session: ClientSession) -> Any:
            return await session.get_prompt(name)

        result = await self._submit(operation)
        return "\n".join(getattr(m.content, "text", "") for m in result.messages)


class Tools:
    """The two connections the workflow uses."""

    def __init__(self, policies: MCPConnection, tracker: MCPConnection):
        self.policies = policies
        self.tracker = tracker

    @classmethod
    def local(cls, tracker_db: str | Path, policy_pdfs: list[str | Path] | None = None,
              allowed_policies: list[str] | None = None, policy_dir: str | Path | None = None) -> "Tools":
        policy_env: dict[str, str] = {}
        if policy_dir is not None:
            policy_env["INSURITHM_POLICY_DIR"] = str(policy_dir)
        if policy_pdfs:
            policy_env["INSURITHM_POLICY_PDFS"] = os.pathsep.join(str(p) for p in policy_pdfs)
        if allowed_policies is not None:
            policy_env["INSURITHM_ALLOWED_POLICIES"] = ",".join(allowed_policies)
        return cls(
            MCPConnection("policy-library", POLICY_LIBRARY_MODULE, policy_env),
            MCPConnection("claims-tracker", CLAIMS_TRACKER_MODULE, {"INSURITHM_TRACKER_DB": str(tracker_db)}),
        )

    async def start(self) -> "Tools":
        await asyncio.gather(self.policies.start(), self.tracker.start())
        return self

    async def stop(self) -> None:
        await asyncio.gather(self.policies.stop(), self.tracker.stop())

    async def __aenter__(self) -> "Tools":
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
