"""LangGraph agents using two local MCP servers for Azure Blob and SQL MI.

See README.md for configuration, the sample SQL schema, and security limitations.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

# The requested script name shadows the installed MCP package.
_original_path = sys.path[:]
try:
    _workspace = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != _workspace
    ]
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
finally:
    sys.path[:] = _original_path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI

# Load the existing example as a module without confusing it with the UTCP SDK.
_spec = importlib.util.spec_from_file_location("azure_utcp_shared", _workspace / "utcp.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load the shared Azure operations from utcp.py")
shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shared
_spec.loader.exec_module(shared)

ServerKind = Literal["blob", "sql"]


def build_server(kind: ServerKind, backend: shared.AzureSdkProtocol) -> Server:
    """Expose one specialist's allowlisted operations over MCP."""
    server = Server(f"azure-{kind}-tools")
    specs = {spec.name: spec for spec in shared.SPECS if spec.name.startswith(kind + "_")}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.args.model_json_schema(),
            )
            for spec in specs.values()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in specs:
            raise ValueError(f"Unknown {kind} tool: {name}")
        values = specs[name].args.model_validate(arguments).model_dump()
        if name == "blob_create" and len(values["content"].encode("utf-8")) > shared.MAX_TEXT_BYTES:
            raise ValueError("Blob content exceeds 256 KiB in UTF-8.")
        return await asyncio.to_thread(backend._execute, name, values)

    return server


async def serve(kind: ServerKind) -> None:
    settings = shared.Settings.from_env()
    credential = DefaultAzureCredential()
    backend = None
    try:
        backend = shared.AzureSdkProtocol(settings, credential)
        server = build_server(kind, backend)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        try:
            if backend is not None:
                await backend.close()
        finally:
            credential.close()


class McpToolCaller:
    """Adapt separate MCP sessions to the shared LangGraph tool interface."""

    def __init__(self, blob: ClientSession, sql: ClientSession):
        self.sessions = {"blob": blob, "sql": sql}

    async def validate_tools(self) -> None:
        for kind, session in self.sessions.items():
            response = await session.list_tools()
            expected = {spec.name for spec in shared.SPECS if spec.name.startswith(kind + "_")}
            actual = {tool.name for tool in response.tools}
            if actual != expected:
                raise RuntimeError(f"MCP {kind} server tool mismatch: expected {expected}, got {actual}")

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        name = tool_name.removeprefix("azure.")
        if name not in shared.SPEC_BY_NAME:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        spec = shared.SPEC_BY_NAME[name]
        arguments = spec.args.model_validate(tool_args).model_dump()
        if name == "blob_create" and len(arguments["content"].encode("utf-8")) > shared.MAX_TEXT_BYTES:
            raise ValueError("Blob content exceeds 256 KiB in UTF-8.")
        if spec.writes:
            await asyncio.to_thread(shared.confirm_write, name, arguments)
        result = await self.sessions[name.split("_", 1)[0]].call_tool(name, arguments)
        if result.isError:
            detail = "\n".join(item.text for item in result.content if isinstance(item, types.TextContent))
            raise RuntimeError(f"MCP tool {name} failed: {detail or 'unknown server error'}")
        if result.structuredContent is None:
            raise RuntimeError(f"MCP tool {name} returned no structured result.")
        return result.structuredContent


async def open_session(stack: AsyncExitStack, kind: ServerKind) -> ClientSession:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--server", kind],
        env=dict(os.environ),
    )
    read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    return session


async def run(request: str) -> str:
    settings = shared.Settings.from_env()
    credential = DefaultAzureCredential()
    try:
        async with AsyncExitStack() as stack:
            blob = await open_session(stack, "blob")
            sql = await open_session(stack, "sql")
            caller = McpToolCaller(blob, sql)
            await caller.validate_tools()
            model = AzureChatOpenAI(
                azure_endpoint=settings.openai_endpoint,
                azure_deployment=settings.openai_deployment,
                api_version=settings.openai_api_version,
                azure_ad_token_provider=get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                ),
                temperature=0,
                max_retries=2,
                timeout=60,
            )
            result = await shared.build_graph(model, caller).ainvoke(
                {"request": request, "plan": [], "results": []}
            )
            return "\n\n".join(result["results"]) or "No Blob Storage or SQL operation was requested."
    finally:
        credential.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Azure MCP multi-agent example.")
    parser.add_argument("request", nargs="?", help="Natural-language request for the Blob and/or SQL agents.")
    parser.add_argument("--server", choices=("blob", "sql"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.server:
        if args.request is not None:
            parser.error("request cannot be used with --server")
        asyncio.run(serve(args.server))
        return
    if not args.request or not args.request.strip():
        parser.error("request must not be empty")
    try:
        print(asyncio.run(run(args.request)))
    except (ValueError, PermissionError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
