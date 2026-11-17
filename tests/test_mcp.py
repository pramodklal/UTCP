import asyncio
import importlib.util
import os
import sys
import unittest
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

MODULE_PATH = Path(__file__).resolve().parents[1] / "mcp.py"
SPEC = importlib.util.spec_from_file_location("azure_mcp_example", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = app
SPEC.loader.exec_module(app)


@asynccontextmanager
async def connected_server(kind, backend):
    server = app.build_server(kind, backend)
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(0)
    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(0)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            server.run,
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )
        try:
            async with app.ClientSession(server_to_client_receive, client_to_server_send) as session:
                await session.initialize()
                yield session
        finally:
            tasks.cancel_scope.cancel()


class McpTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_mcp_sessions_discover_and_call_tools(self):
        blob_backend = MagicMock()
        blob_backend._execute.return_value = {"names": ["reports/a.txt"], "limit": 100}
        sql_backend = MagicMock()
        sql_backend._execute.return_value = {"found": True, "record": {"id": "1", "content": "hello"}}
        async with connected_server("blob", blob_backend) as blob:
            async with connected_server("sql", sql_backend) as sql:
                caller = app.McpToolCaller(blob, sql)
                await caller.validate_tools()
                self.assertEqual(
                    await caller.call_tool("azure.blob_list", {"prefix": "reports/"}),
                    {"names": ["reports/a.txt"], "limit": 100},
                )
                self.assertEqual(
                    (await caller.call_tool("azure.sql_read", {"record_id": "1"}))["record"]["content"],
                    "hello",
                )
                blob_backend._execute.assert_called_once_with("blob_list", {"prefix": "reports/"})
                sql_backend._execute.assert_called_once_with("sql_read", {"record_id": "1"})

    async def test_langgraph_agents_call_both_mcp_servers(self):
        class FakePlanner:
            async def ainvoke(self, messages):
                return app.shared.Plan(agents=["blob", "sql"])

        class FakeModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "test"

            def with_structured_output(self, schema, **kwargs):
                return FakePlanner()

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if isinstance(messages[-1], ToolMessage):
                    message = AIMessage(content="Operation completed.")
                else:
                    blob = "Blob Storage specialist" in messages[0].content
                    name = "blob_list" if blob else "sql_read"
                    args = {"prefix": "reports/"} if blob else {"record_id": "1"}
                    message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])
                return ChatResult(generations=[ChatGeneration(message=message)])

        blob_backend = MagicMock()
        blob_backend._execute.return_value = {"names": ["reports/a.txt"], "limit": 100}
        sql_backend = MagicMock()
        sql_backend._execute.return_value = {"found": True, "record": {"id": "1", "content": "hello"}}
        async with connected_server("blob", blob_backend) as blob:
            async with connected_server("sql", sql_backend) as sql:
                caller = app.McpToolCaller(blob, sql)
                result = await app.shared.build_graph(FakeModel(), caller).ainvoke(
                    {"request": "List reports and read record 1", "plan": [], "results": []}
                )
                self.assertEqual(result["results"], ["blob: Operation completed.", "sql: Operation completed."])
        blob_backend._execute.assert_called_once_with("blob_list", {"prefix": "reports/"})
        sql_backend._execute.assert_called_once_with("sql_read", {"record_id": "1"})

    async def test_server_rejects_bad_arguments_and_unknown_tools(self):
        backend = MagicMock()
        async with connected_server("blob", backend) as session:
            result = await session.call_tool("blob_read", {"name": "a", "unexpected": True})
            self.assertTrue(result.isError)
            result = await session.call_tool("sql_read", {"record_id": "1"})
            self.assertTrue(result.isError)
        backend._execute.assert_not_called()

    async def test_approval_is_required_before_mcp_mutation(self):
        session = AsyncMock()
        caller = app.McpToolCaller(session, session)
        with patch.object(app.shared, "confirm_write", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                await caller.call_tool("azure.sql_delete", {"record_id": "1"})
        session.call_tool.assert_not_awaited()

    async def test_server_error_is_not_reported_as_success(self):
        backend = MagicMock()
        backend._execute.side_effect = ValueError("storage unavailable")
        async with connected_server("blob", backend) as session:
            caller = app.McpToolCaller(session, session)
            with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
                await caller.call_tool("azure.blob_read", {"name": "a"})

    async def test_approved_sql_mutation_calls_server(self):
        backend = MagicMock()
        backend._execute.return_value = {"found": True, "record": {"id": "1", "content": "ok"}}
        async with connected_server("sql", backend) as session:
            caller = app.McpToolCaller(session, session)
            with patch.object(app.shared, "confirm_write") as approval:
                result = await caller.call_tool("azure.sql_update", {"record_id": "1", "content": "ok"})
            approval.assert_called_once_with("sql_update", {"record_id": "1", "content": "ok"})
            self.assertTrue(result["found"])
            backend._execute.assert_called_once_with("sql_update", {"record_id": "1", "content": "ok"})

    async def test_real_stdio_processes_discover_separate_tools(self):
        config = {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "test",
            "AZURE_OPENAI_API_VERSION": "2024-10-21",
            "AZURE_STORAGE_ACCOUNT_URL": "https://example.blob.core.windows.net",
            "AZURE_STORAGE_CONTAINER": "files",
            "AZURE_SQL_SERVER": "example.database.windows.net",
            "AZURE_SQL_DATABASE": "test",
        }
        with patch.dict(os.environ, config):
            async with AsyncExitStack() as stack:
                blob = await asyncio.wait_for(app.open_session(stack, "blob"), timeout=30)
                sql = await asyncio.wait_for(app.open_session(stack, "sql"), timeout=30)
                caller = app.McpToolCaller(blob, sql)
                await caller.validate_tools()
                result = await blob.call_tool("blob_read", {"name": "test", "extra": 1})
                self.assertTrue(result.isError)


if __name__ == "__main__":
    unittest.main()
