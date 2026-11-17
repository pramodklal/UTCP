import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

MODULE_PATH = Path(__file__).resolve().parents[1] / "utcp.py"
SPEC = importlib.util.spec_from_file_location("azure_utcp_example", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = app
SPEC.loader.exec_module(app)


def settings():
    return app.Settings(
        openai_endpoint="https://example.openai.azure.com",
        openai_deployment="test",
        openai_api_version="2024-10-21",
        storage_account_url="https://example.blob.core.windows.net",
        blob_container="files",
        sql_server="example.database.windows.net",
        sql_database="test",
    )


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.credential = MagicMock()
        self.blob_patch = patch.object(app, "BlobServiceClient")
        self.blob_service = self.blob_patch.start().return_value
        self.protocol = app.AzureSdkProtocol(settings(), self.credential)

    async def asyncTearDown(self):
        await self.protocol.close()
        self.blob_patch.stop()

    async def test_real_utcp_registration_and_dispatch(self):
        app.register_azure_protocol(settings(), self.credential)
        client = await app.UtcpClient.create()
        try:
            result = await client.register_manual(app.AzureSdkTemplate(name="azure"))
            self.assertTrue(result.success)
            self.assertEqual(len(result.manual.tools), 8)
            self.blob_service.get_container_client.return_value.list_blobs.return_value = [
                SimpleNamespace(name="reports/a.txt")
            ]
            response = await client.call_tool("azure.blob_list", {"prefix": "reports/"})
            self.assertEqual(response["names"], ["reports/a.txt"])
        finally:
            await client.close()

    async def test_mutation_requires_approval_before_execution(self):
        with patch.object(app, "confirm_write", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                await self.protocol.call_tool(None, "azure.blob_delete", {"name": "a.txt"}, app.AzureSdkTemplate())
        self.blob_service.get_container_client.assert_not_called()

    async def test_blob_create_never_overwrites(self):
        blob = self.blob_service.get_container_client.return_value.get_blob_client.return_value
        with patch.object(app, "confirm_write"):
            result = await self.protocol.call_tool(
                None, "azure.blob_create", {"name": "a.txt", "content": "hello"}, app.AzureSdkTemplate()
            )
        self.assertTrue(result["created"])
        blob.upload_blob.assert_called_once_with(b"hello", overwrite=False)

    async def test_invalid_or_oversized_arguments_do_not_execute(self):
        with self.assertRaises(ValueError):
            await self.protocol.call_tool(
                None, "azure.blob_create", {"name": "a.txt", "content": "\u00e9" * 140000}, app.AzureSdkTemplate()
            )
        with self.assertRaises(ValueError):
            await self.protocol.call_tool(
                None, "azure.sql_read", {"record_id": "id", "sql": "DROP TABLE"}, app.AzureSdkTemplate()
            )
        self.blob_service.get_container_client.assert_not_called()

    async def test_sql_update_uses_parameters_and_commits(self):
        self.credential.get_token.return_value.token = "token"
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.fetchone.return_value = ("id", "updated")
        with patch.object(app.pyodbc, "connect", return_value=connection) as connect:
            result = self.protocol._execute("sql_update", {"record_id": "id", "content": "updated"})
        self.assertEqual(result["record"], {"id": "id", "content": "updated"})
        cursor.execute.assert_called_once_with(
            "UPDATE dbo.AgentRecords SET content = ? OUTPUT INSERTED.id, INSERTED.content WHERE id = ?",
            "updated", "id",
        )
        connection.commit.assert_called_once()
        connection.close.assert_called_once()
        self.assertIn(1256, connect.call_args.kwargs["attrs_before"])

    async def test_sql_failure_closes_without_commit(self):
        self.credential.get_token.return_value.token = "token"
        connection = MagicMock()
        connection.cursor.return_value.execute.side_effect = RuntimeError("SQL failed")
        with patch.object(app.pyodbc, "connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "SQL failed"):
                self.protocol._execute("sql_delete", {"record_id": "id"})
        connection.commit.assert_not_called()
        connection.close.assert_called_once()

    async def test_sql_create_read_and_delete_are_parameterized(self):
        self.credential.get_token.return_value.token = "token"
        cases = [
            ("sql_create", {"record_id": "id", "content": "value"}, ("id", "value"), True),
            ("sql_read", {"record_id": "id"}, ("id",), False),
            ("sql_delete", {"record_id": "id"}, ("id",), True),
        ]
        for operation, arguments, parameters, commits in cases:
            with self.subTest(operation=operation):
                connection = MagicMock()
                connection.cursor.return_value.fetchone.return_value = ("id", "value")
                with patch.object(app.pyodbc, "connect", return_value=connection):
                    result = self.protocol._execute(operation, arguments)
                self.assertTrue(result["found"])
                execute_args = connection.cursor.return_value.execute.call_args.args
                self.assertEqual(execute_args[1:], parameters)
                self.assertEqual(execute_args[0].count("?"), len(parameters))
                self.assertEqual(connection.commit.call_count, int(commits))
                connection.close.assert_called_once()

    async def test_blob_read_and_delete(self):
        blob = self.blob_service.get_container_client.return_value.get_blob_client.return_value
        blob.get_blob_properties.return_value.size = 5
        blob.download_blob.return_value.readall.return_value = b"hello"
        result = await self.protocol.call_tool(
            None, "azure.blob_read", {"name": "a.txt"}, app.AzureSdkTemplate()
        )
        self.assertEqual(result["content"], "hello")
        with patch.object(app, "confirm_write"):
            result = await self.protocol.call_tool(
                None, "azure.blob_delete", {"name": "a.txt"}, app.AzureSdkTemplate()
            )
        self.assertTrue(result["deleted"])
        blob.delete_blob.assert_called_once()


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_agents_run_in_plan_order_and_share_prior_result(self):
        class FakePlanner:
            async def ainvoke(self, messages):
                return app.Plan(agents=["blob", "sql"])

        class FakeAgent:
            def __init__(self, label):
                self.label = label
                self.inputs = []

            async def ainvoke(self, state, config):
                self.inputs.append(state["messages"][0].content)
                return {"messages": [SimpleNamespace(text=self.label + " complete")]}

        blob = FakeAgent("blob")
        sql = FakeAgent("sql")
        model = MagicMock()
        model.with_structured_output.return_value = FakePlanner()
        with patch.object(app, "create_agent", side_effect=[blob, sql]):
            graph = app.build_graph(model, MagicMock())
        result = await graph.ainvoke({"request": "Use both", "plan": [], "results": []})
        self.assertEqual(result["results"], ["blob: blob complete", "sql: sql complete"])
        self.assertIn("blob: blob complete", sql.inputs[0])

    def test_noninteractive_mutation_is_denied(self):
        with patch.object(app.sys.stdin, "isatty", return_value=False):
            with self.assertRaises(PermissionError):
                app.confirm_write("sql_delete", {"record_id": "123"})

    async def test_real_agent_executes_tool_via_utcp_client(self):
        class FakePlanner:
            async def ainvoke(self, messages):
                return app.Plan(agents=["blob"])

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
                    message = AIMessage(content="The blob list was retrieved.")
                else:
                    message = AIMessage(
                        content="",
                        tool_calls=[{"name": "blob_list", "args": {"prefix": "reports/"}, "id": "call-1"}],
                    )
                return ChatResult(generations=[ChatGeneration(message=message)])

        credential = MagicMock()
        with patch.object(app, "BlobServiceClient") as blob_service:
            blob_service.return_value.get_container_client.return_value.list_blobs.return_value = [
                SimpleNamespace(name="reports/a.txt")
            ]
            app.register_azure_protocol(settings(), credential)
            client = await app.UtcpClient.create()
            try:
                await client.register_manual(app.AzureSdkTemplate(name="azure"))
                graph = app.build_graph(FakeModel(), client)
                result = await graph.ainvoke({"request": "List reports", "plan": [], "results": []})
                self.assertEqual(result["results"], ["blob: The blob list was retrieved."])
                blob_service.return_value.get_container_client.return_value.list_blobs.assert_called_once_with(
                    name_starts_with="reports/"
                )
            finally:
                await client.close()


if __name__ == "__main__":
    unittest.main()
