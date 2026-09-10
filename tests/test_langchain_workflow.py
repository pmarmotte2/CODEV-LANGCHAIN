import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create_structured(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed=app.NegotiationOutput(
                reply="Je confirme que l'export est limite aux contrats rouges.",
                project_decisions=[],
                client_decisions=[app.DecisionOutput(
                    text="L'export est limite aux contrats rouges.",
                    evidence="Je confirme que l'export est limite aux contrats rouges.",
                )],
                codev_user_decisions=[],
            ),
            model="gpt-5-nano",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


class LangChainWorkflowTests(unittest.TestCase):
    def test_langchain_adapter_uses_native_structured_output(self):
        parsed = app.SavedSessionTitle(title="Export contrats rouges")

        class FakeStructuredModel:
            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                self.options = kwargs
                return self

            def invoke(self, messages):
                self.messages = messages
                raw = SimpleNamespace(
                    content="",
                    usage_metadata={
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                    response_metadata={"model_name": "gpt-5-nano"},
                )
                return {"parsed": parsed, "raw": raw, "parsing_error": None}

        fake_model = FakeStructuredModel()
        owner = SimpleNamespace(
            provider="openai",
            api_key="test",
            base_url="https://api.openai.com/v1",
        )
        adapter = app.LangChainChatCompletions(owner)
        with patch.object(app, "ChatOpenAI", return_value=fake_model):
            response = adapter.create_structured(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Test"}],
                schema=app.SavedSessionTitle,
            )

        self.assertIs(response.parsed, parsed)
        self.assertEqual(response.usage["total_tokens"], 6)
        self.assertIs(fake_model.schema, app.SavedSessionTitle)
        self.assertEqual(fake_model.options["method"], "json_schema")
        self.assertTrue(fake_model.options["strict"])

    def test_ollama_provider_does_not_require_an_openai_key(self):
        client, model, reasoning_effort = app.get_llm_config("ollama", "light")

        self.assertEqual(client.provider, "ollama")
        self.assertEqual(model, app.OLLAMA_LIGHT_MODEL)
        self.assertIsNone(reasoning_effort)
        self.assertEqual(client.base_url, app.OLLAMA_BASE_URL)

    def test_ollama_model_factory_uses_chat_ollama(self):
        owner = SimpleNamespace(
            provider="ollama",
            api_key="",
            base_url="http://127.0.0.1:11434",
        )
        adapter = app.LangChainChatCompletions(owner)
        sentinel = object()
        with patch.object(app, "ChatOllama", return_value=sentinel) as constructor:
            model = adapter._build_model("qwen3:4b", None)

        self.assertIs(model, sentinel)
        constructor.assert_called_once_with(
            model="qwen3:4b",
            base_url="http://127.0.0.1:11434",
            validate_model_on_init=True,
        )

    def test_ollama_usage_has_no_api_cost(self):
        usage = app.summarize_llm_usage(
            "ollama:qwen3:4b",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        self.assertEqual(usage["estimated_cost_usd"], 0.0)
        self.assertTrue(usage["fully_priced"])

    def test_negotiation_graph_calls_model_once_and_validates_decision(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result = app.NEGOTIATION_GRAPH.invoke({
            "client": client,
            "model": "gpt-5-nano",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Test"}],
            "profile": "sales",
            "document_text": "",
            "codev_user_text": "",
            "existing_decisions": [],
        })

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(len(result["decisions"]), 1)
        self.assertEqual(result["decisions"][0]["source"], "client")

    def test_duplicate_decision_is_not_added_again(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        existing = [{
            "text": "L'export est limite aux contrats rouges.",
            "evidence": "Je confirme que l'export est limite aux contrats rouges.",
            "source": "client",
            "profile": "sales",
        }]

        result = app.NEGOTIATION_GRAPH.invoke({
            "client": client,
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": "Test"}],
            "profile": "sales",
            "document_text": "",
            "codev_user_text": "",
            "existing_decisions": existing,
        })

        self.assertEqual(result["decisions"], [])

    def test_remote_langchain_tracing_is_disabled_by_default(self):
        self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")

    def test_negotiation_schema_rejects_an_incomplete_decision(self):
        with self.assertRaises(ValueError):
            app.NegotiationOutput.model_validate({
                "reply": "Decision confirmee.",
                "project_decisions": [],
                "client_decisions": [{"text": "Il manque la preuve."}],
                "codev_user_decisions": [],
            })

    def test_report_schema_and_local_global_score_calculation(self):
        report = app.FramingReport(
            global_score=99,
            scores=[
                app.MaturityScore(name="Completude", score=40, reason="Partiel"),
                app.MaturityScore(name="Securite", score=10, reason="Non discute"),
                app.MaturityScore(name="Performance", score=20, reason="Mentionne"),
                app.MaturityScore(name="UX", score=30, reason="Partiel"),
            ],
            executive_summary="Synthese",
            critical_points=[],
            clarified_points=[],
            residual_risks=[],
            acceptance_criteria=[],
            next_actions=[],
        )

        normalized = app.normalize_report(report.model_dump())

        self.assertEqual(normalized["global_score"], 20)

    def test_all_llm_structured_outputs_are_pydantic_models(self):
        self.assertTrue(issubclass(app.SavedSessionTitle, app.BaseModel))
        self.assertTrue(issubclass(app.NegotiationOutput, app.BaseModel))
        self.assertTrue(issubclass(app.FramingReport, app.BaseModel))
        self.assertTrue(issubclass(app.ReportImprovement, app.BaseModel))


if __name__ == "__main__":
    unittest.main()
