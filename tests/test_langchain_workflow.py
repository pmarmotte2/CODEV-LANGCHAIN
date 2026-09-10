import json
import os
import unittest
from types import SimpleNamespace

import app


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps({
            "reply": "Je confirme que l'export est limite aux contrats rouges.",
            "project_decisions": [],
            "client_decisions": [{
                "text": "L'export est limite aux contrats rouges.",
                "evidence": "Je confirme que l'export est limite aux contrats rouges.",
            }],
            "codev_user_decisions": [],
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content)),],
            model="gpt-5-nano",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


class LangChainWorkflowTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
