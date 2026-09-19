import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReasonGraph"))

import api_base


class DummyCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class DummyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=DummyCompletions())


class OpenAIAPIExportsTest(unittest.TestCase):
    def test_openai_client_and_request_use_retry_and_timeout(self):
        original_openai = api_base.OpenAI
        api_base.OpenAI = DummyClient
        try:
            api = api_base.OpenAIAPI("test-key", "gpt-4o-mini")

            self.assertEqual(api.client.kwargs["timeout"], api_base.DEFAULT_TIMEOUT_SECONDS)
            self.assertEqual(api.client.kwargs["max_retries"], api_base.DEFAULT_MAX_RETRIES)

            api.generate_response("Hello")

            self.assertEqual(
                api.client.chat.completions.calls[0]["timeout"],
                api_base.DEFAULT_TIMEOUT_SECONDS,
            )
            self.assertNotIn("max_retries", api.client.chat.completions.calls[0])
        finally:
            api_base.OpenAI = original_openai

    def test_openai_uses_fallback_for_legacy_model_name(self):
        class FallbackClient(DummyClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.calls = []

            class chat:
                pass

        class FailingCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise api_base.APIError("Error code: 404 - model_not_found", "OpenAI", 404)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="fallback-ok"))]
                )

        class FallbackOpenAIClient(DummyClient):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.chat = SimpleNamespace(completions=FailingCompletions())

        original_openai = api_base.OpenAI
        api_base.OpenAI = FallbackOpenAIClient
        try:
            api = api_base.OpenAIAPI("test-key", "gpt-4")
            response = api.generate_response("Hello")
            self.assertEqual(response, "fallback-ok")
            self.assertEqual(len(api.client.chat.completions.calls), 2)
            self.assertEqual(api.client.chat.completions.calls[1]["model"], "gpt-4o-mini")
        finally:
            api_base.OpenAI = original_openai


if __name__ == "__main__":
    unittest.main()
