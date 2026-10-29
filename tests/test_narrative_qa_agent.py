import unittest
from types import SimpleNamespace

from narrative_qa_agent import NarrativeQAAgent


class FakeGroundingContext:
    def __init__(self, evidence):
        self.evidence = evidence

    def render(self):
        return "[Example Bank | example.md § Financials]\nRevenue was ₹100 crore."


class FakeGroundingAgent:
    def build_context(self, company, question):
        evidence = (
            SimpleNamespace(citation=lambda: "example.md § Financials"),
        )
        return FakeGroundingContext(evidence)


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Revenue was ₹100 crore. [Example Bank | example.md § Financials]"))]
        )


class NarrativeQAAgentTests(unittest.TestCase):
    def test_answers_with_grounded_context_and_sources(self) -> None:
        agent = NarrativeQAAgent.__new__(NarrativeQAAgent)
        agent.grounding_agent = FakeGroundingAgent()
        agent.company_profiles = {"Example Bank": SimpleNamespace(currency="INR", fiscal_year="Apr-Mar", periods="FY22-FY24")}
        agent.model = "gpt-4o-mini"
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        result = agent.answer("Example Bank", "What was revenue?")

        self.assertIn("₹100 crore", result.answer)
        self.assertEqual(result.sources, ("example.md § Financials",))

    def test_rejects_empty_question(self) -> None:
        agent = NarrativeQAAgent.__new__(NarrativeQAAgent)
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            agent.answer("Example Bank", "   ")


if __name__ == "__main__":
    unittest.main()
