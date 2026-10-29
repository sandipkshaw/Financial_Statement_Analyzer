import unittest
from pathlib import Path

from retrieval_grounding_agent import DeterministicFact, RetrievalGroundingAgent


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(sum(ord(char) for char in text) % 997 + 1), 1.0] for text in texts]


class RetrievalGroundingAgentTests(unittest.TestCase):
    def test_context_combines_retrieved_evidence_and_deterministic_fact(self) -> None:
        source = """# Financial Statement & Grounding Dossier: Example Bank

## Metadata & Entity Overview
- **Reporting Currency**: INR in Crores

## Profit and Loss
| Line Item Code | Line Item Description | FY22 (Audited) | FY23 (Audited) | FY24 (Audited) |
| --- | --- | ---: | ---: | ---: |
| P&L-01 | Operating Profit | 100 | 150 | 180 |
"""
        temp = Path(__name__ + "_example.md")
        temp.write_text(source, encoding="utf-8")
        try:
            agent = RetrievalGroundingAgent(
                {"Example Bank": [source]},
                {"Example Bank": [{"source": temp.name, "section": "Profit and Loss"}]},
                {"Example Bank": source},
                fake_embed,
                result_count=1,
            )
            context = agent.build_context("Example Bank", "operating profit")
            rendered = context.render()
            self.assertEqual(len(context.evidence), 1)
            self.assertEqual(len(context.computed_facts), 1)
            fact = context.computed_facts[0]
            self.assertEqual(fact.period_from, "FY23")
            self.assertEqual(fact.period_to, "FY24")
            self.assertEqual(fact.change_percent, 20.0)
            self.assertIn("DETERMINISTIC FACT", rendered)
            self.assertIn(temp.name, rendered)
        finally:
            temp.unlink(missing_ok=True)

    def test_missing_denominator_returns_no_fact(self) -> None:
        source = "| Line Item | FY22 | FY23 | FY24 |\n| --- | --- | --- | --- |\n| Profit | 10 | 0 | 2 |"
        agent = RetrievalGroundingAgent(
            {"Example Bank": [source]},
            {"Example Bank": [{"source": "example.md", "section": "Financials"}]},
            {"Example Bank": source},
            fake_embed,
            result_count=1,
        )
        self.assertEqual(agent.compute_year_over_year_facts("Example Bank", "profit"), ())


if __name__ == "__main__":
    unittest.main()
