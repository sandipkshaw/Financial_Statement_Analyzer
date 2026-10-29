import unittest

from agentic_orchestrator import AgenticFinancialOrchestrator


class AgenticOrchestratorRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = object.__new__(AgenticFinancialOrchestrator)

    def test_routes_comparison_intent(self):
        route, reason = self.orchestrator._deterministic_route(
            "Compare the operating margins of HDFC Bank and ICICI Bank"
        )
        self.assertEqual(route, "comparison")
        self.assertIn("comparison", reason.lower())

    def test_routes_trend_intent(self):
        route, _ = self.orchestrator._deterministic_route(
            "Are there any DSO or liquidity red flags?"
        )
        self.assertEqual(route, "trend")

    def test_routes_retrieval_intent(self):
        route, _ = self.orchestrator._deterministic_route(
            "Show the source evidence supporting net profit"
        )
        self.assertEqual(route, "retrieval")

    def test_routes_axis_bank_financial_trends_to_trend_agent(self):
        route, _ = self.orchestrator._deterministic_route(
            "What are the key financial trends for Axis Bank Limited?"
        )
        self.assertEqual(route, "trend")

    def test_each_request_is_routed_from_current_question(self):
        first, _ = self.orchestrator._deterministic_route(
            "Compare HDFC Bank and ICICI Bank"
        )
        second, _ = self.orchestrator._deterministic_route(
            "What are the key financial trends for Axis Bank Limited?"
        )
        third, _ = self.orchestrator._deterministic_route(
            "What was Axis Bank's net profit in FY24?"
        )
        self.assertEqual((first, second, third), ("comparison", "trend", "narrative"))

    def test_defaults_to_narrative(self):
        route, _ = self.orchestrator._deterministic_route(
            "What was the bank's deposit growth?"
        )
        self.assertEqual(route, "narrative")

    def test_trend_filter_returns_only_flags_named_in_question(self):
        class Flag:
            def __init__(self, pattern):
                self.pattern = pattern
                self.source = "primary.md"

            def to_markdown(self):
                return f"- {self.pattern}"

        flags = (
            Flag("DSO climbing"),
            Flag("Liquidity deterioration"),
            Flag("Margin compression"),
        )
        selected = self.orchestrator._select_trend_flags(flags, "How did liquidity change?")
        self.assertEqual([flag.pattern for flag in selected], ["Liquidity deterioration"])

    def test_trend_node_generates_query_specific_llm_response(self):
        class Flag:
            pattern = "Liquidity deterioration"
            source = "primary.md"

            def to_markdown(self):
                return "- Liquidity deteriorated by 2 percentage points"

        class Router:
            def __init__(self):
                self.prompt = ""

            def invoke(self, prompt):
                self.prompt = prompt
                return type("Response", (), {"content": "The liquidity ratio fell by 2 points."})()

        self.orchestrator.trend_agent = type(
            "TrendAgent", (), {"analyze_company": lambda self, company: [Flag()]}
        )()
        self.orchestrator.router = Router()
        result = self.orchestrator._trend_node(
            {
                "left_company": "Primary Bank",
                "right_company": "Comparison Bank",
                "question": "How did liquidity change in FY24?",
                "history": "",
            }
        )

        self.assertEqual(result["answer"], "The liquidity ratio fell by 2 points.")
        self.assertIn("How did liquidity change in FY24?", self.orchestrator.router.prompt)
        self.assertIn("2 percentage points", self.orchestrator.router.prompt)

    def test_trend_node_uses_only_primary_company(self):
        class Flag:
            pattern = "Margin compression"
            source = "primary.md"

            def to_markdown(self):
                return "- Primary trend signal"

        class TrendAgent:
            def analyze_company(self, company):
                self.company = company
                return [Flag()]

        trend_agent = TrendAgent()
        self.orchestrator.trend_agent = trend_agent
        result = self.orchestrator._trend_node(
            {
                "left_company": "Primary Bank",
                "right_company": "Comparison Bank",
                "question": "Compare trends",
            }
        )

        self.assertEqual(trend_agent.company, "Primary Bank")
        self.assertIn("Primary Bank", result["answer"])
        self.assertNotIn("Comparison Bank", result["answer"])


if __name__ == "__main__":
    unittest.main()
