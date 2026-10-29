import unittest
from pathlib import Path

from trend_red_flag_agent import TrendRedFlagAgent


class TrendRedFlagAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TrendRedFlagAgent(Path(__file__).resolve().parents[1] / "data")

    def test_returns_all_six_required_patterns(self) -> None:
        path = self.agent.data_dir / "HDFC_Bank_Financial_Statements_FY22_FY24.md"
        flags = self.agent.analyze_file(path)

        self.assertEqual([flag.pattern for flag in flags], list(self.agent.PATTERN_NAMES))

    def test_hdfc_margin_compression_uses_structured_fy23_fy24_values(self) -> None:
        path = self.agent.data_dir / "HDFC_Bank_Financial_Statements_FY22_FY24.md"
        flags = {flag.pattern: flag for flag in self.agent.analyze_file(path)}
        margin = flags["Margin compression"]

        self.assertEqual(margin.status, "flagged")
        values = {observation.period: observation.value for observation in margin.observations}
        self.assertEqual(values, {"FY23": 4.1, "FY24": 3.63})
        self.assertIn("%", margin.to_markdown())

    def test_monetary_trend_values_include_currency_and_crores(self) -> None:
        path = self.agent.data_dir / "HDFC_Bank_Financial_Statements_FY22_FY24.md"
        flags = {flag.pattern: flag for flag in self.agent.analyze_file(path)}
        markdown = flags["Accrual-quality gap"].to_markdown()

        self.assertIn("₹", markdown)
        self.assertIn("crore", markdown)
        self.assertIn("₹", flags["Accrual-quality gap"].summary)
        self.assertIn("crore", flags["Accrual-quality gap"].summary)

    def test_axis_non_flagged_summaries_include_currency_and_crores(self) -> None:
        path = self.agent.data_dir / "Axis_Bank_Financial_Statements_FY22_FY24.md"
        flags = {flag.pattern: flag for flag in self.agent.analyze_file(path)}

        accrual_summary = flags["Accrual-quality gap"].summary
        leverage_summary = flags["Leverage stress"].summary
        self.assertIn("₹", accrual_summary)
        self.assertIn("crore", accrual_summary)
        self.assertIn("₹", leverage_summary)
        self.assertIn("crore", leverage_summary)

    def test_missing_inventory_series_is_qualified_not_flagged(self) -> None:
        path = self.agent.data_dir / "HDFC_Bank_Financial_Statements_FY22_FY24.md"
        flags = {flag.pattern: flag for flag in self.agent.analyze_file(path)}

        self.assertEqual(flags["Inventory build-up"].status, "qualified unavailable")


if __name__ == "__main__":
    unittest.main()
