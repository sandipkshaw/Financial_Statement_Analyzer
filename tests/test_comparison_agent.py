import unittest

from comparison_agent import (
    CompanyProfile,
    chunk_markdown,
    comparison_warnings,
    extract_company_profile,
    format_comparison_response,
    currency_symbol,
)


class ComparisonAgentHelpersTests(unittest.TestCase):
    def test_extracts_company_and_reporting_metadata(self) -> None:
        dossier = """# Financial Statement & Grounding Dossier: Example Bank

## Metadata & Entity Overview
- **Reporting Currency**: INR in Crores
- **Fiscal Year Definition**: April 1 to March 31
- **Periods Covered**: FY22, FY23, FY24
"""
        profile = extract_company_profile(dossier, "example.md")

        self.assertEqual(profile.name, "Example Bank")
        self.assertEqual(profile.currency, "INR in Crores")
        self.assertEqual(profile.fiscal_year, "April 1 to March 31")
        self.assertEqual(profile.periods, "FY22, FY23, FY24")

    def test_chunks_preserve_section_and_overlap(self) -> None:
        profile = CompanyProfile("Example Bank", "example.md", "INR", "Apr-Mar", "FY22")
        chunks = chunk_markdown(
            "## Financials\n" + ("financial evidence " * 20),
            profile,
            chunk_size=100,
            overlap=20,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["section"] == "Financials" for chunk in chunks))
        self.assertTrue(all(chunk["text"].startswith("Example Bank | Financials") for chunk in chunks))
        self.assertIn("financial evidence", chunks[0]["text"])
        self.assertIn("financial evidence", chunks[1]["text"])

    def test_table_chunks_repeat_column_headers(self) -> None:
        profile = CompanyProfile("Example Bank", "example.md", "INR", "Apr-Mar", "FY22")
        table = (
            "## Financials\n| Metric | FY22 | FY23 |\n| :--- | ---: | ---: |\n"
            + "".join(f"| Metric {index} | {index} | {index + 1} |\n" for index in range(12))
        )

        chunks = chunk_markdown(table, profile, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all("| Metric | FY22 | FY23 |\n| :--- | ---: | ---: |" in chunk["text"] for chunk in chunks[1:])
        )

    def test_comparison_warnings_report_currency_and_fiscal_mismatches(self) -> None:
        left = CompanyProfile("A", "a.md", "INR", "Apr-Mar", "FY24")
        right = CompanyProfile("B", "b.md", "USD", "Jan-Dec", "FY24")

        warnings = comparison_warnings(left, right)

        self.assertEqual(len(warnings), 2)
        self.assertIn("Currency mismatch", warnings[0])
        self.assertIn("Fiscal-year mismatch", warnings[1])

    def test_currency_symbol_is_added_for_supported_currencies(self) -> None:
        self.assertEqual(currency_symbol("Indian Rupee (INR) in Crores"), "₹")
        self.assertEqual(currency_symbol("USD"), "$")
        self.assertEqual(currency_symbol("EUR"), "€")
        self.assertEqual(currency_symbol("GBP"), "£")

    def test_matching_reporting_basis_has_no_warnings(self) -> None:
        left = CompanyProfile("A", "a.md", "INR", "Apr-Mar", "FY24")
        right = CompanyProfile("B", "b.md", "INR", "Apr-Mar", "FY24")

        self.assertEqual(comparison_warnings(left, right), ())

    def test_formatted_response_shows_both_companies_side_by_side(self) -> None:
        left = CompanyProfile("A", "a.md", "INR", "Apr-Mar", "FY24")
        right = CompanyProfile("B", "b.md", "USD", "Jan-Dec", "FY24")
        response = format_comparison_response(
            "Revenue remains comparable at a ratio level.",
            left,
            right,
            comparison_warnings(left, right),
        )

        self.assertIn("## A vs. B", response)
        self.assertIn("| A | ₹ INR | Apr-Mar | FY24 |", response)
        self.assertIn("| B | $ USD | Jan-Dec | FY24 |", response)
        self.assertIn("### Side-by-side analysis", response)
        self.assertIn("Reporting status", response)


if __name__ == "__main__":
    unittest.main()
