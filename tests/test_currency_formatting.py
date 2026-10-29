import unittest

from currency_formatting import format_amount


class CurrencyFormattingTests(unittest.TestCase):
    def test_formats_indian_amount_in_crore(self) -> None:
        self.assertEqual(format_amount(1234.5, "₹ crore"), "₹1,234.5 crore")

    def test_accepts_crores_and_cr_units(self) -> None:
        self.assertEqual(format_amount(500, "INR crores"), "₹500 crore")
        self.assertEqual(format_amount(500, "₹ cr."), "₹500 crore")

    def test_negative_amount_uses_currency_before_magnitude(self) -> None:
        self.assertEqual(format_amount(-6020.72, "₹ crore"), "-₹6,020.72 crore")


if __name__ == "__main__":
    unittest.main()
