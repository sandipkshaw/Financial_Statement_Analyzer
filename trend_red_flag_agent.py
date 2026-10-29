"""Deterministic trend and red-flag analysis for financial dossiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from currency_formatting import format_amount


@dataclass(frozen=True)
class MetricObservation:
    metric: str
    period: str
    value: float
    unit: str


@dataclass(frozen=True)
class RedFlag:
    pattern: str
    status: str
    summary: str
    observations: tuple[MetricObservation, ...]
    source: str

    def to_markdown(self) -> str:
        def format_observation(item: MetricObservation) -> str:
            if re.search(r"crore|\bcr\b", item.unit.casefold().replace(".", "")):
                return f"{item.metric} {item.period}={format_amount(item.value, item.unit)}"
            return f"{item.metric} {item.period}={item.value:g}{item.unit}"

        evidence = "; ".join(format_observation(item) for item in self.observations) or "required inputs unavailable"
        return f"- **{self.pattern} — {self.status}:** {self.summary} Evidence: {evidence}. Source: `{self.source}`"


class TrendRedFlagAgent:
    """Compute supportable trend flags without using a language model."""

    PATTERN_NAMES = (
        "Accrual-quality gap",
        "DSO climbing",
        "Margin compression",
        "Leverage stress",
        "Inventory build-up",
        "Liquidity deterioration",
    )

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @staticmethod
    def _number(value: str) -> float | None:
        text = value.replace(",", "")
        text = text.replace("(", "-").replace(")", "")
        if not text or re.search(r"n/?a|unavailable|not stated", text, re.I):
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text.replace("−", "-"))
        return float(match.group(0)) if match else None

    def _extract_tables(self, text: str, source: str) -> tuple[dict[str, dict[str, float]], dict[str, str], str]:
        values: dict[str, dict[str, float]] = {}
        units: dict[str, str] = {}
        section = "Dossier"
        reporting_currency = re.search(
            r"\*\*Reporting Currency\*\*:\s*([^\n]+)", text, re.IGNORECASE
        )
        currency_text = reporting_currency.group(1) if reporting_currency else ""
        if re.search(r"INR|Rupee", currency_text, re.IGNORECASE):
            currency_symbol = "₹"
        elif re.search(r"USD|Dollar", currency_text, re.IGNORECASE):
            currency_symbol = "$"
        elif re.search(r"EUR|Euro", currency_text, re.IGNORECASE):
            currency_symbol = "€"
        elif re.search(r"GBP|Pound", currency_text, re.IGNORECASE):
            currency_symbol = "£"
        else:
            currency_symbol = ""
        monetary_unit = f"{currency_symbol} crore" if currency_symbol and "crore" in currency_text.casefold() else ""
        lines = text.splitlines()
        for index, line in enumerate(lines):
            heading = re.match(r"^#{1,6}\s+(.+)$", line.strip())
            if heading:
                section = heading.group(1).strip()
            if not line.startswith("|"):
                continue
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            period_positions = {
                position: re.sub(r"\s+\([^)]*\)", "", header[position]).strip()
                for position in range(len(header))
                if re.search(r"FY\d{2}", header[position])
            }
            if not period_positions or index + 1 >= len(lines):
                continue
            separator = lines[index + 1]
            for data_line in lines[index + 2:]:
                if not data_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in data_line.strip().strip("|").split("|")]
                if len(cells) != len(header):
                    continue
                metric_candidates = [
                    (position, re.sub(r"<[^>]+>|[*]", "", cells[position]).strip())
                    for position in range(len(cells))
                    if position not in period_positions
                    and re.search(r"[A-Za-z]", cells[position])
                ]
                if not metric_candidates:
                    continue
                descriptive_positions = [
                    position for position, heading in enumerate(header)
                    if re.search(r"description|line item|ratio name", heading, re.I)
                ]
                descriptive = next(
                    ((position, cleaned.strip()) for position, cleaned in metric_candidates
                     if position in descriptive_positions and cleaned.strip()),
                    None,
                )
                metric_position, metric = descriptive or max(metric_candidates, key=lambda item: len(item[1]))
                metric = metric.strip()
                normalized = self._clean(metric)
                if not normalized or normalized in {"ratio category", "ratio name"}:
                    continue
                periods = values.setdefault(normalized, {})
                is_ratio_table = "ratio" in section.casefold() or "ratio" in " ".join(header).casefold()
                has_percent = "%" in " ".join(cells)
                units.setdefault(normalized, "%" if has_percent or is_ratio_table else monetary_unit)
                for position, period in period_positions.items():
                    if position >= len(cells):
                        continue
                    value = self._number(cells[position])
                    canonical = re.sub(r"\s+", "", period)
                    if value is not None:
                        periods[canonical] = value
        return values, units, section

    def analyze_file(self, path: Path) -> tuple[RedFlag, ...]:
        text = path.read_text(encoding="utf-8")
        values, units, _ = self._extract_tables(text, path.name)
        source = path.name

        def exact(*aliases: str) -> tuple[str, dict[str, float]] | None:
            for metric, periods in values.items():
                if metric in {self._clean(alias) for alias in aliases}:
                    return metric, periods
            return None

        def contains(*phrases: str) -> tuple[str, dict[str, float]] | None:
            for metric, periods in values.items():
                if all(phrase in metric for phrase in phrases):
                    return metric, periods
            return None

        def observations(metric: str, periods: dict[str, float], prior: str = "FY23", latest: str = "FY24") -> tuple[MetricObservation, ...]:
            return tuple(
                MetricObservation(metric, period, periods[period], units.get(self._clean(metric), ""))
                for period in (prior, latest)
                if period in periods
            )

        def monetary_text(metric: str, periods: dict[str, float], period: str) -> str:
            unit = units.get(self._clean(metric), "")
            return format_amount(periods[period], unit)

        flags: list[RedFlag] = []

        # Accrual quality: PAT rises while net operating cash flow falls.
        pat = contains("net profit", "period") or exact("Net Profit for the Period (PAT)", "Profit After Tax") or contains("net profit")
        cfo = contains("net cash flow", "operating activities") or contains("net operating cash flow") or exact("Net Cash Flow from Operating Activities (CFO)", "Net Operating Cash Flow", "Operating Cash Flow")
        if pat and cfo and "FY23" in pat[1] and "FY24" in pat[1] and "FY23" in cfo[1] and "FY24" in cfo[1]:
            gap = pat[1]["FY24"] > pat[1]["FY23"] and cfo[1]["FY24"] < cfo[1]["FY23"]
            if gap:
                accrual_summary = (
                    "PAT increased from "
                    f"{monetary_text(pat[0], pat[1], 'FY23')} to {monetary_text(pat[0], pat[1], 'FY24')} "
                    "while net operating cash flow declined from "
                    f"{monetary_text(cfo[0], cfo[1], 'FY23')} to {monetary_text(cfo[0], cfo[1], 'FY24')} "
                    "between FY23 and FY24."
                )
            else:
                accrual_summary = (
                    "No PAT/CFO divergence met the flag criteria between FY23 and FY24: PAT was "
                    f"{monetary_text(pat[0], pat[1], 'FY23')} to {monetary_text(pat[0], pat[1], 'FY24')}, "
                    "while net operating cash flow was "
                    f"{monetary_text(cfo[0], cfo[1], 'FY23')} to {monetary_text(cfo[0], cfo[1], 'FY24')}."
                )
            flags.append(RedFlag("Accrual-quality gap", "flagged" if gap else "not flagged", accrual_summary, observations(pat[0], pat[1]) + observations(cfo[0], cfo[1]), f"{source} § Profit & Loss; Cash Flow Statement"))
        else:
            flags.append(RedFlag("Accrual-quality gap", "qualified unavailable", "Required PAT or CFO values are unavailable, so this indicator cannot be established.", (), f"{source} § P&L; Cash Flow Statement"))

        # These bank dossiers do not contain a consistent DSO/input series.
        receivables = contains("receivables")
        revenue = contains("revenue")
        if receivables and revenue:
            r23, r24 = receivables[1].get("FY23"), receivables[1].get("FY24")
            s23, s24 = revenue[1].get("FY23"), revenue[1].get("FY24")
            if r23 and r24 and s23 and s24:
                d23, d24 = r23 / s23 * 365, r24 / s24 * 365
                flagged = d24 > d23
                flags.append(RedFlag("DSO climbing", "flagged" if flagged else "not flagged", "Estimated DSO increased FY23 to FY24." if flagged else "Estimated DSO did not increase between FY23 and FY24.", (MetricObservation("Estimated DSO", "FY23", d23, " days"), MetricObservation("Estimated DSO", "FY24", d24, " days")), source))
                returner = None
            else:
                flags.append(RedFlag("DSO climbing", "qualified unavailable", "Receivables or revenue is missing for both FY23 and FY24.", (), source))
        else:
            flags.append(RedFlag("DSO climbing", "qualified unavailable", "The dossier does not provide a consistent receivables/revenue series for DSO calculation.", (), source))

        # Margin compression: supported percentage-point declines in named margin ratios.
        margin = contains("net interest margin") or contains("operating margin") or contains("net profit margin")
        if margin and "FY23" in margin[1] and "FY24" in margin[1]:
            change = margin[1]["FY24"] - margin[1]["FY23"]
            flagged = change < -0.10
            flags.append(RedFlag("Margin compression", "flagged" if flagged else "not flagged", f"{margin[0]} moved by {change:.2f} percentage points FY23 to FY24." , observations(margin[0], margin[1]), f"{source} § Key Banking & Financial Ratios"))
        else:
            flags.append(RedFlag("Margin compression", "qualified unavailable", "No comparable margin series is available for FY23 and FY24.", (), f"{source} § Key Banking & Financial Ratios"))

        # Leverage: borrowings growth relative to net worth growth.
        borrowings = contains("borrowings")
        net_worth = contains("shareholders funds net worth") or contains("net worth")
        if borrowings and net_worth and all(p in borrowings[1] for p in ("FY23", "FY24")) and all(p in net_worth[1] for p in ("FY23", "FY24")):
            debt_growth = (borrowings[1]["FY24"] / borrowings[1]["FY23"] - 1) if borrowings[1]["FY23"] else 0
            equity_growth = (net_worth[1]["FY24"] / net_worth[1]["FY23"] - 1) if net_worth[1]["FY23"] else 0
            flagged = debt_growth > equity_growth + 0.10
            if flagged:
                leverage_summary = (
                    "Borrowings increased from "
                    f"{monetary_text(borrowings[0], borrowings[1], 'FY23')} to {monetary_text(borrowings[0], borrowings[1], 'FY24')}, "
                    "faster than net worth, which increased from "
                    f"{monetary_text(net_worth[0], net_worth[1], 'FY23')} to {monetary_text(net_worth[0], net_worth[1], 'FY24')}; "
                    "the growth gap exceeded 10 percentage points."
                )
            else:
                leverage_summary = (
                    "Borrowings did not grow more than 10 percentage points faster than net worth: borrowings were "
                    f"{monetary_text(borrowings[0], borrowings[1], 'FY23')} and {monetary_text(borrowings[0], borrowings[1], 'FY24')}, "
                    "while net worth was "
                    f"{monetary_text(net_worth[0], net_worth[1], 'FY23')} and {monetary_text(net_worth[0], net_worth[1], 'FY24')}."
                )
            flags.append(RedFlag("Leverage stress", "flagged" if flagged else "not flagged", leverage_summary, observations(borrowings[0], borrowings[1]) + observations(net_worth[0], net_worth[1]), f"{source} § Balance Sheet"))
        else:
            flags.append(RedFlag("Leverage stress", "qualified unavailable", "Borrowings and shareholders' funds/net worth are required for this indicator.", (), f"{source} § Balance Sheet"))

        # Inventory is uncommon in banking dossiers and is intentionally not inferred.
        inventory = contains("inventories") or contains("inventory")
        sales = contains("total interest earned") or contains("revenue") or contains("net operating income")
        if inventory and sales and all(p in inventory[1] for p in ("FY23", "FY24")) and all(p in sales[1] for p in ("FY23", "FY24")):
            inventory_growth = inventory[1]["FY24"] / inventory[1]["FY23"] - 1 if inventory[1]["FY23"] else 0
            sales_growth = sales[1]["FY24"] / sales[1]["FY23"] - 1 if sales[1]["FY23"] else 0
            flagged = inventory_growth > sales_growth + 0.10
            if flagged:
                inventory_summary = (
                    "Inventory increased from "
                    f"{monetary_text(inventory[0], inventory[1], 'FY23')} to {monetary_text(inventory[0], inventory[1], 'FY24')} "
                    "while sales increased from "
                    f"{monetary_text(sales[0], sales[1], 'FY23')} to {monetary_text(sales[0], sales[1], 'FY24')}; "
                    "inventory growth exceeded sales growth by more than 10 percentage points."
                )
            else:
                inventory_summary = "Inventory did not grow more than 10 percentage points faster than sales."
            flags.append(RedFlag("Inventory build-up", "flagged" if flagged else "not flagged", inventory_summary, observations(inventory[0], inventory[1]) + observations(sales[0], sales[1]), source))
        else:
            flags.append(RedFlag("Inventory build-up", "qualified unavailable", "The dossier has no consistent inventory and sales series for FY23-FY24.", (), source))

        # Liquidity deterioration from the CASA or current-liquidity ratio.
        liquidity = contains("casa ratio") or contains("current ratio")
        capital = contains("capital adequacy ratio")
        if liquidity and all(p in liquidity[1] for p in ("FY23", "FY24")):
            change = liquidity[1]["FY24"] - liquidity[1]["FY23"]
            flagged = change < -1.0
            flags.append(RedFlag("Liquidity deterioration", "flagged" if flagged else "not flagged", f"{liquidity[0]} changed by {change:.2f} percentage points FY23 to FY24." , observations(liquidity[0], liquidity[1]), f"{source} § Key Banking & Financial Ratios"))
        elif capital and all(p in capital[1] for p in ("FY23", "FY24")):
            change = capital[1]["FY24"] - capital[1]["FY23"]
            flagged = change < -1.0
            flags.append(RedFlag("Liquidity deterioration", "flagged" if flagged else "not flagged", f"{capital[0]} changed by {change:.2f} percentage points FY23 to FY24." , observations(capital[0], capital[1]), f"{source} § Key Banking & Financial Ratios"))
        else:
            flags.append(RedFlag("Liquidity deterioration", "qualified unavailable", "No comparable CASA/current-ratio or capital-adequacy series is available for FY23-FY24.", (), f"{source} § Key Banking & Financial Ratios"))

        return tuple(flags)

    def analyze_company(self, company: str) -> tuple[RedFlag, ...]:
        matches = [path for path in sorted(self.data_dir.glob("*.md")) if company.casefold() in path.read_text(encoding="utf-8").casefold()]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely locate a dossier for {company}.")
        return self.analyze_file(matches[0])
