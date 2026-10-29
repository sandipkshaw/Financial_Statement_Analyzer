"""FAISS retrieval and deterministic grounding facts for financial dossiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import faiss
import numpy as np

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class RetrievedEvidence:
    company: str
    source: str
    section: str
    text: str

    def citation(self) -> str:
        return f"{self.source} § {self.section}"


@dataclass(frozen=True)
class DeterministicFact:
    company: str
    metric: str
    period_from: str
    period_to: str
    from_value: float
    to_value: float
    change_percent: float

    def to_context(self) -> str:
        return (
            f"DETERMINISTIC FACT | {self.company} | {self.metric}: "
            f"{self.period_from}={self.from_value:g}; {self.period_to}={self.to_value:g}; "
            f"year-over-year change={self.change_percent:.2f}% | formula=({self.period_to} - "
            f"{self.period_from}) / {self.period_from} * 100"
        )


@dataclass(frozen=True)
class GroundingContext:
    evidence: tuple[RetrievedEvidence, ...]
    computed_facts: tuple[DeterministicFact, ...]

    def render(self) -> str:
        blocks = [
            f"[{item.company or 'Unknown company'} | {item.citation()}]\n"
            f"{item.text or ''}"
            for item in self.evidence
        ]
        blocks.extend(fact.to_context() for fact in self.computed_facts)
        return "\n\n---\n\n".join(str(block) for block in blocks if block is not None)


class RetrievalGroundingAgent:
    """Index dossiers in FAISS and assemble cited evidence plus calculated facts."""

    def __init__(
        self,
        documents_by_company: dict[str, list[str]],
        metadata_by_company: dict[str, list[dict[str, str | int]]],
        source_texts_by_company: dict[str, str],
        embed: EmbeddingFunction,
        result_count: int = 6,
    ) -> None:
        self._documents = documents_by_company
        self._metadata = metadata_by_company
        self._source_texts = source_texts_by_company
        self._embed = embed
        self._result_count = result_count
        self._indexes: dict[str, faiss.Index] = {}
        self._index_dossiers()

    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        profile_for_text: Callable[[str, str], object],
        chunk_markdown: Callable[[str, object], list[dict[str, str]]],
        embed: EmbeddingFunction,
        result_count: int = 6,
    ) -> "RetrievalGroundingAgent":
        documents: dict[str, list[str]] = {}
        metadata: dict[str, list[dict[str, str | int]]] = {}
        source_texts: dict[str, str] = {}
        for path in sorted(data_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            profile = profile_for_text(text, path.name)
            company = getattr(profile, "name")
            chunks = chunk_markdown(text, profile)
            documents[company] = [chunk["text"] for chunk in chunks]
            metadata[company] = [
                {"source": path.name, "section": chunk["section"]}
                for chunk in chunks
            ]
            source_texts[company] = text
        return cls(documents, metadata, source_texts, embed, result_count)

    def _index_dossiers(self) -> None:
        for company, documents in self._documents.items():
            embeddings = np.asarray(self._embed(documents), dtype=np.float32)
            if embeddings.size == 0:
                raise ValueError(f"No embeddings were created for {company}.")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._indexes[company] = index

    def retrieve(self, company: str, query: str, query_embedding: list[float] | None = None) -> list[RetrievedEvidence]:
        query_embedding = query_embedding or self._embed([query])[0]
        query_vector = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        count = min(self._result_count, self._indexes[company].ntotal)
        if count == 0:
            return []
        _, indices = self._indexes[company].search(query_vector, count)
        evidence: list[RetrievedEvidence] = []
        for position in indices[0]:
            index = int(position)
            if index < 0:
                continue
            item_metadata = self._metadata[company][index]
            evidence.append(
                RetrievedEvidence(
                    company=company,
                    source=str(item_metadata["source"]),
                    section=str(item_metadata["section"]),
                    text=self._documents[company][index],
                )
            )
        return evidence

    @staticmethod
    def _number(raw: str) -> float | None:
        cleaned = raw.strip().replace(",", "")
        cleaned = cleaned.replace("—", "-").replace("–", "-")
        if not cleaned or cleaned in {"-", "N/A", "NA", "Unavailable"}:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        return float(match.group(0)) if match else None

    def compute_year_over_year_facts(self, company: str, query: str) -> tuple[DeterministicFact, ...]:
        """Calculate supported metric growth directly from Markdown table values."""
        text = self._source_texts[company]
        period_columns = ("FY22", "FY23", "FY24")
        facts: list[DeterministicFact] = []
        query_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", query) if len(word) >= 4}

        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4:
                continue
            metric = re.sub(r"<[^>]+>|\*", "", cells[1] if len(cells) > 3 else cells[0]).strip()
            if not metric:
                continue
            values: dict[str, float] = {}
            for period, raw in zip(period_columns, cells[-3:]):
                value = self._number(raw)
                if value is not None:
                    values[period] = value
            if "FY23" not in values or "FY24" not in values or values["FY23"] == 0:
                continue
            metric_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", metric) if len(word) >= 4}
            if query_words and metric_words.isdisjoint(query_words):
                continue
            change = (values["FY24"] - values["FY23"]) / values["FY23"] * 100
            facts.append(
                DeterministicFact(company, metric, "FY23", "FY24", values["FY23"], values["FY24"], change)
            )
        return tuple(facts[:8])

    def build_context(self, company: str, query: str, evidence_count: int | None = None) -> GroundingContext:
        evidence = self.retrieve(company, query)
        if evidence_count is not None:
            evidence = evidence[:evidence_count]
        facts = self.compute_year_over_year_facts(company, query)
        return GroundingContext(tuple(evidence), facts)
