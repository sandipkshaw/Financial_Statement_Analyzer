"""LangGraph-powered agent router for the financial analysis system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from comparison_agent import ComparisonAgent
from narrative_qa_agent import NarrativeQAAgent
from trend_red_flag_agent import TrendRedFlagAgent

Route = Literal["trend", "retrieval", "narrative", "comparison"]


class RouteDecision(BaseModel):
    route: Route = Field(description="Agent to execute for the user's request")
    reason: str = Field(description="Brief reason for selecting the agent")


class AgentState(TypedDict, total=False):
    question: str
    history: str
    left_company: str
    right_company: str
    route: Route
    route_reason: str
    answer: str
    sources: tuple[str, ...]
    warnings: tuple[str, ...]


class AgenticFinancialOrchestrator:
    """Route each chat request to the most appropriate specialized agent."""

    def __init__(
        self,
        data_dir: Path,
        comparison_agent: ComparisonAgent,
        narrative_agent: NarrativeQAAgent,
        trend_agent: TrendRedFlagAgent,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.data_dir = data_dir
        self.comparison_agent = comparison_agent
        self.narrative_agent = narrative_agent
        self.trend_agent = trend_agent
        self.router = None
        if api_key:
            self.router = ChatOpenAI(model=model, temperature=0, api_key=api_key)
        self.graph = self._build_graph()

    def _deterministic_route(self, question: str) -> tuple[Route, str]:
        text = f" {question.casefold()} "

        source_terms = ("source", "evidence", "retrieve", "retrieval", "grounding", "citation", "cite")
        comparison_terms = (
            "compare", "comparison", "versus", " vs. ", " vs ",
            "between ", "side by side", "which company", "better than",
        )
        trend_terms = (
            "trend", "red flag", "margin compression", "liquidity deterioration",
            "leverage stress", "dso", "inventory build", "accrual-quality",
            "across periods", "over the last", "year over year", "yoy", "period-over-period",
            "how did", "changed", "change in", "increased", "decreased", "movement in",
        )

        # Sources are specialized even when the source request mentions a trend.
        if any(term in text for term in source_terms):
            return "retrieval", "The request asks for grounded sources or evidence."

        # Only explicit cross-company requests belong to the comparison agent.
        if any(term in text for term in comparison_terms):
            return "comparison", "The request explicitly asks for comparison."
        if " difference " in text or " differences " in text:
            return "comparison", "The request asks for differences between companies or metrics."

        if any(term in text for term in trend_terms):
            return "trend", "The request asks for trends or red flags."
        return "narrative", "No specialized comparison, trend, or source intent was detected."

    def route(self, state: AgentState) -> dict[str, object]:
        fallback_route, fallback_reason = self._deterministic_route(state["question"])
        if self.router is None:
            return {"route": fallback_route, "route_reason": fallback_reason}

        # Deterministic intent is stronger than the LLM for clear specialist requests.
        # This also prevents a previous prompt or agent from biasing routing.
        if fallback_route != "narrative":
            return {"route": fallback_route, "route_reason": fallback_reason}

        prompt = (
            "Choose exactly one route from the intent of the CURRENT QUESTION only.\n"
            "Do not use the previous agent, previous route, or earlier questions to decide.\n"
            "comparison: explicitly compare two named companies or their metrics.\n"
            "trend: trends, period changes, growth directions, or red flags for a company.\n"
            "retrieval: ask for retrieved evidence, grounding, sources, or citations.\n"
            "narrative: answer one company's current statement or metric question.\n\n"
            f"Conversation context (informational only):\n{state.get('history', '')}\n"
            f"Selected primary company: {state['left_company']}\n"
            f"Current question: {state['question']}"
        )
        try:
            decision = self.router.with_structured_output(RouteDecision).invoke(prompt)
            return {"route": decision.route, "route_reason": decision.reason}
        except Exception:
            return {"route": fallback_route, "route_reason": fallback_reason}

    @staticmethod
    def _select_trend_flags(flags: tuple[object, ...], question: str) -> tuple[object, ...]:
        """Keep a trend response focused on indicators named in the prompt."""
        text = question.casefold()
        aliases = {
            "Accrual-quality gap": ("accrual", "cash conversion", "pat", "cash flow"),
            "DSO climbing": ("dso", "receivable", "collection", "days sales outstanding"),
            "Margin compression": ("margin", "profitability", "roe", "return on assets"),
            "Leverage stress": ("leverage", "debt", "capital adequacy", "gearing"),
            "Inventory build-up": ("inventory", "stock build", "stock build-up"),
            "Liquidity deterioration": ("liquidity", "liquid", "casa", "current ratio", "cash ratio"),
        }
        selected = tuple(
            flag for flag in flags
            if any(alias in text for alias in aliases.get(flag.pattern, ()))
        )
        return selected or flags

    def _generate_grounded_response(
        self,
        state: AgentState,
        role: str,
        evidence: str,
        deterministic_fallback: str,
    ) -> str:
        if getattr(self, "router", None) is None:
            return deterministic_fallback
        prompt = (
            f"You are the {role}. Answer the user's current question only.\n"
            "Do not repeat a generic overview, and do not introduce unsupported figures.\n"
            "Use the supplied evidence and requested company as facts; explain calculations only when present.\n"
            "Cite financial figures using the source labels supplied.\n"
            "Keep the answer focused, concise, and directly responsive to the current question.\n\n"
            f"Conversation context:\n{state.get('history', '') or 'None'}\n"
            f"Company: {state['left_company']}\n"
            f"Current question:\n{state['question']}\n\n"
            f"Grounded evidence:\n{evidence}"
        )
        try:
            response = self.router.invoke(prompt)
            text = getattr(response, "content", None) or str(response)
            return text.strip() or deterministic_fallback
        except Exception:
            return deterministic_fallback

    def _trend_node(self, state: AgentState) -> dict[str, object]:
        company = state["left_company"]
        flags = self.trend_agent.analyze_company(company)
        flags = self._select_trend_flags(flags, state["question"])
        evidence = "\n".join(flag.to_markdown() for flag in flags)
        fallback = (
            f"### {company}\n{evidence}\n\n"
            "These indicators are prompts for review, not conclusions of misconduct or financial distress."
        )
        sources = tuple(dict.fromkeys(flag.source for flag in flags))
        answer = self._generate_grounded_response(
            state, "Trend & Red Flag Agent", evidence, fallback
        )
        return {"answer": answer, "sources": sources, "warnings": ()}

    def _retrieval_node(self, state: AgentState) -> dict[str, object]:
        grounding = self.comparison_agent.grounding_agent
        if grounding is None:
            raise RuntimeError("The retrieval/grounding agent was not initialized.")
        context = grounding.build_context(state["left_company"], state["question"])
        evidence_lines = [
            f"- [{item.company} | {item.citation()}] {item.text.replace(chr(10), ' ')}"
            for item in context.evidence
        ]
        fact_lines = [fact.to_context() for fact in context.computed_facts]
        fallback_parts = ["### Retrieved grounded evidence"]
        fallback_parts.extend(evidence_lines)
        if fact_lines:
            fallback_parts.extend(["### Deterministic facts", *fact_lines])
        if not evidence_lines and not fact_lines:
            fallback_parts.append("No supporting evidence was retrieved for this request.")
        fallback = "\n".join(fallback_parts)
        evidence = "\n".join([*evidence_lines, *fact_lines]) or "No supporting evidence was retrieved."
        answer = self._generate_grounded_response(
            state, "Retrieval & Grounding Agent", evidence, fallback
        )
        return {
            "answer": answer,
            "sources": tuple(dict.fromkeys(item.citation() for item in context.evidence)),
            "warnings": (),
        }

    def _narrative_node(self, state: AgentState) -> dict[str, object]:
        result = self.narrative_agent.answer(
            state["left_company"],
            state["question"],
            history=state.get("history", ""),
        )
        return {"answer": result.answer, "sources": result.sources, "warnings": ()}

    def _comparison_node(self, state: AgentState) -> dict[str, object]:
        if state["left_company"] == state["right_company"]:
            raise ValueError("Comparison requests require two different selected companies.")
        result = self.comparison_agent.compare(
            state["left_company"],
            state["right_company"],
            state["question"],
            history=state.get("history", ""),
        )
        return {"answer": result.answer, "sources": result.sources, "warnings": result.warnings}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("trend", self._trend_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("narrative", self._narrative_node)
        graph.add_node("comparison", self._comparison_node)
        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {"trend": "trend", "retrieval": "retrieval", "narrative": "narrative", "comparison": "comparison"},
        )
        graph.add_edge("trend", END)
        graph.add_edge("retrieval", END)
        graph.add_edge("narrative", END)
        graph.add_edge("comparison", END)
        return graph.compile()

    def invoke(
        self,
        question: str,
        left_company: str,
        right_company: str,
        history: str = "",
    ) -> AgentState:
        return self.graph.invoke(
            {
                "question": question,
                "history": history,
                "left_company": left_company,
                "right_company": right_company,
            }
        )
