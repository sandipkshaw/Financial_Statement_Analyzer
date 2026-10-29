"""Streamlit chat interface for side-by-side financial comparisons."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError

from agentic_orchestrator import AgenticFinancialOrchestrator
from comparison_agent import ComparisonAgent, extract_company_profile
from narrative_qa_agent import NarrativeQAAgent
from trend_red_flag_agent import TrendRedFlagAgent

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")
api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_comparison_agent(data_dir: str) -> ComparisonAgent:
    return ComparisonAgent(Path(data_dir), api_key=api_key)


@st.cache_resource
def get_trend_agent(data_dir: str) -> TrendRedFlagAgent:
    return TrendRedFlagAgent(Path(data_dir))


@st.cache_resource
def get_narrative_agent(data_dir: str) -> NarrativeQAAgent:
    comparison_agent = get_comparison_agent(data_dir)
    if comparison_agent.grounding_agent is None:
        raise RuntimeError("The retrieval/grounding agent was not initialized.")
    return NarrativeQAAgent(comparison_agent.grounding_agent, comparison_agent.profiles, api_key=api_key)


@st.cache_resource
def get_orchestrator(data_dir: str) -> AgenticFinancialOrchestrator:
    comparison_agent = get_comparison_agent(data_dir)
    return AgenticFinancialOrchestrator(
        data_dir=Path(data_dir),
        comparison_agent=comparison_agent,
        narrative_agent=get_narrative_agent(data_dir),
        trend_agent=get_trend_agent(data_dir),
        api_key=api_key,
    )


def set_pending_question(question: str) -> None:
    st.session_state.pending_question = question


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


def suggested_questions_for_intent(
    intent_question: str,
    left_name: str,
    right_name: str,
) -> tuple[str, str, str]:
    intent = intent_question.lower()
    if any(word in intent for word in ("compare", "comparison", "versus", " vs ")):
        return (
            f"Which company has stronger profitability: {left_name} or {right_name}?",
            f"Compare the leverage and liquidity of {left_name} and {right_name}",
            f"What are the biggest risks in the {left_name} and {right_name} comparison?",
        )
    if any(word in intent for word in ("cash flow", "cashflow", "free cash")):
        return (
            f"Why did operating cash flow change for {left_name}?",
            f"How did free cash flow trend for {left_name}?",
            f"What are the main cash flow red flags for {left_name}?",
        )
    if any(word in intent for word in ("red flag", "risk", "warning", "concern")):
        return (
            f"Which liquidity risks are visible for {left_name}?",
            f"Which leverage indicators changed for {left_name}?",
            f"Summarize the key red flags and supporting figures for {left_name}",
        )
    if any(word in intent for word in ("margin", "profitability", "return", "earnings", "revenue")):
        return (
            f"What caused profitability to change for {left_name}?",
            f"How did {left_name}'s margins trend over the available periods?",
            f"Compare revenue growth with operating profit for {left_name}",
        )
    if any(word in intent for word in ("trend", "over time", "period", "change")):
        return (
            f"What were {left_name}'s strongest improvements?",
            f"What deteriorated most for {left_name}?",
            f"Which red flags support the recent trends for {left_name}?",
        )
    return (
        f"What are the key financial trends for {left_name}?",
        f"What are the main red flags for {left_name}?",
        f"Compare {left_name} and {right_name}",
    )


def render_suggested_questions(
    left_name: str,
    right_name: str,
    key_suffix: str,
    intent_question: str = "",
    show_heading: bool = False,
) -> None:
    if show_heading:
        st.markdown("**Suggested questions**")
    suggestion_columns = st.columns(3)
    suggested_questions = suggested_questions_for_intent(intent_question, left_name, right_name)
    for index, (column, suggested_question) in enumerate(zip(suggestion_columns, suggested_questions)):
        column.button(
            suggested_question,
            key=f"suggested_{key_suffix}_{index}",
            on_click=set_pending_question,
            args=(suggested_question,),
            width="stretch",
        )


def available_companies() -> list[str]:
    companies: list[str] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        profile = extract_company_profile(path.read_text(encoding="utf-8"), path.name)
        companies.append(profile.name)
    if len(companies) < 2:
        raise RuntimeError(f"At least two company dossiers are required in {DATA_DIR}.")
    return companies


st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide",
)
st.title("Financial Statement Analyzer")
st.caption(
    "Ask questions about the primary company or compare two companies using their financial reports."
)

try:
    companies = available_companies()
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"Could not load the company dossiers: {error}")
    st.stop()

with st.sidebar:
    st.header("Companies")
    left_name = st.selectbox("Primary company", companies, index=0)
    right_options = [name for name in companies if name != left_name]
    right_name = st.selectbox("Comparison company", right_options, index=0)
    st.divider()
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        clear_chat()
        st.rerun()
    st.divider()
    st.markdown(
        "The comparison uses local Markdown dossiers and OpenAI embeddings/chat "
        "models. Set `OPENAI_API_KEY` in your environment or a `.env` file."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

input_placeholder = (
    f"Ask about {left_name}, compare {left_name} and {right_name}, or request trend analysis..."
)
pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input(input_placeholder, key="conversation_question")
question = pending_question or typed_question

if not st.session_state.messages and not question:
    render_suggested_questions(
        left_name,
        right_name,
        "conversation_initial",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for warning in message.get("warnings", []):
            st.warning(warning)
        st.markdown(str(message.get("content") or ""))
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not os.getenv("OPENAI_API_KEY"):
            error_message = (
                "OPENAI_API_KEY is not set. Add it to the process environment or "
                "a `.env` file in the project folder, then restart Streamlit."
            )
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Configuration required:** {error_message}"}
            )
        else:
            try:
                history = "\n".join(
                    f"{message['role']}: {str(message.get('content') or '')}"
                    for message in st.session_state.messages[-8:-1]
                )
                with st.spinner("Routing the request to the required agent..."):
                    result = get_orchestrator(str(DATA_DIR)).invoke(
                        question=question,
                        left_company=left_name,
                        right_company=right_name,
                        history=history,
                    )
                answer = result["answer"]
                result_sources = list(result.get("sources", ()))
                result_warnings = list(result.get("warnings", ()))
                st.caption(f"Agent: {result['route'].replace('_', ' ').title()}")
                for warning in result_warnings:
                    st.warning(warning)
                st.markdown(answer)
                with st.expander("Grounded sources"):
                    for source in result_sources:
                        st.markdown(f"- `{source}`")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "warnings": result_warnings,
                        "sources": result_sources,
                    }
                )
            except (OpenAIError, OSError, RuntimeError, ValueError) as error:
                error_message = f"Agent request failed: {error}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

if (
    len(st.session_state.messages) >= 2
    and st.session_state.messages[-1]["role"] == "assistant"
):
    last_user_message = next(
        (
            message["content"]
            for message in reversed(st.session_state.messages)
            if message["role"] == "user"
        ),
        "",
    )
    render_suggested_questions(
        left_name,
        right_name,
        f"conversation_{len(st.session_state.messages)}",
        intent_question=last_user_message,
    )

