import argparse
import asyncio
import logging
import os
import re
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import dotenv
import httpx
from pydantic import BaseModel, Field

from tavily import TavilyClient

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from asknews_sdk import AskNewsSDK

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_llm_json(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f'"{match.group(1)}": {nums[0]}' if nums else match.group(0)

    text = re.sub(
        r'"(value|percentile|probability|prediction_in_decimal'
        r'|revised_prediction_in_decimal|multiplier|delta)":\s*"([^"]+)"',
        clean_num,
        text,
    )
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]

    def extract_json_payload(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        if s[0] in "[{":
            depth = 0
            open_char = s[0]
            close_char = "}" if open_char == "{" else "]"
            for idx, ch in enumerate(s):
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        return s[: idx + 1]
        start = min((s.find("{") if "{" in s else len(s)), (s.find("[") if "[" in s else len(s)))
        if start >= len(s):
            return s
        open_char = s[start]
        close_char = "}" if open_char == "{" else "]"
        depth = 0
        for idx in range(start, len(s)):
            ch = s[idx]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return s

    return extract_json_payload(text).strip()


def safe_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    try:
        if isinstance(data, model_cls):
            return data
        if isinstance(data, (str, bytes)):
            s = data.decode() if isinstance(data, bytes) else data
            return model_cls.model_validate_json(sanitize_llm_json(s))
        if isinstance(data, dict):
            return model_cls.model_validate(data)
        return model_cls(**data)
    except Exception as e:
        logger.error(f"MODEL INSTANTIATION FAILED for {model_cls.__name__}: {e}")
        raise


class RawPercentile(BaseModel):
    percentile: float
    value: float


# ═══════════════════════════════════════════════════════════════════════════════
# Research providers & YFinance Live Data
# ═══════════════════════════════════════════════════════════════════════════════

class ExaSearcher:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required for Exa search.")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 6) -> str:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "query": query,
            "numResults": num_results,
            "type": "neural",
            "useAutoprompt": True,
            "category": "news",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = []
                for r in data.get("results", []):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = (r.get("text", "") or "")[:900]
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                return (
                    "[Exa Search Results]\n" + "\n\n".join(results)
                    if results
                    else "[Exa search failed]"
                )
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return "[Exa search failed]"

def _fetch_yfinance_data_sync(ticker: str, as_of: Optional[date] = None) -> str:
    """Fetch market context for `ticker`, never reading past `as_of`.

    `period="3mo"` and `tk.info` are both relative to the wall clock, so the old
    version always returned today's prices and today's 52-week range. In a
    pastcast replay that is future data leaking into a forecast the bot is
    supposed to be making from an earlier moment. Both are now bounded by
    `as_of`, and the 52-week range is computed from the fetched window instead
    of `tk.info`, which has no cutoff parameter at all.
    """
    if not YFINANCE_AVAILABLE: return ""
    try:
        cutoff = as_of or datetime.now(timezone.utc).date()
        tk = yf.Ticker(ticker)
        # 52w window so the range below is computable; end is exclusive in yfinance.
        hist = tk.history(start=cutoff - timedelta(days=372), end=cutoff + timedelta(days=1))
        if hist.empty: return ""
        hist = hist[hist.index.date <= cutoff]
        if hist.empty: return ""
        spot = hist['Close'].iloc[-1]
        window_52 = hist['Close'].tail(252)
        high_52 = f"{float(window_52.max()):.2f}"
        low_52 = f"{float(window_52.min()):.2f}"
        vol = hist['Close'].tail(63).pct_change().dropna().std() * math.sqrt(252)
        monthly_vol = vol * math.sqrt(21/252)
        rw_p10 = spot * math.exp(-1.28 * monthly_vol)
        rw_p90 = spot * math.exp(1.28 * monthly_vol)
        return (f"--- MARKET DATA ({ticker}) as of {cutoff.isoformat()} ---\n"
                f"Spot Price: {spot:.2f}\n"
                f"52-Week Range: {low_52} - {high_52}\n"
                f"Volatility (Annual): {vol:.2%}\n"
                f"Random Walk (1-Mo): P10={rw_p10:.2f}, P50={spot:.2f}, P90={rw_p90:.2f}\n\n")
    except Exception:
        return ""

# ═══════════════════════════════════════════════════════════════════════════════
# Forecasting principles & Engine
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastingPrinciples:
    @staticmethod
    def get_generic_base_rate() -> str:
        return (
            "BASE RATE: In the absence of strong evidence, default to historical frequencies "
            "or uniform priors where applicable. Most novel events have low base rates."
        )

    @staticmethod
    def get_generic_fermi_prompt() -> str:
        return (
            "FERMI GUIDANCE:\n"
            "1) Define the target quantity precisely.\n"
            "2) Decompose into drivers/factors.\n"
            "3) Estimate each factor using available evidence.\n"
            "4) Combine factors algebraically.\n"
            "5) Quantify uncertainty; keep intervals wide unless evidence is strong."
        )

    @staticmethod
    def apply_time_decay(prob: float, close_time: Optional[datetime]) -> float:
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = max(0.0, (close_time - now).total_seconds() / 86400.0)
        if days > 365:
            return 0.85 * prob + 0.15 * 0.5
        if days > 180:
            return 0.90 * prob + 0.10 * 0.5
        if days > 90:
            return 0.95 * prob + 0.05 * 0.5
        return prob

    @staticmethod
    def logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def sigmoid(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    @classmethod
    def extremize_logit(cls, p: float, strength: float) -> float:
        strength = float(np.clip(strength, 0.5, 7.0))
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.01, 0.99))


class DecompositionOutput(BaseModel):
    subquestions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)

class NumericRegime(str, Enum):
    PARTIAL_REVEAL_SUM = "partial_reveal_sum"
    STRUCTURED_TS = "structured_ts"
    GENERIC = "generic"

class PartialRevealExtract(BaseModel):
    known_subtotal: Optional[float] = None
    known_parts: Optional[int] = Field(default=None, ge=0)
    total_parts: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = None

class ReferenceClassExtract(BaseModel):
    reference_totals: List[float] = Field(default_factory=list)
    trend_multiplier: Optional[float] = None
    notes: Optional[str] = None

class BoundedMultiplier(BaseModel):
    multiplier: float

@dataclass
class BotFeatureFlags:
    enable_extremize: bool = True
    enable_decomposition: bool = True
    enable_numeric_regimes: bool = True
    enable_red_team: bool = True
    enable_consistency_check: bool = True

class ReasoningTrace:
    def __init__(self, question_text: str, bot_name: str = "dezzy"):
        self.bot_name = bot_name
        self.question_text = question_text
        self._steps: List[Tuple[str, str]] = []

    def add(self, label: str, detail: str) -> None:
        self._steps.append((label, str(detail)))
        logger.info(f"[{self.bot_name}] {label}: {detail[:200]}")

    def add_narrative(self, run_index: int, text: str) -> None:
        trimmed = (text or "").strip()[:1500]
        if len((text or "").strip()) > 1500:
            trimmed += "\n… [truncated]"
        # ANONYMIZED: Changed LLM to Agent to hide model details
        self._steps.append((f"Agent narrative (run {run_index})", trimmed))
        logger.debug(f"[{self.bot_name}] run {run_index} narrative captured")

    def render(self) -> str:
        lines = [
            f"╔══ [{self.bot_name.upper()}] REASONING TRACE ══════════════════════════════════",
            f"║  Question : {self.question_text[:120]}",
            f"║  Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "╠══ STEPS ══════════════════════════════════════════════════════════════",
        ]
        for i, (label, detail) in enumerate(self._steps, 1):
            lines.append(f"║")
            lines.append(f"║  [{i:02d}] {label}")
            for line in detail.splitlines():
                for chunk in [line[j : j + 110] for j in range(0, max(len(line), 1), 110)]:
                    lines.append(f"║        {chunk}")
        lines.append("║")
        lines.append("╚═══════════════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main bot class — Dezzy
# ═══════════════════════════════════════════════════════════════════════════════

class Dezzy(ForecastBot):
    _CONVICTION_RE = re.compile(
        r"(?i)\b(confirmed|officially|announced|signed|passed|enacted|"
        r"launched|deployed|released|completed|achieved|won|elected|appointed|"
        r"definitively|conclusively|clearly|undeniably|certainly|already\s+has|"
        r"ruled\s+out|impossible\s+by)\b"
    )

    def __init__(
        self,
        *args,
        bot_name: str = "dezzy",
        flags: Optional[BotFeatureFlags] = None,
        runs_per_question: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bot_name = bot_name
        self.flags = flags or BotFeatureFlags()
        self.runs_per_question = int(max(1, runs_per_question))

        self.tavily = (
            TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            if os.getenv("TAVILY_API_KEY") else None
        )
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        # Accept either name. The workflows have always exported ASKNEWS_SECRET
        # while this constructor only read ASKNEWS_CLIENT_SECRET, so self.asknews
        # was silently None on every CI run and _run_asknews_search could only
        # ever return "[AskNews not configured]".
        asknews_id = os.getenv("ASKNEWS_CLIENT_ID")
        asknews_secret = os.getenv("ASKNEWS_CLIENT_SECRET") or os.getenv("ASKNEWS_SECRET")
        self.asknews = (
            AskNewsSDK(client_id=asknews_id, client_secret=asknews_secret)
            if asknews_id and asknews_secret
            else None
        )

        self._research_cache: Dict[str, str] = {}
        self._recent_binary_predictions: List[Tuple[str, float]] = []
        self._active_tournament: str = ""
        self._footprint_counts: Dict[str, int] = {}
        self._no_evidence_count: int = 0

    def set_active_tournament(self, tid: str) -> None:
        self._active_tournament = str(tid).strip().lower()
        logger.info(f"[{self.bot_name}] Active tournament set to: '{self._active_tournament}'")

    # Number of optimized queries searched per question. Tavily is billed per
    # search and is NOT covered by the tournament inference credits, so this is
    # the main recurring cost knob in the research stack.
    MAX_SEARCH_QUERIES = 3

    @staticmethod
    def default_tournament_ids() -> List[str]:
        return ["33022", "market-pulse-26q2"]

    def _llm_config_defaults(self) -> Dict[str, str]:
        return {
            "default":         "openrouter/openai/gpt-5.1",
            "parser":          "openrouter/openai/gpt-4.1-mini",
            "query_optimizer": "openrouter/anthropic/claude-sonnet-4-5",
            "critic":          "openrouter/openai/o3",
            "red_team":        "openrouter/openai/gpt-5.1",
            "decomposer":      "openrouter/anthropic/claude-sonnet-4-5",
            "summarizer":      "openrouter/openai/gpt-4.1",
            "researcher":      "openrouter/openai/gpt-oss-120b",
            "online_researcher": "openrouter/openai/gpt-oss-120b",
            "research_synthesizer": "openrouter/openai/gpt-oss-120b",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Research & YFinance
    # ──────────────────────────────────────────────────────────────────────────

    # Tags that represent actual retrieved web evidence. Model-recall steps are
    # deliberately NOT in here: _run_gptoss_research and _run_mimo_research ask a
    # model to answer from its own weights, which is not a source. Counting them
    # as one used to inflate _research_quality_weight to the "two sources" value
    # on a run that had made a single Tavily call.
    _WEB_SOURCE_TAGS = {
        "tavily": "[Tavily Data]",
        "exa": "[Exa Search Results]",
        "asknews": "[AskNews Results]",
    }
    _MODEL_RECALL_TAGS = {
        "gptoss": "[GPT-OSS Research]",
        "mimo": "[MiMo Research]",
    }

    def _search_footprint(self, research: str) -> str:
        """Distinct web sources that returned evidence, comma separated, or "none".

        A source counts when its success tag is present. The previous version
        disqualified a source if any failure marker for it appeared anywhere in
        the text, which silently dropped Tavily entirely as soon as one of
        several Tavily queries failed.
        """
        research = research or ""
        used = [name for name, tag in self._WEB_SOURCE_TAGS.items() if tag in research]
        return ",".join(used) if used else "none"

    def _model_recall_footprint(self, research: str) -> str:
        research = research or ""
        used = [name for name, tag in self._MODEL_RECALL_TAGS.items() if tag in research]
        return ",".join(used) if used else "none"

    # A forecast made with zero retrieved web evidence is shrunk at least this far
    # toward the reference class (0.5 for binary, uniform for multiple choice) and
    # is never extremized.
    NO_EVIDENCE_SHRINK = 0.45
    # Numeric equivalent: scale deviations from the median outward by this factor.
    NO_EVIDENCE_WIDEN = 1.60

    def _note_research_footprint(self, research: str) -> str:
        """Return the web-source footprint. Never raises.

        Was `_ensure_some_research_or_raise`, which raised when the footprint was
        "none". That abstained on the question. Tournament leaderboards sum peer
        scores, so an abstention contributes exactly 0 by construction, and in a
        spot-scored round coverage is all-or-nothing. Decision (2026-08-03): keep
        the forecast, shrink it hard toward the reference class, never extremize
        it, and tag it so it can be filtered out or evaluated later.
        """
        return self._search_footprint(research)

    def _trace_research_footprint(self, trace: "ReasoningTrace", research: str) -> Tuple[str, bool]:
        """Record the footprint on the published rationale and in the run tally.

        The trace line is a stable `key=value` block, semicolon separated, so it
        is a filterable field rather than prose to grep. Keys are not reordered
        or renamed without a note here.
        """
        web = self._note_research_footprint(research)
        recall = self._model_recall_footprint(research)
        quality = self._research_quality_weight(research)
        no_evidence = (web == "none")
        n_web = 0 if no_evidence else len(web.split(","))

        self._footprint_counts[web] = self._footprint_counts.get(web, 0) + 1
        if no_evidence:
            self._no_evidence_count += 1

        trace.add(
            "Research footprint",
            f"research_footprint={web}; web_sources={n_web}; model_recall={recall}; "
            f"research_quality={quality:.2f}; no_evidence={'true' if no_evidence else 'false'}",
        )
        if no_evidence:
            trace.add(
                "Research footprint",
                "No web evidence retrieved. Forecasting anyway rather than abstaining "
                "(an abstention scores 0), with shrink floored at "
                f"{self.NO_EVIDENCE_SHRINK:.2f} and extremize disabled.",
            )
        return web, no_evidence

    def research_footprint_summary(self) -> str:
        total = sum(self._footprint_counts.values())
        if not total:
            return "no_forecasts=0"
        parts = " | ".join(f"{k}={v}" for k, v in sorted(self._footprint_counts.items()))
        pct = 100.0 * self._no_evidence_count / total
        return (f"forecasts={total}; no_evidence={self._no_evidence_count} ({pct:.0f}%); "
                f"by_footprint[{parts}]")

    # Research quality by number of distinct web sources that returned evidence.
    # The previous mapping was {1: 0.65, 2: 0.82}.get(n, 0.7), which is NOT
    # monotone: three sources scored 0.7, BELOW two sources at 0.82. Wiring Exa
    # and AskNews in without fixing this would have made better research report
    # as lower quality, producing heavier shrink and less extremizing.
    _QUALITY_BY_WEB_SOURCES = {0: 0.25, 1: 0.60, 2: 0.78, 3: 0.88}
    _QUALITY_MAX = 0.92

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        n = 0 if srcs == "none" else len(srcs.split(","))
        return self._QUALITY_BY_WEB_SOURCES.get(n, self._QUALITY_MAX)

    def _grounding_instructions(self) -> str:
        return (
            "You are Dezzy, an evidence-based forecasting assistant. "
            f"Current date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. "
            "Do not rely on model memory or training cutoff for recent events. "
            "Ground every forecast in the supplied research, direct evidence, and any credible source context provided. "
            "If the evidence is weak or stale, say so explicitly and avoid overclaiming."
        )

    def _build_grounded_context(self, question: MetaculusQuestion, research: str, premortem: str) -> str:
        return clean_indents(f"""
            Question: {getattr(question, 'question_text', '')}
            Resolution criteria: {getattr(question, 'resolution_criteria', '')}

            Grounding instructions:
            {self._grounding_instructions()}

            Research briefing:
            {research}

            Premortem analysis:
            {premortem}
        """)

    async def _run_premortem_analysis(self, question: MetaculusQuestion, research: str) -> str:
        try:
            llm = self.get_llm("critic", "llm")
            prompt = clean_indents(f"""
                Imagine the correct answer is the opposite of your best guess.
                List 3 plausible, evidence-based reasons you could be wrong.
                {self._grounding_instructions()}
                Question: {question.question_text}
                Research: {research[:4000]}
            """)
            return (await llm.invoke(prompt)).strip() or "Premortem unavailable."
        except Exception as e:
            logger.warning(f"Premortem analysis failed: {e}")
            return "Premortem unavailable."

    async def _decompose_question(self, question: MetaculusQuestion) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition: return None
        try:
            llm = self.get_llm("decomposer", "llm")
            prompt = clean_indents(f"""
                Decompose the forecasting question into 3-6 subquestions, key entities, and key metrics.
                Return ONLY JSON: {{"subquestions":[...], "key_entities":[...], "key_metrics":[...]}}
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
            """)
            raw = await llm.invoke(prompt)
            return safe_model(DecompositionOutput, raw) 
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            return None

    async def _optimize_search_query(self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]) -> List[str]:
        llm = self.get_llm("query_optimizer", "llm")
        extra = ""
        if decomp and decomp.subquestions: extra += "\nSubquestions:\n" + "\n".join(f"- {s}" for s in decomp.subquestions[:6])
        if decomp and decomp.key_entities: extra += "\nEntities:\n" + ", ".join(decomp.key_entities[:12])
        prompt = f"Rewrite this forecasting question into 3 precise web search queries.\nQuestion: {question.question_text}\n{extra}\nOutput ONLY JSON list: [\"q1\",\"q2\",\"q3\"]"
        try:
            resp = await llm.invoke(prompt)
            queries = json.loads(sanitize_llm_json(resp))
            cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            return cleaned[:3] if cleaned else [question.question_text[:160]]
        except Exception:
            return [question.question_text[:160]]

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily: return "[Tavily not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.tavily.search(query=query, search_depth="advanced", max_results=6, include_answer=False, include_raw_content=False))
            context = "\n".join([f"Source: {r.get('url','')}\nContent: {r.get('content','')}" for r in response.get("results", [])])
            return f"[Tavily Data]\n{context}" if context.strip() else "[Tavily search failed]"
        except Exception as e:
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher: return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _run_asknews_search(self, query: str) -> str:
        if not self.asknews: return "[AskNews not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.asknews.news.search_news(query=query, n_articles=6, hours_back=24*7, strategy="latest news"))
            results = []
            for article in response.articles:
                title = article.title
                url = article.url
                snippet = (article.summary or "")[:900]
                results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
            return f"[AskNews Results]\n" + "\n\n".join(results) if results else "[AskNews search failed]"
        except Exception as e:
            logger.error(f"AskNews search failed: {e}")
            return "[AskNews search failed]"

    async def _run_mimo_research(self, question: MetaculusQuestion, research: str) -> str:
        try:
            llm = GeneralLlm(model="openrouter/xiaomi/mimo-v2-pro", temperature=0.1)
            prompt = clean_indents(f"""
                You are a research assistant. Research this forecasting question using your knowledge and provide:
                1. Key factual findings.
                2. Signals supporting YES/higher outcome.
                3. Signals supporting NO/lower outcome.
                Question: {question.question_text}
                Existing research: {research[:2000] if research else 'None'}
            """)
            response = await llm.invoke(prompt)
            return f"[MiMo Research]\n{response.strip()}"
        except Exception as e:
            logger.error(f"MiMo research failed: {e}")
            return "[MiMo research failed]"

    async def _run_gptoss_research(self, question: MetaculusQuestion, research: str) -> str:
        try:
            llm = GeneralLlm(model="openrouter/openai/gpt-oss-120b", temperature=0.1)
            prompt = clean_indents(f"""
                You are a research assistant. Research this forecasting question using the Tavily results and your knowledge and provide:
                1. Key factual findings.
                2. Signals supporting YES/higher outcome.
                3. Signals supporting NO/lower outcome.
                Question: {question.question_text}
                Existing research: {research[:2000] if research else 'None'}
            """)
            response = await llm.invoke(prompt)
            return f"[GPT-OSS Research]\n{response.strip()}"
        except Exception as e:
            logger.error(f"GPT-OSS research failed: {e}")
            return "[GPT-OSS research failed]"

    async def _summarize_research(self, question: MetaculusQuestion, raw_research: str) -> str:
        llm = self.get_llm("summarizer", "llm")
        prompt = clean_indents(f"""
            You are summarizing web research for a forecaster. Write exactly 3 sentences covering:
            1. The most relevant factual finding. 2. Strongest signal for YES/higher. 3. Strongest signal for NO/lower.
            Question: {question.question_text}\nResearch:\n{raw_research[:3000]}
        """)
        try: return (await llm.invoke(prompt)).strip()
        except Exception: return "[Research summary unavailable]"

    async def run_research(self, question: MetaculusQuestion) -> str:
        cache_key = getattr(question, "page_url", None) or question.question_text[:80]
        if cache_key in self._research_cache: return self._research_cache[cache_key]

        # LIVE FINANCIAL DATA INJECTION
        fin_data = ""
        is_finance = "market" in self._active_tournament or any(k in question.question_text.lower() for k in ["stock", "price", "market cap", "revenue", "gdp", "inflation"])
        if is_finance:
            try:
                extract_prompt = f"Extract the single most relevant Yahoo Finance ticker for this question (e.g. AAPL, ^GSPC, CL=F). If it is a macroeconomic indicator without a direct ticker, reply NONE.\nQuestion: {question.question_text}"
                ticker = await self.get_llm("parser", "llm").invoke(extract_prompt)
                ticker = ticker.strip().upper()
                if ticker and ticker != "NONE":
                    loop = asyncio.get_running_loop()
                    as_of = datetime.now(timezone.utc).date()
                    fin_data = await loop.run_in_executor(None, _fetch_yfinance_data_sync, ticker, as_of)
            except Exception as e:
                logger.warning(f"Ticker extraction failed: {e}")

        decomp = await self._decompose_question(question)
        queries = await self._optimize_search_query(question, decomp)
        queries = [q for q in queries if q][: self.MAX_SEARCH_QUERIES] or [question.question_text[:160]]

        # One search per optimized query. Previously the queries were joined with
        # " OR " into a single Tavily call at max_results=6, so three separately
        # optimized queries competed for six result slots between them.
        web_tasks = [self._run_tavily_search(q) for q in queries]
        web_tasks.append(self._run_exa_search(queries[0]))
        web_tasks.append(self._run_asknews_search(queries[0]))

        web_results = await asyncio.gather(*web_tasks, return_exceptions=True)
        web_cleaned = [
            f"[Search failed: {str(res)}]" if isinstance(res, BaseException) else res
            for res in web_results
        ]
        web_text = "\n".join(web_cleaned).strip()
        trace_srcs = self._search_footprint(web_text)
        logger.info(
            f"[{self.bot_name}] research footprint: web={trace_srcs} "
            f"(queries={len(queries)}, quality={self._research_quality_weight(web_text):.2f})"
        )

        # Model-recall pass runs last so it can actually see the retrieved evidence.
        # It used to be handed "" while its own prompt referred to Tavily results.
        # It does not count toward the web-source total; see _search_footprint.
        recall = await self._run_gptoss_research(question, web_text)
        cleaned = web_cleaned + [recall]
        
        research = (
            f"{fin_data}"
            f"{ForecastingPrinciples.get_generic_base_rate()}\n\n"
            f"{ForecastingPrinciples.get_generic_fermi_prompt()}\n\n"
            f"{chr(10).join(cleaned).strip()}"
        )
        self._note_research_footprint(research)
        self._research_cache[cache_key] = research
        return research

    # ──────────────────────────────────────────────────────────────────────────
    # Core Aggregation & Confidence Gate
    # ──────────────────────────────────────────────────────────────────────────

    # Confidence gate tuning.
    # "spread" is NOT the same quantity on every path:
    #   binary / multiple-choice -> absolute probability spread across runs, bounded 0..1
    #   numeric                  -> relative interval width (p90-p10)/|median|, unbounded
    # so only the probability paths may be compared against SPREAD_LIMIT_PROB.
    SPREAD_LIMIT_PROB = 0.20
    MAX_LOW_CONFIDENCE_SHRINK = 0.30

    def _spring_ai_confidence_shrink(
        self,
        trace: ReasoningTrace,
        spread: float,
        quality: float,
        kind: str = "probability",
    ) -> float:
        """Grade confidence WITHOUT aborting the question.

        Returns extra shrink-toward-ignorance alpha in [0, MAX_LOW_CONFIDENCE_SHRINK].

        This used to raise RuntimeError. The raise was captured per question by
        forecast_on_tournament(..., return_exceptions=True) (main.py:1153), but
        log_report_summary() re-raises on any captured exception
        (forecasting_tools/forecast_bots/forecast_bot.py, raise_errors=True by
        default), so one gated question exited the process with code 1 and marked
        the whole scheduled run failed.
        """
        is_spring_ai = self._active_tournament in ["33022", str(MetaculusClient().CURRENT_AI_COMPETITION_ID)]
        if not is_spring_ai:
            return 0.0

        if kind != "probability":
            trace.add(
                "Spring AI Confidence Gate",
                f"numeric relative width={spread:.4f}, quality={quality:.2f} - recorded only; "
                f"a relative interval width is not comparable to a probability spread, so it is not gated.",
            )
            return 0.0

        trace.add("Spring AI Confidence Gate", f"Evaluating... spread={spread:.4f}, quality={quality:.2f}")

        alpha = 0.0
        reasons: List[str] = []
        if spread > self.SPREAD_LIMIT_PROB:
            over = (spread - self.SPREAD_LIMIT_PROB) / self.SPREAD_LIMIT_PROB
            alpha += 0.15 * over
            reasons.append(f"spread {spread:.2f} > {self.SPREAD_LIMIT_PROB:.2f}")
        if quality < 0.65:
            alpha += 0.15
            reasons.append(f"research quality {quality:.2f} < 0.65")

        if not reasons:
            trace.add("Spring AI Confidence Gate", "PASSED. High confidence adjudged.")
            return 0.0

        alpha = float(np.clip(alpha, 0.0, self.MAX_LOW_CONFIDENCE_SHRINK))
        trace.add(
            "Spring AI Confidence Gate",
            f"LOW CONFIDENCE ({'; '.join(reasons)}). Forecasting anyway with extra shrink alpha={alpha:.2f}.",
        )
        return alpha

    @staticmethod
    def _median(xs: List[float]) -> float:
        xs = [float(x) for x in xs if np.isfinite(float(x))]
        if not xs: return 0.5
        xs.sort()
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])

    @staticmethod
    def _shrink_to_half(p: float, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.clip((1 - alpha) * p + alpha * 0.5, 0.0, 1.0))

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        if not getattr(question, "close_time", None): return 0.15
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        return 0.20 if days_to_close > 180 else 0.10

    def _agreement_strength(self, probs: List[float]) -> float:
        if not probs: return 0.0
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return float(np.clip(1.0 - (spread / 0.30), 0.0, 1.0))

    def _extremize_strength(self, research: str, probs: List[float], question: MetaculusQuestion) -> float:
        if not self.flags.enable_extremize: return 1.0
        quality = self._research_quality_weight(research)
        agree = self._agreement_strength(probs)
        base = 1.0 + 0.45 * (quality - 0.5) * 2.0 * agree
        close_time = getattr(question, "close_time", None)
        if close_time:
            days = (close_time - datetime.now(timezone.utc)).days
            if days < 60: base = 1.0 + (base - 1.0) * 0.6
        return float(np.clip(base, 0.95, 1.6))

    @staticmethod
    def _extremize_gate(p: float) -> bool:
        p = float(p)
        return 0.02 < p < 0.98 and p != 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Red-team & Consistency
    # ──────────────────────────────────────────────────────────────────────────

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float, trace: ReasoningTrace) -> float:
        if not self.flags.enable_red_team:
            trace.add("Red-team", "SKIPPED")
            return initial_pred
        self._note_research_footprint(research)
        llm = self.get_llm("red_team", "llm")
        try:
            raw = await llm.invoke(clean_indents(f"""
                Find the SINGLE STRONGEST argument that the current forecast is WRONG.
                Question: {question.question_text}
                Current forecast: {initial_pred:.2%}
                Research: {research[:2500]}
                Output JSON only on last line: {{"revised_prediction_in_decimal": 0.XX, "counter_argument": "one sentence summary"}}
            """))
            trace.add("Red-team narrative", (raw or "").strip()[:800])
            last_line = [l.strip() for l in raw.splitlines() if l.strip()][-1]
            parsed = json.loads(sanitize_llm_json(last_line))
            result = float(np.clip(float(parsed.get("revised_prediction_in_decimal", initial_pred)), 0.0, 1.0))
            trace.add("Red-team result", f"revised={result:.4f} (Δ={result - initial_pred:+.4f}) | counter: \"{parsed.get('counter_argument', '')}\"")
            return result
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            return initial_pred

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float, trace: ReasoningTrace) -> bool:
        if not self.flags.enable_consistency_check or len(self._recent_binary_predictions) < 2: return True
        recent_summary = "\n".join([f"Q: {qt} → Pred: {p:.2%}" for qt, p in self._recent_binary_predictions[-3:]])
        llm = self.get_llm("parser", "llm")
        prompt = f"Is this new forecast logically consistent with prior forecasts?\nNew: {question.question_text} → {proposed_pred:.2%}\nPrior:\n{recent_summary}\nAnswer YES or NO only."
        try:
            response = await llm.invoke(prompt)
            result = "YES" in (response or "").upper()
            trace.add("Consistency check", f"{'PASSED' if result else 'FAILED — applying shrink'}")
            return result
        except Exception:
            return True

    # ──────────────────────────────────────────────────────────────────────────
    # Numeric parsing & Regimes
    # ──────────────────────────────────────────────────────────────────────────

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(f"Extract numeric distribution list of objects (percentile, value) for 0.1, 0.2, 0.4, 0.6, 0.8, 0.9. Units: {question.unit_of_measure}")

    @staticmethod
    def _extract_percentile_block(text: str) -> str:
        m = re.search(r"(Percentile\s*10\s*:.*?Percentile\s*90\s*:.*?)(?:\n\s*\n|$)", text or "", flags=re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip()
        return "\n".join(line.strip() for line in (text or "").splitlines() if re.search(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, flags=re.IGNORECASE)).strip()

    @staticmethod
    def _normalize_raw_percentiles(raw: List[RawPercentile]) -> List[Percentile]:
        return [Percentile(percentile=max(0.0, min(1.0, float(rp.percentile) / 100.0 if float(rp.percentile) > 1.0 else float(rp.percentile))), value=float(rp.value)) for rp in raw]

    @staticmethod
    def _require_standard_percentiles(pcts: List[Percentile]) -> List[Percentile]:
        by = {round(float(p.percentile), 3): p for p in pcts}
        return [by[round(r, 3)] for r in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]] if not [r for r in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9] if round(r, 3) not in by] else []

    @staticmethod
    def _enforce_monotone(pcts: List[Percentile]) -> List[Percentile]:
        pcts = sorted(pcts, key=lambda x: float(x.percentile))
        for i in range(1, len(pcts)):
            if pcts[i].value <= pcts[i - 1].value: pcts[i].value = pcts[i - 1].value + 1e-6
        return pcts

    @staticmethod
    def _bounds_fallback(question: NumericQuestion) -> List[Percentile]:
        lo, hi = float(question.lower_bound), float(question.upper_bound)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = 0.0, 1.0
        w = {0.1: 0.05, 0.2: 0.15, 0.4: 0.40, 0.6: 0.60, 0.8: 0.85, 0.9: 0.95}
        return Dezzy._enforce_monotone([Percentile(percentile=p, value=lo + (hi - lo) * w[p]) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]])

    @staticmethod
    def _median_from_40_60(pcts: List[Percentile]) -> float:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        return 0.5 * (by[0.4] + by[0.6]) if 0.4 in by and 0.6 in by else float(sorted(pcts, key=lambda x: x.percentile)[len(pcts) // 2].value) if pcts else 0.0

    @staticmethod
    def _format_pcts(pcts: List[Percentile]) -> str:
        return " | ".join(f"P{int(round(float(p.percentile) * 100))}={p.value:.6g}" for p in pcts)

    @staticmethod
    def _clip_to_question_bounds(pcts: List[Percentile], question: NumericQuestion) -> List[Percentile]:
        """Keep values inside the question's declared bounds. Widening can push a
        percentile past them; the structured-timeseries path already did this
        inline, this is the same logic under a name."""
        hi = question.nominal_upper_bound if getattr(question, "nominal_upper_bound", None) is not None else getattr(question, "upper_bound", None)
        lo = question.nominal_lower_bound if getattr(question, "nominal_lower_bound", None) is not None else getattr(question, "lower_bound", None)
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            return pcts
        if not (np.isfinite(lo_f) and np.isfinite(hi_f) and hi_f > lo_f):
            return pcts
        for p in pcts:
            p.value = float(np.clip(float(p.value), lo_f, hi_f))
        return Dezzy._enforce_monotone(pcts)

    @staticmethod
    def _widen_percentiles(pcts: List[Percentile], factor: float) -> List[Percentile]:
        """Scale deviations from the central value outward. Numeric analogue of
        shrinking a probability toward 0.5: it keeps the location estimate and
        widens the uncertainty."""
        if factor <= 1.0 or not pcts:
            return pcts
        center = Dezzy._median_from_40_60(pcts)
        widened = [
            Percentile(percentile=float(p.percentile), value=float(center + (float(p.value) - center) * factor))
            for p in pcts
        ]
        return Dezzy._enforce_monotone(widened)

    @staticmethod
    def _p10_p90(pcts: List[Percentile]) -> Tuple[Optional[float], Optional[float]]:
        """Extract P10 and P90 values from a list of Percentile objects."""
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        return by.get(0.1), by.get(0.9)

    async def _parse_numeric_percentiles_robust(self, question: NumericQuestion, text: str, stage: str) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
        for attempt, source in enumerate([text, self._extract_percentile_block(text)], 1):
            if not source: continue
            try:
                raw: List[RawPercentile] = await structure_output(source, list[RawPercentile], model=parser_llm, additional_instructions=self._numeric_parsing_instructions(question), num_validation_samples=1)
                std = self._require_standard_percentiles(self._normalize_raw_percentiles(raw))
                if std: return self._enforce_monotone(std)
            except Exception: pass
        try:
            reformatted = await parser_llm.invoke(f"Rewrite into EXACTLY 6 Percentile lines.\nText:\n{text}")
            raw3: List[RawPercentile] = await structure_output(self._extract_percentile_block(reformatted) or reformatted, list[RawPercentile], model=parser_llm, additional_instructions=self._numeric_parsing_instructions(question), num_validation_samples=1)
            std3 = self._require_standard_percentiles(self._normalize_raw_percentiles(raw3))
            if std3: return self._enforce_monotone(std3)
        except Exception: pass
        return self._bounds_fallback(question)

    def _extract_date_range_generic(self, text: str) -> Optional[Tuple[date, date]]:
        m = re.search(r"\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*-\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*", text or "", flags=re.IGNORECASE)
        if not m: return None
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                s, e = datetime.strptime(m.group(1), fmt).date(), datetime.strptime(m.group(2), fmt).date()
                return (e, s) if s > e else (s, e)
            except Exception: continue
        return None

    def _detect_numeric_regime(self, question: NumericQuestion, research: str) -> NumericRegime:
        if not self.flags.enable_numeric_regimes: return NumericRegime.GENERIC
        if any(c in (research or "").lower() for c in ["sum to", "subtotal", "observed"]) and self._extract_date_range_generic(question.question_text or ""): return NumericRegime.PARTIAL_REVEAL_SUM
        dr = self._extract_date_range_generic(question.question_text or "")
        return NumericRegime.STRUCTURED_TS if dr and 2 <= (dr[1] - dr[0]).days + 1 <= 31 else NumericRegime.GENERIC

    async def _llm_extract_partial_reveal(self, question: NumericQuestion, research: str) -> PartialRevealExtract:
        raw = await self.get_llm("parser", "llm").invoke(f"Return JSON: {{\"known_subtotal\": null, \"known_parts\": null, \"total_parts\": null, \"notes\": null}}\nQuestion: {question.question_text}\nResearch:\n{research}")
        return safe_model(PartialRevealExtract, raw)

    async def _llm_extract_reference_class(self, question: NumericQuestion, research: str) -> ReferenceClassExtract:
        raw = await self.get_llm("parser", "llm").invoke(f"Return JSON: {{\"reference_totals\": [], \"trend_multiplier\": null, \"notes\": null}}\nQuestion: {question.question_text}\nResearch:\n{research}")
        return safe_model(ReferenceClassExtract, raw)

    async def _bounded_multiplier(self, question: NumericQuestion, research: str, baseline: float, *, lo: float, hi: float) -> float:
        raw = await self.get_llm("critic", "llm").invoke(f"Return JSON: {{\"multiplier\": 1.00}}\nBaseline: {baseline}\nResearch:\n{research}\nRules: must be in [{lo:.6f}, {hi:.6f}]")
        return float(np.clip(float(getattr(safe_model(BoundedMultiplier, raw), "multiplier")), lo, hi))

    def _mult_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        return (0.98, 1.02) if h <= 21 else (0.96, 1.04) if h <= 60 else (0.92, 1.08)

    def _horizon_days_from_text(self, question: NumericQuestion) -> Optional[int]:
        dr = self._extract_date_range_generic(question.question_text or "")
        return (dr[1] - dr[0]).days + 1 if dr else None

    @staticmethod
    def _normal_percentiles_from_mean_sd(mean: float, sd: float) -> List[Percentile]:
        z = {0.1: -1.2816, 0.2: -0.8416, 0.4: -0.2533, 0.6: 0.2533, 0.8: 0.8416, 0.9: 1.2816}
        return Dezzy._enforce_monotone([Percentile(percentile=p, value=float(mean + z[p] * sd)) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]])

    # ──────────────────────────────────────────────────────────────────────────
    # Model calls + multi-run
    # ──────────────────────────────────────────────────────────────────────────

    async def _single_model_forecast(self, question: MetaculusQuestion, research: str, run_index: int, trace: ReasoningTrace, grounded_context: Optional[str] = None) -> Any:
        self._note_research_footprint(research)
        model = "openrouter/openai/o3"
        llm = GeneralLlm(model=model, temperature=self._get_temperature(question))
        context = grounded_context or self._build_grounded_context(question, research, "Premortem unavailable.")

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster. Think step by step before giving your answer.
                {self._grounding_instructions()}
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Context:
                {context}
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                OUTPUT ONLY VALID JSON on the very last line: {{"prediction_in_decimal": 0.50}}
            """))
            trace.add_narrative(run_index, "\n".join(line for line in (raw or "").splitlines() if not line.strip().startswith("{")).strip())
            return await structure_output(sanitize_llm_json(raw), BinaryPrediction, model=self.get_llm("parser", "llm"), num_validation_samples=1)

        if isinstance(question, MultipleChoiceQuestion):
            schema_example = json.dumps({"predicted_options": [{"option_name": opt, "probability": round(1 / len(question.options), 3)} for opt in question.options]})
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster.
                {self._grounding_instructions()}
                Question: {question.question_text}
                Options: {question.options}
                Context:
                {context}
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                OUTPUT ONLY VALID JSON on the very last line: {schema_example}
            """))
            trace.add_narrative(run_index, "\n".join(line for line in (raw or "").splitlines() if not line.strip().startswith("{")).strip())
            return await structure_output(sanitize_llm_json(raw), PredictedOptionList, model=self.get_llm("parser", "llm"), num_validation_samples=1)

        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster.
                {self._grounding_instructions()}
                Question: {question.question_text}
                Units: {question.unit_of_measure or "Not stated"} | Bounds: [{lower}, {upper}]
                Context:
                {context}
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                The LAST thing you write is EXACTLY these 6 lines:
                Percentile 10: XX
                ...
                Percentile 90: XX
            """))
            narrative_lines = []
            for line in (raw or "").splitlines():
                if re.match(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, re.IGNORECASE): break
                narrative_lines.append(line)
            trace.add_narrative(run_index, "\n".join(narrative_lines).strip())
            return await self._parse_numeric_percentiles_robust(question, raw, stage=f"run{run_index}")

        raise TypeError(f"Unsupported question type: {type(question)}")

    async def _multi_run(self, question: MetaculusQuestion, research: str, trace: ReasoningTrace, grounded_context: Optional[str] = None) -> List[Any]:
        outs: List[Any] = []
        for i in range(self.runs_per_question):
            try:
                outs.append(await self._single_model_forecast(question, research, i + 1, trace, grounded_context=grounded_context))
            except Exception as e:
                logger.warning(f"run {i+1}/{self.runs_per_question} failed: {e}")
                trace.add(f"Run {i+1}", f"FAILED: {e}")
        return outs

    def _fallback_binary_prediction(self, question: BinaryQuestion, trace: ReasoningTrace) -> ReasonedPrediction[float]:
        trace.add("Fallback prediction", "All independent binary runs failed; returning neutral probability 0.50.")
        final_p = 0.5
        trace.add("★ FINAL PREDICTION", f"{final_p:.4f}  ({final_p:.1%})")
        self._recent_binary_predictions.append((question.question_text[:120], final_p))
        if len(self._recent_binary_predictions) > 20:
            self._recent_binary_predictions.pop(0)
        return ReasonedPrediction(prediction_value=final_p, reasoning=trace.render())

    def _fallback_mc_prediction(self, question: MultipleChoiceQuestion, trace: ReasoningTrace) -> ReasonedPrediction[PredictedOptionList]:
        uniform = 1.0 / max(1, len(question.options))
        final = [{"option_name": o, "probability": uniform} for o in question.options]
        trace.add("Fallback prediction", "All independent MC runs failed; returning uniform distribution.")
        trace.add("★ FINAL PREDICTION", " | ".join(f"{x['option_name']}={x['probability']:.1%}" for x in final))
        return ReasonedPrediction(prediction_value=safe_model(PredictedOptionList, {"predicted_options": final}), reasoning=trace.render())

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Aggregations & Logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        _web, no_evidence = self._trace_research_footprint(trace, research)

        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)
        quality = self._research_quality_weight(research)

        runs = await self._multi_run(question, research, trace, grounded_context=grounded_context)
        if not runs:
            return self._fallback_binary_prediction(question, trace)

        probs = [float(r.prediction_in_decimal) for r in runs]
        run_med = self._median(probs)
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        
        # Spring AI Confidence Gate (grades confidence, never aborts the question)
        low_conf_shrink = self._spring_ai_confidence_shrink(trace, spread, quality)

        trace.add(f"Multi-run aggregation ({len(probs)} runs)", f"individual={[f'{p:.4f}' for p in probs]} | median={run_med:.4f} | spread={spread:.4f}")
        applied: List[str] = []

        shrink = 0.28 if spread >= 0.20 else (0.22 if quality < 0.70 else 0.12)
        shrink = float(np.clip(shrink + low_conf_shrink, 0.0, 0.60))
        if no_evidence:
            shrink = max(shrink, self.NO_EVIDENCE_SHRINK)
        base_p = self._shrink_to_half(run_med, shrink)
        applied.append(
            f"shrink(alpha={shrink:.2f})"
            + (f"+low-conf({low_conf_shrink:.2f})" if low_conf_shrink > 0 else "")
            + ("+no-evidence-floor" if no_evidence else "")
        )

        red_p = await self._red_team_forecast(question, research, base_p, trace)
        combined = 0.6 * base_p + 0.4 * red_p
        applied.append("blend(red-team)")

        if not await self._check_consistency(question, combined, trace):
            combined = 0.5 * combined + 0.5 * 0.5
            applied.append("consistency-shrink")

        # Dynamic Extremize. Never sharpen a forecast that has no web evidence behind it.
        if no_evidence:
            p_ext = combined
            applied.append("extremize(off: no web evidence)")
            trace.add("Extremize", "SKIPPED - research_footprint=none, so the forecast is not sharpened.")
        elif self.flags.enable_extremize:
            if self._extremize_gate(combined):
                ext_strength = self._extremize_strength(research, probs + [combined], question)
                p_ext = ForecastingPrinciples.extremize_logit(combined, ext_strength)
                applied.append(f"extremize(x{ext_strength:.2f})")
            else:
                p_ext = combined
                applied.append("extremize(gated-off)")
        else:
            p_ext = combined

        p_time = ForecastingPrinciples.apply_time_decay(p_ext, getattr(question, "close_time", None))
        if abs(p_time - p_ext) > 1e-6: applied.append("time-decay")

        try:
            if hasattr(self, "apply_bayesian_calibration"):
                p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
                if abs(p_cal - p_time) > 1e-6: applied.append("bayes-calibration")
            else: p_cal = p_time
        except Exception: p_cal = p_time

        final_p = float(np.clip(p_cal, 0.01, 0.99))
        trace.add("Pipeline summary", f"controls applied: {', '.join(applied)}")
        trace.add("★ FINAL PREDICTION", f"{final_p:.4f}  ({final_p:.1%})")

        self._recent_binary_predictions.append((question.question_text[:120], final_p))
        if len(self._recent_binary_predictions) > 20: self._recent_binary_predictions.pop(0)

        return ReasonedPrediction(prediction_value=final_p, reasoning=trace.render())

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        _web, no_evidence = self._trace_research_footprint(trace, research)

        trace.add("Research summary", await self._summarize_research(question, research))
        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)
        quality = self._research_quality_weight(research)

        runs = await self._multi_run(question, research, trace, grounded_context=grounded_context)
        if not runs:
            return self._fallback_mc_prediction(question, trace)

        opt_names = list(question.options)
        per_opt: Dict[str, List[float]] = {o: [] for o in opt_names}
        for r in runs:
            try: cur = {o.option_name: float(o.probability) for o in r.predicted_options}
            except Exception: continue
            for o in opt_names: per_opt[o].append(float(cur.get(o, 0.0)))

        med_probs = {o: self._median(per_opt[o]) if per_opt[o] else 0.0 for o in opt_names}
        
        # Calculate MC spread for Confidence Gate
        max_spread = max([(max(per_opt[o]) - min(per_opt[o])) for o in opt_names if per_opt[o]] + [0])
        low_conf_shrink = self._spring_ai_confidence_shrink(trace, max_spread, quality)

        uniform = 1.0 / max(1, len(opt_names))
        alpha = 0.10 if quality >= 0.75 else 0.18
        alpha = float(np.clip(alpha + low_conf_shrink, 0.0, 0.60))
        if no_evidence:
            alpha = max(alpha, self.NO_EVIDENCE_SHRINK)
            trace.add("No-evidence handling", f"shrink toward uniform floored at alpha={alpha:.2f}; extremize not used on this path.")
        shrunk = {o: (1 - alpha) * med_probs[o] + alpha * uniform for o in opt_names}

        total = float(sum(max(0.0, v) for v in shrunk.values()))
        final = [{"option_name": o, "probability": uniform} for o in opt_names] if total <= 0 else [{"option_name": o, "probability": float(np.clip(shrunk[o] / total, 0.0, 1.0))} for o in opt_names]

        trace.add("★ FINAL PREDICTION", " | ".join(f"{x['option_name']}={x['probability']:.1%}" for x in final))
        return ReasonedPrediction(prediction_value=safe_model(PredictedOptionList, {"predicted_options": final}), reasoning=trace.render())

    async def _run_forecast_on_numeric_generic(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        _web, no_evidence = self._trace_research_footprint(trace, research)

        trace.add("Research summary", await self._summarize_research(question, research))
        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)
        quality = self._research_quality_weight(research)

        runs = await self._multi_run(question, research, trace, grounded_context=grounded_context)
        if not runs: raise RuntimeError("All numeric runs failed.")

        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_pct: Dict[float, List[float]] = {p: [] for p in required}

        for r in runs:
            try:
                for pct in r:
                    p, v = round(float(pct.percentile) / 100.0 if float(pct.percentile) > 1.0 else float(pct.percentile), 3), float(pct.value)
                    if p in per_pct and np.isfinite(v): per_pct[p].append(v)
            except Exception: continue

        agg: List[Percentile] = []
        for p in required:
            vals = per_pct.get(round(p, 3), [])
            if vals: agg.append(Percentile(percentile=p, value=float(self._median(vals))))
            else:
                pcts = self._bounds_fallback(question)
                return ReasonedPrediction(prediction_value=NumericDistribution.from_question(pcts, question), reasoning=trace.render())

        agg = self._enforce_monotone(agg)
        if no_evidence:
            agg = self._clip_to_question_bounds(self._widen_percentiles(agg, self.NO_EVIDENCE_WIDEN), question)
            trace.add(
                "No-evidence handling",
                f"research_footprint=none; interval widened x{self.NO_EVIDENCE_WIDEN:.2f} about the median "
                f"-> {self._format_pcts(agg)}",
            )

        # Calculate numeric relative spread for Spring AI Gate
        p10, p90 = self._p10_p90(agg)
        med = self._median_from_40_60(agg)
        rel_spread = (p90 - p10) / abs(med) if p10 is not None and p90 is not None and med != 0 else 0.0
        self._spring_ai_confidence_shrink(trace, rel_spread, quality, kind="relative_width")

        trace.add("★ FINAL PREDICTION", self._format_pcts(agg))
        return ReasonedPrediction(prediction_value=NumericDistribution.from_question(agg, question), reasoning=trace.render())

    async def _forecast_numeric_partial_reveal(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Research summary", await self._summarize_research(question, research))
        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)
        
        try: ex = await self._llm_extract_partial_reveal(question, research)
        except Exception: return await self._run_forecast_on_numeric_generic(question, research)
        if ex.known_subtotal is None: return await self._run_forecast_on_numeric_generic(question, research)

        known = float(ex.known_subtotal)
        if not np.isfinite(known) or known <= 0: return await self._run_forecast_on_numeric_generic(question, research)

        remainder_baseline = 0.75 * known
        horizon = self._horizon_days_from_text(question)
        mult = await self._bounded_multiplier(question, research, remainder_baseline, lo=self._mult_bounds_for_horizon(horizon)[0], hi=self._mult_bounds_for_horizon(horizon)[1])
        total_mean = known + remainder_baseline * mult
        sd = max(0.10 * total_mean, 0.05 * known)

        pcts = self._normal_percentiles_from_mean_sd(total_mean, sd)
        for p in pcts:
            if p.value < known: p.value = known
        pcts = self._enforce_monotone(pcts)
        trace.add("★ FINAL PREDICTION", self._format_pcts(pcts))
        return ReasonedPrediction(prediction_value=NumericDistribution.from_question(pcts, question), reasoning=trace.render())

    async def _forecast_numeric_structured_ts(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Research summary", await self._summarize_research(question, research))
        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)

        baseline = 0.5 * (float(question.lower_bound) + float(question.upper_bound))
        try:
            ref = await self._llm_extract_reference_class(question, research)
            refs = [float(x) for x in (ref.reference_totals or []) if np.isfinite(float(x)) and float(x) > 0]
            if refs:
                baseline = float(np.median(refs))
                if ref.trend_multiplier is not None and 0.85 <= float(ref.trend_multiplier) <= 1.15: baseline *= float(ref.trend_multiplier)
        except Exception: pass

        horizon = self._horizon_days_from_text(question)
        mult = await self._bounded_multiplier(question, research, baseline, lo=self._mult_bounds_for_horizon(horizon)[0], hi=self._mult_bounds_for_horizon(horizon)[1])
        mean = baseline * mult

        lo, hi = float(question.lower_bound), float(question.upper_bound)
        width = hi - lo if np.isfinite(hi - lo) and hi > lo else max(1.0, abs(mean))
        sd = float(np.clip(0.10 * abs(mean) + 0.05 * width, 1e-9, 0.35 * abs(mean) + 1e-9))

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        for p in pcts:
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo: p.value = float(np.clip(p.value, lo, hi))
        pcts = self._enforce_monotone(pcts)
        trace.add("★ FINAL PREDICTION", self._format_pcts(pcts))
        return ReasonedPrediction(prediction_value=NumericDistribution.from_question(pcts, question), reasoning=trace.render())

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        self._note_research_footprint(research)
        if not self.flags.enable_numeric_regimes: return await self._run_forecast_on_numeric_generic(question, research)

        regime = self._detect_numeric_regime(question, research)
        if regime == NumericRegime.PARTIAL_REVEAL_SUM:
            try: return await self._forecast_numeric_partial_reveal(question, research)
            except Exception: return await self._run_forecast_on_numeric_generic(question, research)
        if regime == NumericRegime.STRUCTURED_TS:
            try: return await self._forecast_numeric_structured_ts(question, research)
            except Exception: return await self._run_forecast_on_numeric_generic(question, research)

        return await self._run_forecast_on_numeric_generic(question, research)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="dezzy: Tavily+Exa, OpenRouter, YFinance, and Spring AI Gate")
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    parser.add_argument("--bot-name", type=str, default="dezzy")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent agent runs to aggregate per question")
    parser.add_argument("--no-extremize", action="store_true")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-numeric-regimes", action="store_true")
    parser.add_argument("--no-red-team", action="store_true")
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--research-check", action="store_true",
                        help="Execute the web search clients only, print what each returned, and exit nonzero if a configured client returns no evidence. No Metaculus or LLM calls.")
    parser.add_argument("--research-check-query", type=str,
                        default="US Federal Reserve interest rate decision September 2026",
                        help="Query used by --research-check.")

    args = parser.parse_args()

    if args.research_check:
        # Executes the three web search clients directly against whatever creds are
        # in the environment. No Metaculus call, no LLM call, no forecast. Exits
        # nonzero if a client that looks configured fails to return evidence, so it
        # works as a merge gate rather than as something to read and shrug at.
        probe = Dezzy.__new__(Dezzy)
        Dezzy.__init__(
            probe,
            research_reports_per_question=1, predictions_per_research_report=1,
            publish_reports_to_metaculus=False, bot_name="research-check",
        )
        query = args.research_check_query
        print(f"\n=== research stack check ===\nquery: {query!r}\n")

        configured = {
            "tavily":  ("TAVILY_API_KEY", probe.tavily is not None),
            "exa":     ("EXA_API_KEY", probe.exa_searcher is not None),
            "asknews": ("ASKNEWS_CLIENT_ID + ASKNEWS_CLIENT_SECRET/ASKNEWS_SECRET", probe.asknews is not None),
        }
        for name, (envs, ok) in configured.items():
            print(f"  client {name:8} configured={str(ok):5}  from {envs}")
        print()

        async def _probe() -> int:
            results = await asyncio.gather(
                probe._run_tavily_search(query),
                probe._run_exa_search(query),
                probe._run_asknews_search(query),
                return_exceptions=True,
            )
            names = ["tavily", "exa", "asknews"]
            blocks, failures = [], []
            for name, res in zip(names, results):
                if isinstance(res, BaseException):
                    print(f"--- {name}: RAISED {type(res).__name__}: {res}")
                    failures.append(name)
                    continue
                text = res or ""
                blocks.append(text)
                first = text.splitlines()[0] if text.splitlines() else "(empty)"
                print(f"--- {name}: {len(text)} bytes | first line: {first}")
                print("    " + (text[:400].replace(chr(10), chr(10) + "    ")))
                print()
                if "not configured" in first:
                    print(f"    -> {name} NOT CONFIGURED")
                elif "failed" in first.lower():
                    print(f"    -> {name} FAILED despite being configured")
                    failures.append(name)

            combined = "\n".join(blocks)
            footprint = probe._search_footprint(combined)
            quality = probe._research_quality_weight(combined)
            n = 0 if footprint == "none" else len(footprint.split(","))
            print(f"=== result ===\nresearch_footprint={footprint}; web_sources={n}; "
                  f"research_quality={quality:.2f}; no_evidence={'true' if n == 0 else 'false'}")

            for name, (_envs, ok) in configured.items():
                if ok and name in failures:
                    print(f"GATE FAIL: {name} is configured but returned no evidence.")
            if failures:
                print(f"\nGATE FAIL: {len(failures)} configured client(s) did not return evidence: {failures}")
                return 1
            if n == 0:
                print("\nGATE FAIL: no web evidence from any client.")
                return 1
            print(f"\nGATE PASS: {n}/3 clients returned evidence.")
            return 0

        raise SystemExit(asyncio.run(_probe()))

    flags = BotFeatureFlags(
        enable_extremize=not args.no_extremize, enable_decomposition=not args.no_decomposition,
        enable_numeric_regimes=not args.no_numeric_regimes, enable_red_team=not args.no_red_team,
        enable_consistency_check=not args.no_consistency,
    )

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("EXA_API_KEY"):
        raise RuntimeError("Set at least one of TAVILY_API_KEY or EXA_API_KEY in your environment.")

    bot = Dezzy(
        research_reports_per_question=1, predictions_per_research_report=1,
        publish_reports_to_metaculus=True, skip_previously_forecasted_questions=False,
        bot_name=args.bot_name, flags=flags, runs_per_question=max(1, int(args.runs)),
    )

    client = MetaculusClient()

    async def run_all():
        if args.mode == "tournament":
            reports: List[Any] = []
            for tid in Dezzy.default_tournament_ids():
                bot.set_active_tournament(tid)
                reports.extend(await bot.forecast_on_tournament(tid, return_exceptions=True))
            return reports

        if args.mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            bot.set_active_tournament(str(client.CURRENT_METACULUS_CUP_ID))
            return await bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)

        bot.skip_previously_forecasted_questions = False
        bot.set_active_tournament("market-pulse-26q2")
        return await bot.forecast_on_tournament("market-pulse-26q2", return_exceptions=True)

    reports = asyncio.run(run_all())
    # Emitted before log_report_summary because that call re-raises on any captured
    # exception. A large no_evidence count means the research stack is broken, and
    # that must be visible in the run log rather than inferred from a leaderboard.
    logger.info(f"[{args.bot_name}] research footprint summary: {bot.research_footprint_summary()}")
    forecast_total = sum(bot._footprint_counts.values())
    if bot._no_evidence_count:
        logger.warning(
            f"[{args.bot_name}] {bot._no_evidence_count}/{forecast_total} forecast(s) published with no retrieved web evidence"
        )
    # A warning line is only a signal if somebody reads it, and nobody reads a green
    # run. If most of a run forecast on nothing, the research stack is broken and the
    # run must go red.
    no_evidence_majority = forecast_total > 0 and (bot._no_evidence_count * 2 > forecast_total)

    try:
        bot.log_report_summary(reports)
    finally:
        if no_evidence_majority:
            logger.error(
                f"[{args.bot_name}] {bot._no_evidence_count} of {forecast_total} forecasts had no web evidence "
                f"- majority of the run. Treating as a broken research stack."
            )
    if no_evidence_majority:
        raise RuntimeError(
            f"Research stack degraded: {bot._no_evidence_count}/{forecast_total} forecasts had "
            f"research_footprint=none. Check TAVILY_API_KEY, EXA_API_KEY and the AskNews credentials."
        )
