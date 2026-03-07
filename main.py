
import argparse
import asyncio
import logging
import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import dotenv
import httpx
from pydantic import BaseModel, Field

from tavily import TavilyClient

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
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_llm_json(text: str) -> str:
    """Cleans up common LLM JSON issues."""
    if text is None:
        return ""
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f"\"{match.group(1)}\": {nums[0]}" if nums else match.group(0)

    text = re.sub(
        r"\"(value|percentile|probability|prediction_in_decimal|revised_prediction_in_decimal|multiplier|delta)\":\s*\"([^\"]+)\"",
        clean_num,
        text,
    )
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


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


# ---------------------------------------------------------------------------
# Research providers
# ---------------------------------------------------------------------------

class ExaSearcher:
    """Uses EXA_API_KEY only."""

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
                return "[Exa Search Results]\n" + "\n\n".join(results) if results else "[Exa search failed]"
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return "[Exa search failed]"


# ---------------------------------------------------------------------------
# Forecasting principles
# ---------------------------------------------------------------------------

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
    def apply_time_decay(
        prob: float,
        close_time: Optional[datetime],
        question_volatility: str = "normal",
    ) -> float:
        """
        IMPROVEMENT 4: Context-sensitive time decay.

        volatility="slow"  â†’ structural/long-term trend questions; decay halved
        volatility="normal"â†’ default behaviour
        volatility="fast"  â†’ volatile political/event questions; decay amplified

        Old version applied the same aggressive weights regardless of question
        type, which unfairly crushed confident forecasts on slow-moving topics.
        """
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = max(0.0, (close_time - now).total_seconds() / 86400.0)

        vol_scale = {"slow": 0.5, "normal": 1.0, "fast": 1.5}.get(question_volatility, 1.0)

        if days > 365:
            w = min(0.70 * vol_scale, 0.90)
        elif days > 180:
            w = min(0.50 * vol_scale, 0.80)
        elif days > 90:
            w = min(0.30 * vol_scale, 0.60)
        else:
            w = 0.0

        return (1.0 - w) * prob + w * 0.5

    @staticmethod
    def logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def sigmoid(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    @classmethod
    def extremize_logit(cls, p: float, strength: float) -> float:
        strength = float(np.clip(strength, 0.5, 1.8))
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Schemas / Regimes
# ---------------------------------------------------------------------------

class DecompositionOutput(BaseModel):
    subquestions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)
    # IMPROVEMENT 4: volatility classification used for context-sensitive decay
    volatility: str = Field(default="normal", description="slow | normal | fast")


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


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

@dataclass
class BotFeatureFlags:
    enable_extremize: bool = True
    enable_decomposition: bool = True
    enable_numeric_regimes: bool = True
    enable_red_team: bool = True
    enable_consistency_check: bool = True


# ---------------------------------------------------------------------------
# Run result dataclasses â€” carry narrative alongside prediction
# (IMPROVEMENT 1)
# ---------------------------------------------------------------------------

@dataclass
class BinaryRunResult:
    probability: float
    narrative: str  # full LLM chain-of-thought, previously discarded


@dataclass
class MCRunResult:
    predicted_options: Any   # PredictedOptionList
    narrative: str


@dataclass
class NumericRunResult:
    percentiles: List[Percentile]
    narrative: str


# ---------------------------------------------------------------------------
# Reasoning trace builder
# ---------------------------------------------------------------------------

class ReasoningTrace:
    """
    Accumulates every step of Dezzy's decision â€” including the full LLM
    narrative for each run â€” and renders a human-readable block embedded in
    every ReasonedPrediction.
    """

    def __init__(self, question_text: str, bot_name: str = "dezzy"):
        self.bot_name = bot_name
        self.question_text = question_text
        self._steps: List[Tuple[str, str]] = []

    def add(self, label: str, detail: str) -> None:
        self._steps.append((label, str(detail)))
        logger.info(f"[{self.bot_name}] {label}: {str(detail)[:200]}")

    def add_narrative(self, run_num: int, narrative: str) -> None:
        """IMPROVEMENT 1: Preserve full LLM chain-of-thought per run."""
        self._steps.append((f"LLM narrative (run {run_num})", narrative.strip()))
        logger.info(f"[{self.bot_name}] LLM narrative run {run_num}: {narrative[:120].strip()}â€¦")

    def add_research_summary(self, summary: str) -> None:
        """IMPROVEMENT 7: Prepend a research summary so readers see evidence quality first."""
        # Insert at position 0 so it appears right after the header
        self._steps.insert(0, ("Research summary", summary.strip()))

    def render(self) -> str:
        lines = [
            f"â•”â•â• [{self.bot_name.upper()}] REASONING TRACE â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•",
            f"â•‘  Question : {self.question_text[:120]}",
            f"â•‘  Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "â• â•â• STEPS â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•",
        ]
        for i, (label, detail) in enumerate(self._steps, 1):
            lines.append(f"â•‘  {i:02d}. {label}")
            for chunk in [detail[j : j + 110] for j in range(0, len(detail), 110)]:
                lines.append(f"â•‘       {chunk}")
        lines.append("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main bot â€” Dezzy
# ---------------------------------------------------------------------------

class Dezzy(ForecastBot):
    """
    Dezzy â€” a transparent, conservative superforecaster bot.

    Research      : Tavily + Exa (cached per question URL â€” improvement 3)
    LLM           : OpenRouter free router
    Aggregation   : multi-run median â†’ conservative shrink â†’ red-team blend
                    â†’ extremize fixed gate [0.10, 0.90] â†’ context-sensitive
                    time-decay â†’ Bayes calibration
    Reasoning     : Full LLM narratives + mechanical steps in every
                    ReasonedPrediction (improvements 1 & 7)
    """

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

        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None

        # IMPROVEMENT 3: Research caches keyed by question URL
        self._research_cache: Dict[str, str] = {}
        self._research_summary_cache: Dict[str, str] = {}
        self._volatility_cache: Dict[str, str] = {}

        # IMPROVEMENT 5: Only binary predictions stored for consistency checks
        self._recent_binary_predictions: List[Tuple[str, float]] = []

    def _llm_config_defaults(self) -> Dict[str, str]:
        free = "openrouter/openrouter/free"
        return {
            "default": free,
            "parser": free,
            "query_optimizer": free,
            "critic": free,
            "red_team": free,
            "decomposer": free,
            "summarizer": free,
        }

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _search_footprint(self, research: str) -> str:
        used: list[str] = []

        def ok(tag: str, fail_markers: list[str]) -> bool:
            return (tag in research) and not any(m in research for m in fail_markers)

        if ok("[Tavily Data]", ["[Tavily not configured]", "[Tavily search failed]"]):
            used.append("tavily")
        if ok("[Exa Search Results]", ["[Exa not configured]", "[Exa search failed]"]):
            used.append("exa")
        return ",".join(used) if used else "none"

    def _ensure_some_research_or_raise(self, research: str) -> None:
        if self._search_footprint(research) == "none":
            raise RuntimeError("No research evidence available (Tavily and Exa both failed or not configured).")

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.65, 2: 0.82}.get(n, 0.7)

    def _question_url(self, question: MetaculusQuestion) -> str:
        return getattr(question, "page_url", None) or question.question_text[:80]

    async def _decompose_question(self, question: MetaculusQuestion) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition:
            return None
        try:
            llm = self.get_llm("decomposer", "llm")
            prompt = clean_indents(
                f"""
Decompose the forecasting question into subquestions, key entities, key metrics, and a volatility
classification.

volatility must be one of:
  "slow"   â€” structural trend, changes over years (e.g. population, GDP)
  "normal" â€” moderate pace of change
  "fast"   â€” event-driven, political, could flip in days/weeks

Return ONLY JSON:
{{"subquestions":[...], "key_entities":[...], "key_metrics":[...], "volatility": "normal"}}

Question:
{question.question_text}

Resolution criteria:
{question.resolution_criteria}
"""
            )
            raw = await llm.invoke(prompt)
            return safe_model(DecompositionOutput, raw)  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            return None

    async def _optimize_search_query(
        self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]
    ) -> List[str]:
        llm = self.get_llm("query_optimizer", "llm")
        extra = ""
        if decomp and decomp.subquestions:
            extra += "\nSubquestions:\n" + "\n".join(f"- {s}" for s in decomp.subquestions[:6])
        if decomp and decomp.key_entities:
            extra += "\nEntities:\n" + ", ".join(decomp.key_entities[:12])
        if decomp and decomp.key_metrics:
            extra += "\nMetrics:\n" + ", ".join(decomp.key_metrics[:12])

        prompt = f"""
Rewrite this forecasting question into 3 precise web search queries.
Prefer entity names, key metrics, and date ranges.

Question: {question.question_text}
{extra}

Output ONLY JSON list: ["q1","q2","q3"]
""".strip()

        try:
            resp = await llm.invoke(prompt)
            queries = json.loads(sanitize_llm_json(resp))
            cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            return cleaned[:3] if cleaned else [question.question_text[:160]]
        except Exception:
            return [question.question_text[:160]]

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily:
            return "[Tavily not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=6,
                    include_answer=False,
                    include_raw_content=False,
                ),
            )
            context = "\n".join(
                f"Source: {r.get('url','')}\nContent: {r.get('content','')}"
                for r in response.get("results", [])
            )
            return f"[Tavily Data]\n{context}" if context.strip() else "[Tavily search failed]"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        """
        IMPROVEMENT 7: 3-5 sentence research summary for the trace header.
        Cached so it is only generated once per question.
        """
        url = self._question_url(question)
        if url in self._research_summary_cache:
            return self._research_summary_cache[url]
        try:
            llm = self.get_llm("summarizer", "llm")
            prompt = clean_indents(f"""
Summarise the key evidence below in 3-5 sentences relevant to the forecasting question.
Include: what the research found, key uncertainties, and any base rate or market signal found.

Question: {question.question_text}

Research:
{research[:3000]}
""")
            summary = (await llm.invoke(prompt)).strip()
        except Exception:
            summary = "[Research summary unavailable]"
        self._research_summary_cache[url] = summary
        return summary

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        IMPROVEMENT 3: Cache by question URL.
        Multi-run no longer re-fetches identical research N times.
        """
        url = self._question_url(question)
        if url in self._research_cache:
            logger.info(f"[dezzy] Research cache HIT for {url[:80]}")
            return self._research_cache[url]

        decomp = await self._decompose_question(question)
        queries = await self._optimize_search_query(question, decomp)
        optimized_query = " OR ".join(queries)

        results = await asyncio.gather(
            self._run_tavily_search(optimized_query),
            self._run_exa_search(optimized_query),
            return_exceptions=True,
        )

        cleaned: list[str] = []
        for res in results:
            if isinstance(res, Exception):
                cleaned.append(f"[Search failed: {str(res)}]")
            else:
                cleaned.append(res)
        combined = "\n\n".join(cleaned).strip()

        research = (
            f"{ForecastingPrinciples.get_generic_base_rate()}\n\n"
            f"{ForecastingPrinciples.get_generic_fermi_prompt()}\n\n"
            f"{combined}"
        )

        self._ensure_some_research_or_raise(research)
        self._research_cache[url] = research

        # IMPROVEMENT 4: Store volatility from decomposition for time-decay use
        if decomp and decomp.volatility in ("slow", "normal", "fast"):
            self._volatility_cache[url] = decomp.volatility

        return research

    def _get_volatility(self, question: MetaculusQuestion) -> str:
        return self._volatility_cache.get(self._question_url(question), "normal")

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _median(xs: List[float]) -> float:
        xs = [float(x) for x in xs if np.isfinite(float(x))]
        if not xs:
            return 0.5
        xs.sort()
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])

    @staticmethod
    def _shrink_to_half(p: float, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.clip((1 - alpha) * p + alpha * 0.5, 0.0, 1.0))

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        if not getattr(question, "close_time", None):
            return 0.15
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        return 0.20 if days_to_close > 180 else 0.10

    def _agreement_strength(self, probs: List[float]) -> float:
        if not probs:
            return 0.0
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return float(np.clip(1.0 - (spread / 0.30), 0.0, 1.0))

    def _extremize_strength(
        self, research: str, probs: List[float], question: MetaculusQuestion
    ) -> float:
        if not self.flags.enable_extremize:
            return 1.0
        quality = self._research_quality_weight(research)
        agree = self._agreement_strength(probs)
        base = 1.0 + 0.45 * (quality - 0.5) * 2.0 * agree
        close_time = getattr(question, "close_time", None)
        if close_time:
            days = (close_time - datetime.now(timezone.utc)).days
            if days < 60:
                base = 1.0 + (base - 1.0) * 0.6
        return float(np.clip(base, 0.95, 1.6))

    @staticmethod
    def _extremize_gate(p: float) -> bool:
        """
        IMPROVEMENT 2: Fixed extremize gate.

        Old gate [0.60, 0.98] skipped anything below 0.60 â€” exactly where
        extremization is most useful (pushing weak signal away from 0.5).

        New gate [0.10, 0.90]: extremize any forecast that isn't already
        near the extremes. Values outside this range are already confident
        enough and don't need further pushing.
        """
        return 0.10 <= float(p) <= 0.90

    async def _red_team_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        initial_pred: float,
        trace: ReasoningTrace,
    ) -> float:
        """
        IMPROVEMENT 6: Sharpened red-team prompt.

        Old prompt dumped the full research and asked vaguely for problems.
        New prompt asks for the SINGLE strongest counter-argument, producing
        more focused, actionable corrections.
        """
        if not self.flags.enable_red_team:
            trace.add("Red-team", "SKIPPED (flag disabled)")
            return initial_pred
        self._ensure_some_research_or_raise(research)
        llm = self.get_llm("red_team", "llm")
        try:
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a skeptical superforecaster red-teamer.

Your task: identify the SINGLE strongest argument against the current forecast,
then give a revised probability that accounts for it.

Focus on ONE of:
  (a) Base-rate neglect â€” is the forecaster ignoring how rarely this happens historically?
  (b) Resolution pitfall â€” could the question resolve differently than the forecaster assumed?
  (c) Missing disconfirming evidence â€” what key fact from the research was underweighted?

Question: {question.question_text}

Research (condensed):
{research[:2000]}

Current forecast: {initial_pred:.2%}

First state in ONE sentence: the strongest counter-argument.
Then output ONLY JSON:
{{"counter_argument": "...", "revised_prediction_in_decimal": 0.XX}}
"""
                )
            )
            counter_match = re.search(r'"counter_argument"\s*:\s*"([^"]+)"', raw or "")
            counter_txt = counter_match.group(1) if counter_match else "[not extracted]"

            parsed = await structure_output(
                sanitize_llm_json(raw),
                dict,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            val = float(parsed.get("revised_prediction_in_decimal"))
            result = float(np.clip(val, 0.0, 1.0))
            trace.add("Red-team counter-argument", counter_txt)
            trace.add(
                "Red-team result",
                f"revised={result:.4f} (from initial={initial_pred:.4f}, Î”={result - initial_pred:+.4f})",
            )
            return result
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            trace.add("Red-team", f"FAILED ({e}); keeping initial={initial_pred:.4f}")
            return initial_pred

    async def _check_consistency(
        self,
        question: MetaculusQuestion,
        proposed_pred: float,
        trace: ReasoningTrace,
    ) -> bool:
        """
        IMPROVEMENT 5: Consistency check uses only prior binary predictions.

        Old version mixed binary probabilities with normalised numeric medians
        (e.g. med / (|med| + 1)), which are not comparable and caused the LLM
        to evaluate nonsensical comparisons.
        """
        if not self.flags.enable_consistency_check:
            trace.add("Consistency check", "SKIPPED (flag disabled)")
            return True
        if len(self._recent_binary_predictions) < 2:
            trace.add("Consistency check", "SKIPPED (fewer than 2 prior binary predictions)")
            return True

        recent_summary = "\n".join(
            f"Q: {qt} â†’ Pred: {p:.2%}" for qt, p in self._recent_binary_predictions[-3:]
        )
        llm = self.get_llm("parser", "llm")
        prompt = f"""
Is this new binary forecast logically consistent with prior binary forecasts on related topics?

New: {question.question_text} â†’ {proposed_pred:.2%}

Prior binary forecasts:
{recent_summary}

Answer YES or NO only.
""".strip()
        try:
            response = await llm.invoke(prompt)
            result = "YES" in (response or "").upper()
            trace.add("Consistency check", "PASSED" if result else "FAILED â€” applying consistency shrink")
            return result
        except Exception:
            trace.add("Consistency check", "ERROR â€” treating as consistent")
            return True

    # ------------------------------------------------------------------
    # Numeric parsing
    # ------------------------------------------------------------------

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(
            f"""
Extract a numeric forecast distribution from the text.

Output MUST be a list of objects with fields:
  - percentile  (10/20/40/60/80/90 OR 0.1/0.2/0.4/0.6/0.8/0.9)
  - value       (in units: {question.unit_of_measure}, no scientific notation)

Required percentiles: 10, 20, 40, 60, 80, 90 (exactly six).
Values must be strictly increasing with percentile.
"""
        )

    @staticmethod
    def _extract_percentile_block(text: str) -> str:
        m = re.search(
            r"(Percentile\s*10\s*:.*?Percentile\s*90\s*:.*?)(?:\n\s*\n|$)",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        lines = [
            line.strip()
            for line in (text or "").splitlines()
            if re.search(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, flags=re.IGNORECASE)
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_raw_percentiles(raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p /= 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        return out

    @staticmethod
    def _require_standard_percentiles(pcts: List[Percentile]) -> List[Percentile]:
        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        by = {round(float(p.percentile), 3): p for p in pcts}
        if any(round(r, 3) not in by for r in required):
            return []
        return [by[round(r, 3)] for r in required]

    @staticmethod
    def _enforce_monotone(pcts: List[Percentile]) -> List[Percentile]:
        pcts = sorted(pcts, key=lambda x: float(x.percentile))
        for i in range(1, len(pcts)):
            if pcts[i].value <= pcts[i - 1].value:
                pcts[i].value = pcts[i - 1].value + 1e-6
        return pcts

    @staticmethod
    def _bounds_fallback(question: NumericQuestion) -> List[Percentile]:
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        w = {0.1: 0.05, 0.2: 0.15, 0.4: 0.40, 0.6: 0.60, 0.8: 0.85, 0.9: 0.95}
        pcts = [Percentile(percentile=p, value=lo + (hi - lo) * w[p]) for p in w]
        return Dezzy._enforce_monotone(pcts)

    @staticmethod
    def _median_from_40_60(pcts: List[Percentile]) -> float:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        if 0.4 in by and 0.6 in by:
            return 0.5 * (by[0.4] + by[0.6])
        return float(sorted(pcts, key=lambda x: x.percentile)[len(pcts) // 2].value) if pcts else 0.0

    @staticmethod
    def _p10_p90(pcts: List[Percentile]) -> Tuple[Optional[float], Optional[float]]:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        return by.get(0.1), by.get(0.9)

    async def _parse_numeric_percentiles_robust(
        self, question: NumericQuestion, text: str, stage: str
    ) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
        instructions = self._numeric_parsing_instructions(question)

        for attempt, label in enumerate(["direct", "block-extract", "reformat"], 1):
            try:
                if label == "reformat":
                    reform_prompt = clean_indents(
                        f"""
Rewrite into EXACTLY these 6 lines (no extra text):
Percentile 10: <number>
Percentile 20: <number>
Percentile 40: <number>
Percentile 60: <number>
Percentile 80: <number>
Percentile 90: <number>
Rules: units={question.unit_of_measure}, no scientific notation, strictly increasing.
Text:
{text}
"""
                    )
                    text_to_parse = (
                        self._extract_percentile_block(await parser_llm.invoke(reform_prompt)) or text
                    )
                elif label == "block-extract":
                    text_to_parse = self._extract_percentile_block(text)
                    if not text_to_parse:
                        continue
                else:
                    text_to_parse = text

                raw: List[RawPercentile] = await structure_output(
                    text_to_parse,
                    list[RawPercentile],
                    model=parser_llm,
                    additional_instructions=instructions,
                    num_validation_samples=1,
                )
                normalized = self._normalize_raw_percentiles(raw)
                standard = self._require_standard_percentiles(normalized)
                if standard:
                    return self._enforce_monotone(standard)
            except Exception as e:
                logger.warning(f"[{stage}] parse attempt {attempt} ({label}) failed: {e}")

        logger.warning(f"[{stage}] all parse attempts failed; using bounds fallback.")
        return self._bounds_fallback(question)

    # ------------------------------------------------------------------
    # Numeric regimes
    # ------------------------------------------------------------------

    def _extract_date_range_generic(self, text: str) -> Optional[Tuple[date, date]]:
        m = re.search(
            r"\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*-\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*",
            text or "",
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                start = datetime.strptime(m.group(1), fmt).date()
                end = datetime.strptime(m.group(2), fmt).date()
                if start > end:
                    start, end = end, start
                return start, end
            except Exception:
                continue
        return None

    def _has_partial_observations(self, research: str, question: NumericQuestion) -> bool:
        r = (research or "").lower()
        cues = ["sum to", "subtotal", "observed", "published", "known", "so far", "to date", "remaining"]
        return (
            any(c in r for c in cues)
            and self._extract_date_range_generic(question.question_text or "") is not None
        )

    def _detect_numeric_regime(self, question: NumericQuestion, research: str) -> NumericRegime:
        if not self.flags.enable_numeric_regimes:
            return NumericRegime.GENERIC
        if self._has_partial_observations(research, question):
            return NumericRegime.PARTIAL_REVEAL_SUM
        dr = self._extract_date_range_generic(question.question_text or "")
        if dr:
            start, end = dr
            if 2 <= (end - start).days + 1 <= 31:
                return NumericRegime.STRUCTURED_TS
        return NumericRegime.GENERIC

    async def _llm_extract_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> PartialRevealExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"known_subtotal": null, "known_parts": null, "total_parts": null, "notes": null}}

Question: {question.question_text}
Research: {research}

Extract known_subtotal if research states a subtotal/sum; known_parts and total_parts if inferable.
""")
        return safe_model(PartialRevealExtract, await parser.invoke(prompt))  # type: ignore[return-value]

    async def _llm_extract_reference_class(
        self, question: NumericQuestion, research: str
    ) -> ReferenceClassExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"reference_totals": [], "trend_multiplier": null, "notes": null}}

Question: {question.question_text}
Research: {research}

Extract comparable reference totals and an optional trend multiplier (0.85-1.15).
""")
        return safe_model(ReferenceClassExtract, await parser.invoke(prompt))  # type: ignore[return-value]

    async def _bounded_multiplier(
        self, question: NumericQuestion, research: str, baseline: float, *, lo: float, hi: float
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"multiplier": 1.00}}
Question: {question.question_text}
Baseline: {baseline}
Research: {research}
Rules: multiplier within [{lo:.6f}, {hi:.6f}]. Output only JSON.
""")
        model = safe_model(BoundedMultiplier, await critic.invoke(prompt))  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "multiplier")), lo, hi))

    def _mult_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days or 30
        if h <= 21:
            return (0.98, 1.02)
        if h <= 60:
            return (0.96, 1.04)
        return (0.92, 1.08)

    def _horizon_days_from_text(self, question: NumericQuestion) -> Optional[int]:
        dr = self._extract_date_range_generic(question.question_text or "")
        return (dr[1] - dr[0]).days + 1 if dr else None

    @staticmethod
    def _normal_percentiles_from_mean_sd(mean: float, sd: float) -> List[Percentile]:
        z = {0.1: -1.2816, 0.2: -0.8416, 0.4: -0.2533, 0.6: 0.2533, 0.8: 0.8416, 0.9: 1.2816}
        pcts = [Percentile(percentile=p, value=float(mean + z[p] * sd)) for p in z]
        return Dezzy._enforce_monotone(pcts)

    # ------------------------------------------------------------------
    # Single model calls â€” narrative-preserving (IMPROVEMENT 1)
    # ------------------------------------------------------------------

    async def _single_model_forecast_binary(
        self, question: BinaryQuestion, research: str
    ) -> BinaryRunResult:
        """Returns probability AND the full LLM chain-of-thought narrative."""
        temp = self._get_temperature(question)
        llm = GeneralLlm(model=self._llm_config_defaults()["default"], temperature=temp)

        raw = await llm.invoke(
            clean_indents(
                f"""
You are a professional superforecaster named Dezzy.

Question: {question.question_text}

Background: {getattr(question, 'background_info', '')}

Resolution criteria: {getattr(question, 'resolution_criteria', '')}

Research:
{research}

Think step by step and write your reasoning clearly:
1. Reference class â€” what fraction of similar past questions resolved YES?
2. Evidence supporting YES from the research above
3. Evidence supporting NO from the research above
4. Status quo â€” what happens if nothing changes before the resolution date?
5. Key uncertainties that could shift the outcome
6. Calibrated synthesis â€” blend outside view (base rate) with inside view evidence

Write your full reasoning, then end with ONLY this JSON on the last line:
{{"prediction_in_decimal": 0.50}}
"""
            )
        )

        prediction = await structure_output(
            sanitize_llm_json(raw),
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=1,
        )
        return BinaryRunResult(
            probability=float(np.clip(prediction.prediction_in_decimal, 0.01, 0.99)),
            narrative=raw.strip(),
        )

    async def _single_model_forecast_mc(
        self, question: MultipleChoiceQuestion, research: str
    ) -> MCRunResult:
        temp = self._get_temperature(question)
        llm = GeneralLlm(model=self._llm_config_defaults()["default"], temperature=temp)
        schema_example = json.dumps({
            "predicted_options": [
                {"option_name": opt, "probability": round(1 / max(1, len(question.options)), 3)}
                for opt in question.options
            ]
        })

        raw = await llm.invoke(
            clean_indents(
                f"""
You are a professional superforecaster named Dezzy.

Question: {question.question_text}
Options: {question.options}

Background: {getattr(question, 'background_info', '')}

Resolution criteria: {getattr(question, 'resolution_criteria', '')}

Research:
{research}

Think step by step:
1. Reference class â€” historically, how often does each option type win in similar questions?
2. Evidence from research favouring each option
3. Status quo â€” which option does the current trajectory favour?
4. Key uncertainties
5. Calibrated synthesis â€” assign probabilities summing to exactly 1.0

Write your full reasoning, then end with ONLY this JSON on the last line:
{schema_example}
"""
            )
        )

        prediction = await structure_output(
            sanitize_llm_json(raw),
            PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=1,
        )
        return MCRunResult(predicted_options=prediction, narrative=raw.strip())

    async def _single_model_forecast_numeric(
        self, question: NumericQuestion, research: str
    ) -> NumericRunResult:
        temp = self._get_temperature(question)
        llm = GeneralLlm(model=self._llm_config_defaults()["default"], temperature=temp)
        units = question.unit_of_measure or "Not stated"
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound

        raw = await llm.invoke(
            clean_indents(
                f"""
You are a professional superforecaster named Dezzy.

Question: {question.question_text}
Units: {units}
Bounds: [{lower}, {upper}]

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Think step by step:
1. Reference class â€” historical base rate or typical range for this quantity
2. Trend suggested by the research
3. Key upside risks (value higher than expected)
4. Key downside risks (value lower than expected)
5. How wide the uncertainty interval should be given the evidence quality

Write your full reasoning, then end with EXACTLY these 6 lines (no other text after):
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"""
            )
        )

        percentiles = await self._parse_numeric_percentiles_robust(question, raw, stage="model_numeric")
        return NumericRunResult(percentiles=percentiles, narrative=raw.strip())

    # ------------------------------------------------------------------
    # Multi-run
    # ------------------------------------------------------------------

    async def _multi_run_binary(
        self, question: BinaryQuestion, research: str
    ) -> List[BinaryRunResult]:
        results: List[BinaryRunResult] = []
        for i in range(self.runs_per_question):
            try:
                results.append(await self._single_model_forecast_binary(question, research))
            except Exception as e:
                logger.warning(f"Binary run {i+1}/{self.runs_per_question} failed: {e}")
        return results

    async def _multi_run_mc(
        self, question: MultipleChoiceQuestion, research: str
    ) -> List[MCRunResult]:
        results: List[MCRunResult] = []
        for i in range(self.runs_per_question):
            try:
                results.append(await self._single_model_forecast_mc(question, research))
            except Exception as e:
                logger.warning(f"MC run {i+1}/{self.runs_per_question} failed: {e}")
        return results

    async def _multi_run_numeric(
        self, question: NumericQuestion, research: str
    ) -> List[NumericRunResult]:
        results: List[NumericRunResult] = []
        for i in range(self.runs_per_question):
            try:
                results.append(await self._single_model_forecast_numeric(question, research))
            except Exception as e:
                logger.warning(f"Numeric run {i+1}/{self.runs_per_question} failed: {e}")
        return results

    # ------------------------------------------------------------------
    # Forecasting: Binary
    # ------------------------------------------------------------------

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)
        quality = self._research_quality_weight(research)
        volatility = self._get_volatility(question)
        trace.add("Research sources", f"{self._search_footprint(research)} | quality={quality:.2f} | volatility={volatility}")

        # IMPROVEMENT 7: Research summary in trace header
        summary = await self._summarize_research(question, research)
        trace.add_research_summary(summary)

        # --- Multi-run ---
        runs = await self._multi_run_binary(question, research)
        if not runs:
            raise RuntimeError("All binary runs failed.")

        # IMPROVEMENT 1: Embed full LLM narratives
        for i, run in enumerate(runs, 1):
            trace.add_narrative(i, run.narrative)

        probs = [r.probability for r in runs]
        run_med = self._median(probs)
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        trace.add(
            f"Multi-run aggregation ({len(probs)} runs)",
            f"individual={[f'{p:.4f}' for p in probs]} | median={run_med:.4f} | spread={spread:.4f}",
        )

        applied: List[str] = []

        # --- Conservative shrink ---
        shrink = 0.12
        if spread >= 0.20:
            shrink = 0.28
            applied.append("shrink(high-spread)")
        if quality < 0.70:
            shrink = max(shrink, 0.22)
            applied.append("shrink(low-research)")
        base_p = self._shrink_to_half(run_med, shrink)
        trace.add(
            "Conservative shrink",
            f"alpha={shrink:.2f} | {run_med:.4f} â†’ {base_p:.4f} | triggers=[{', '.join(applied) or 'none'}]",
        )

        # --- Red-team (sharpened prompt) ---
        red_p = await self._red_team_forecast(question, research, base_p, trace)
        combined = 0.6 * base_p + 0.4 * red_p
        applied.append("blend(red-team)")
        trace.add("Red-team blend", f"0.6Ã—{base_p:.4f} + 0.4Ã—{red_p:.4f} = {combined:.4f}")

        # --- Consistency check (binary-only) ---
        if not await self._check_consistency(question, combined, trace):
            before = combined
            combined = 0.5 * combined + 0.5 * 0.5
            applied.append("consistency-shrink")
            trace.add("Consistency shrink", f"{before:.4f} â†’ {combined:.4f}")

        # --- Extremize (fixed gate [0.10, 0.90]) ---
        gate_hit = self.flags.enable_extremize and self._extremize_gate(combined)
        if gate_hit:
            ext_strength = self._extremize_strength(research, probs + [combined], question)
            p_ext = ForecastingPrinciples.extremize_logit(combined, ext_strength)
            applied.append(f"extremize(x{ext_strength:.2f})")
            trace.add(
                "Extremize",
                f"gate=OPEN pâˆˆ[0.10,0.90] | strength={ext_strength:.3f} | {combined:.4f} â†’ {p_ext:.4f}",
            )
        else:
            p_ext = combined
            applied.append("extremize(gated-off)")
            reason = "flag disabled" if not self.flags.enable_extremize else f"p={combined:.4f} outside [0.10,0.90]"
            trace.add("Extremize", f"gate=CLOSED ({reason}) | p unchanged={p_ext:.4f}")

        # --- Context-sensitive time decay ---
        close_time = getattr(question, "close_time", None)
        p_time = ForecastingPrinciples.apply_time_decay(p_ext, close_time, question_volatility=volatility)
        if abs(p_time - p_ext) > 1e-6:
            applied.append(f"time-decay({volatility})")
            days_left = int((close_time - datetime.now(timezone.utc)).days) if close_time else "?"
            trace.add("Time decay", f"volatility={volatility} | days_left={days_left} | {p_ext:.4f} â†’ {p_time:.4f}")
        else:
            trace.add("Time decay", f"no change (volatility={volatility}, <90 days or no close_time)")

        # --- Bayesian calibration ---
        if hasattr(self, "apply_bayesian_calibration"):
            try:
                p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
                if abs(p_cal - p_time) > 1e-6:
                    applied.append("bayes-calibration")
                    trace.add("Bayes calibration", f"{p_time:.4f} â†’ {p_cal:.4f}")
                else:
                    trace.add("Bayes calibration", "no change")
            except Exception:
                p_cal = p_time
                trace.add("Bayes calibration", "FAILED â€” keeping previous value")
        else:
            p_cal = p_time
            trace.add("Bayes calibration", "not available on this ForecastBot base")

        final_p = float(np.clip(p_cal, 0.01, 0.99))
        trace.add("Final clamp [0.01, 0.99]", f"{p_cal:.4f} â†’ {final_p:.4f}")
        trace.add("Pipeline summary", f"controls applied: {', '.join(applied)}")
        trace.add("â˜… FINAL PREDICTION", f"{final_p:.4f}  ({final_p:.1%})")

        # IMPROVEMENT 5: Only store binary predictions for consistency checks
        self._recent_binary_predictions.append((question.question_text[:120], final_p))
        if len(self._recent_binary_predictions) > 10:
            self._recent_binary_predictions.pop(0)

        return ReasonedPrediction(prediction_value=final_p, reasoning=trace.render())

    # ------------------------------------------------------------------
    # Forecasting: Multiple choice
    # ------------------------------------------------------------------

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)
        quality = self._research_quality_weight(research)
        trace.add("Research sources", f"{self._search_footprint(research)} | quality={quality:.2f}")
        trace.add("Options", str(list(question.options)))

        summary = await self._summarize_research(question, research)
        trace.add_research_summary(summary)

        runs = await self._multi_run_mc(question, research)
        if not runs:
            raise RuntimeError("All MC runs failed.")

        for i, run in enumerate(runs, 1):
            trace.add_narrative(i, run.narrative)

        opt_names = list(question.options)
        per_opt: Dict[str, List[float]] = {o: [] for o in opt_names}
        for r in runs:
            try:
                cur = {o.option_name: float(o.probability) for o in r.predicted_options.predicted_options}
            except Exception:
                continue
            for o in opt_names:
                per_opt[o].append(float(cur.get(o, 0.0)))

        med_probs = {o: self._median(per_opt[o]) if per_opt[o] else 0.0 for o in opt_names}
        trace.add(
            f"Multi-run medians ({len(runs)} runs)",
            " | ".join(f"{o}={v:.4f}" for o, v in med_probs.items()),
        )

        uniform = 1.0 / max(1, len(opt_names))
        alpha = 0.10 if quality >= 0.75 else 0.18
        shrunk = {o: (1 - alpha) * med_probs[o] + alpha * uniform for o in opt_names}
        trace.add("Shrink to uniform", f"alpha={alpha:.2f} | uniform={uniform:.4f}")

        total = float(sum(max(0.0, v) for v in shrunk.values()))
        final = (
            [{"option_name": o, "probability": uniform} for o in opt_names]
            if total <= 0
            else [{"option_name": o, "probability": float(np.clip(shrunk[o] / total, 0.0, 1.0))} for o in opt_names]
        )

        trace.add("Normalized probs", " | ".join(f"{x['option_name']}={x['probability']:.4f}" for x in final))
        trace.add("â˜… FINAL PREDICTION", " | ".join(f"{x['option_name']}={x['probability']:.1%}" for x in final))

        final_val = safe_model(PredictedOptionList, {"predicted_options": final})  # type: ignore[assignment]
        return ReasonedPrediction(prediction_value=final_val, reasoning=trace.render())

    # ------------------------------------------------------------------
    # Forecasting: Numeric (generic)
    # ------------------------------------------------------------------

    async def _run_forecast_on_numeric_generic(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Numeric regime", "GENERIC")
        trace.add(
            "Research sources",
            f"{self._search_footprint(research)} | quality={self._research_quality_weight(research):.2f}",
        )

        summary = await self._summarize_research(question, research)
        trace.add_research_summary(summary)

        runs = await self._multi_run_numeric(question, research)
        if not runs:
            raise RuntimeError("All numeric runs failed.")

        for i, run in enumerate(runs, 1):
            trace.add_narrative(i, run.narrative)

        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_pct: Dict[float, List[float]] = {p: [] for p in required}

        for r in runs:
            for pct in r.percentiles:
                p = float(pct.percentile)
                if p > 1.0:
                    p /= 100.0
                p = round(p, 3)
                v = float(pct.value)
                if p in per_pct and np.isfinite(v):
                    per_pct[p].append(v)

        agg: List[Percentile] = []
        for p in required:
            vals = per_pct.get(round(p, 3), [])
            if vals:
                agg.append(Percentile(percentile=p, value=float(self._median(vals))))
            else:
                trace.add("Fallback triggered", f"no values for p={p} â€” bounds-based fallback")
                pcts = self._bounds_fallback(question)
                dist = NumericDistribution.from_question(pcts, question)
                trace.add("â˜… FINAL (bounds fallback)", self._format_pcts(pcts))
                return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

        agg = self._enforce_monotone(agg)
        med = self._median_from_40_60(agg)
        p10, p90 = self._p10_p90(agg)
        trace.add(f"Aggregated percentiles ({len(runs)} runs)", self._format_pcts(agg))
        trace.add("Monotone enforced", "yes")
        trace.add("Distribution summary", f"medianâ‰ˆ{med:.6g} | P10={p10:.6g} | P90={p90:.6g}")
        trace.add("â˜… FINAL PREDICTION", self._format_pcts(agg))

        dist = NumericDistribution.from_question(agg, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

    async def _forecast_numeric_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Numeric regime", "PARTIAL_REVEAL_SUM")

        summary = await self._summarize_research(question, research)
        trace.add_research_summary(summary)

        try:
            ex = await self._llm_extract_partial_reveal(question, research)
        except Exception:
            trace.add("Partial-reveal extract", "FAILED â€” falling back to generic")
            return await self._run_forecast_on_numeric_generic(question, research)

        if ex.known_subtotal is None or not np.isfinite(float(ex.known_subtotal)) or float(ex.known_subtotal) <= 0:
            trace.add("Partial-reveal extract", f"known_subtotal={ex.known_subtotal} invalid â€” falling back")
            return await self._run_forecast_on_numeric_generic(question, research)

        known = float(ex.known_subtotal)
        trace.add("Known subtotal", f"{known:.6g} | notes={ex.notes}")

        remainder_baseline = 0.75 * known
        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(question, research, remainder_baseline, lo=lo_m, hi=hi_m)
        total_mean = known + remainder_baseline * mult
        sd = max(0.10 * total_mean, 0.05 * known)
        trace.add("Remainder estimate", f"baseline={remainder_baseline:.6g} Ã— mult={mult:.4f} | total_mean={total_mean:.6g} | sd={sd:.6g}")

        pcts = self._normal_percentiles_from_mean_sd(total_mean, sd)
        for p in pcts:
            if p.value < known:
                p.value = known
        pcts = self._enforce_monotone(pcts)
        trace.add("â˜… FINAL PREDICTION", self._format_pcts(pcts))

        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(pcts, question),
            reasoning=trace.render(),
        )

    async def _forecast_numeric_structured_ts(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Numeric regime", "STRUCTURED_TS")

        summary = await self._summarize_research(question, research)
        trace.add_research_summary(summary)

        baseline = 0.5 * (float(question.lower_bound) + float(question.upper_bound))
        try:
            ref = await self._llm_extract_reference_class(question, research)
            refs = [float(x) for x in (ref.reference_totals or []) if np.isfinite(float(x)) and float(x) > 0]
            if refs:
                baseline = float(np.median(refs))
                trace.add("Reference class", f"totals={refs} | median_baseline={baseline:.6g}")
                if ref.trend_multiplier is not None and np.isfinite(float(ref.trend_multiplier)):
                    tm = float(ref.trend_multiplier)
                    if 0.85 <= tm <= 1.15:
                        baseline *= tm
                        trace.add("Trend multiplier", f"Ã—{tm:.4f} â†’ adjusted_baseline={baseline:.6g}")
            else:
                trace.add("Reference class", f"no usable totals â€” midpoint baseline={baseline:.6g}")
        except Exception as e:
            trace.add("Reference class extract", f"FAILED ({e}) â€” midpoint baseline={baseline:.6g}")

        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(question, research, baseline, lo=lo_m, hi=hi_m)
        mean = baseline * mult
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        width = hi - lo if np.isfinite(hi - lo) and hi > lo else max(1.0, abs(mean))
        sd = float(np.clip(0.10 * abs(mean) + 0.05 * width, 1e-9, 0.35 * abs(mean) + 1e-9))
        trace.add("Final mean & sd", f"baseline={baseline:.6g} Ã— mult={mult:.4f} â†’ mean={mean:.6g} | sd={sd:.6g}")

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        for p in pcts:
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                p.value = float(np.clip(p.value, lo, hi))
        pcts = self._enforce_monotone(pcts)
        trace.add("â˜… FINAL PREDICTION", self._format_pcts(pcts))

        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(pcts, question),
            reasoning=trace.render(),
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        if not self.flags.enable_numeric_regimes:
            return await self._run_forecast_on_numeric_generic(question, research)

        regime = self._detect_numeric_regime(question, research)
        try:
            if regime == NumericRegime.PARTIAL_REVEAL_SUM:
                return await self._forecast_numeric_partial_reveal(question, research)
            if regime == NumericRegime.STRUCTURED_TS:
                return await self._forecast_numeric_structured_ts(question, research)
        except Exception as e:
            logger.warning(f"Regime {regime} failed; fallback to generic: {e}")

        return await self._run_forecast_on_numeric_generic(question, research)

    async def _run_forecast_on_numeric_wrapper(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_forecast_on_numeric(question, research)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_pcts(pcts: List[Percentile]) -> str:
        return " | ".join(
            f"P{int(round(float(p.percentile) * 100))}={p.value:.6g}" for p in pcts
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="dezzy: Tavily+Exa, OpenRouter free, multi-run, full reasoning trace"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument("--bot-name", type=str, default="dezzy")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-extremize", action="store_true")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-numeric-regimes", action="store_true")
    parser.add_argument("--no-red-team", action="store_true")
    parser.add_argument("--no-consistency", action="store_true")

    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    flags = BotFeatureFlags(
        enable_extremize=not args.no_extremize,
        enable_decomposition=not args.no_decomposition,
        enable_numeric_regimes=not args.no_numeric_regimes,
        enable_red_team=not args.no_red_team,
        enable_consistency_check=not args.no_consistency,
    )

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("EXA_API_KEY"):
        raise RuntimeError("Set at least one of TAVILY_API_KEY or EXA_API_KEY.")

    bot = Dezzy(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        bot_name=args.bot_name,
        flags=flags,
        runs_per_question=max(1, int(args.runs)),
    )

    client = MetaculusClient()

    async def run_all():
        if run_mode == "tournament":
            seasonal, minibench = await asyncio.gather(
                bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True),
                bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True),
            )
            return seasonal + minibench

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)

        bot.skip_previously_forecasted_questions = False
        EXAMPLE_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [client.get_question_by_url(u.strip()) for u in EXAMPLE_URLS]
        single, market_pulse = await asyncio.gather(
            bot.forecast_questions(questions, return_exceptions=True),
            bot.forecast_on_tournament("market-pulse-26q1", return_exceptions=True),
        )
        return single + market_pulse

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
