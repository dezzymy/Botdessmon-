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


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_llm_json(text: str) -> str:
    """Cleans up common LLM JSON issues (numeric underscores, quoted numbers, fences)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Research providers
# ═══════════════════════════════════════════════════════════════════════════════

class ExaSearcher:
    """Neural search via EXA_API_KEY."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# Forecasting principles
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
        """
        IMPROVEMENT 4: Softened time-decay weights.
        Long-horizon structural questions are no longer aggressively pushed to 0.5.
        Old weights: >365d → (0.3p + 0.7×0.5), >180d → (0.5p + 0.5×0.5), >90d → (0.7p + 0.3×0.5)
        New weights: >365d → (0.85p + 0.15×0.5), >180d → (0.90p + 0.10×0.5), >90d → (0.95p + 0.05×0.5)
        """
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
        strength = float(np.clip(strength, 0.5, 1.8))
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Schemas / Regimes
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Feature flags
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BotFeatureFlags:
    enable_extremize: bool = True
    enable_decomposition: bool = True
    enable_numeric_regimes: bool = True
    enable_red_team: bool = True
    enable_consistency_check: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning trace builder  (IMPROVEMENT 1 + 7)
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningTrace:
    """
    Accumulates every step of Dezzy's decision process — including the LLM's
    own narrative reasoning and a research summary — and renders it as a
    human-readable block embedded in every ReasonedPrediction.
    """

    def __init__(self, question_text: str, bot_name: str = "dezzy"):
        self.bot_name = bot_name
        self.question_text = question_text
        self._steps: List[Tuple[str, str]] = []

    def add(self, label: str, detail: str) -> None:
        self._steps.append((label, str(detail)))
        logger.info(f"[{self.bot_name}] {label}: {detail[:200]}")

    def add_narrative(self, run_index: int, text: str) -> None:
        """Store the LLM's raw chain-of-thought reasoning for a given run."""
        # Trim to first 1500 chars to keep trace readable but informative
        trimmed = (text or "").strip()[:1500]
        if len((text or "").strip()) > 1500:
            trimmed += "\n… [truncated]"
        self._steps.append((f"LLM narrative (run {run_index})", trimmed))
        logger.debug(f"[{self.bot_name}] run {run_index} narrative captured ({len(text)} chars)")

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
                # wrap at 110 chars
                for chunk in [line[j : j + 110] for j in range(0, max(len(line), 1), 110)]:
                    lines.append(f"║       {chunk}")
        lines.append("║")
        lines.append("╚═══════════════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main bot class — Dezzy
# ═══════════════════════════════════════════════════════════════════════════════

class Dezzy(ForecastBot):
    """
    Dezzy — transparent, conservative superforecaster bot.

    All 7 improvements from code-review are implemented:
      1. LLM narrative captured in trace
      2. Extremize gate fixed (now covers values near 0.5)
      3. Research cached per question URL
      4. Time decay softened for long-horizon questions
      5. Consistency check scoped to binary-only predictions
      6. Red-team prompt sharpened with targeted counter-argument request
      7. Research summary embedded as first trace step
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

        self.tavily = (
            TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            if os.getenv("TAVILY_API_KEY")
            else None
        )
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None

        # IMPROVEMENT 3: research cache keyed by question URL
        self._research_cache: Dict[str, str] = {}

        # IMPROVEMENT 5: only binary predictions stored for consistency checks
        self._recent_binary_predictions: List[Tuple[str, float]] = []  # (question_text, prob)

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

    # ──────────────────────────────────────────────────────────────────────────
    # Research
    # ──────────────────────────────────────────────────────────────────────────

    def _search_footprint(self, research: str) -> str:
        used: List[str] = []

        def ok(tag: str, fail_markers: List[str]) -> bool:
            return (tag in research) and (not any(m in research for m in fail_markers))

        if ok("[Tavily Data]", ["[Tavily not configured]", "[Tavily search failed]"]):
            used.append("tavily")
        if ok("[Exa Search Results]", ["[Exa not configured]", "[Exa search failed]"]):
            used.append("exa")
        return ",".join(used) if used else "none"

    def _ensure_some_research_or_raise(self, research: str) -> None:
        if self._search_footprint(research) == "none":
            raise RuntimeError(
                "No research evidence available (Tavily and Exa both failed or not configured)."
            )

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.65, 2: 0.82}.get(n, 0.7)

    async def _decompose_question(
        self, question: MetaculusQuestion
    ) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition:
            return None
        try:
            llm = self.get_llm("decomposer", "llm")
            prompt = clean_indents(
                f"""
Decompose the forecasting question into:
- 3-6 subquestions for research
- key entities
- key metrics

Return ONLY JSON:
{{"subquestions":[...], "key_entities":[...], "key_metrics":[...]}}

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
        self,
        question: MetaculusQuestion,
        decomp: Optional[DecompositionOutput],
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
                [
                    f"Source: {r.get('url','')}\nContent: {r.get('content','')}"
                    for r in response.get("results", [])
                ]
            )
            return f"[Tavily Data]\n{context}" if context.strip() else "[Tavily search failed]"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _summarize_research(self, question: MetaculusQuestion, raw_research: str) -> str:
        """
        IMPROVEMENT 7: Generate a concise 3-sentence summary of what the
        research actually found, to be recorded as the first trace step.
        """
        llm = self.get_llm("summarizer", "llm")
        prompt = clean_indents(
            f"""
You are summarizing web research for a forecaster.
Write exactly 3 sentences covering:
  1. The most relevant factual finding from the research.
  2. The strongest signal pointing toward YES / a higher value.
  3. The strongest signal pointing toward NO / a lower value.

Be specific — name figures, dates, and sources where present.

Question: {question.question_text}

Research:
{raw_research[:3000]}
"""
        )
        try:
            return (await llm.invoke(prompt)).strip()
        except Exception as e:
            logger.warning(f"Research summary failed: {e}")
            return "[Research summary unavailable]"

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        IMPROVEMENT 3: Cache research by question URL so multi-run never
        re-fetches or re-decomposes the same question.
        """
        cache_key = getattr(question, "page_url", None) or question.question_text[:80]
        if cache_key in self._research_cache:
            logger.info(f"[{self.bot_name}] Research cache hit: {cache_key}")
            return self._research_cache[cache_key]

        decomp = await self._decompose_question(question)
        queries = await self._optimize_search_query(question, decomp)
        optimized_query = " OR ".join(queries)

        results = await asyncio.gather(
            self._run_tavily_search(optimized_query),
            self._run_exa_search(optimized_query),
            return_exceptions=True,
        )
        cleaned: List[str] = []
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
        self._research_cache[cache_key] = research
        return research

    # ──────────────────────────────────────────────────────────────────────────
    # Core utilities
    # ──────────────────────────────────────────────────────────────────────────

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
        IMPROVEMENT 2: Corrected extremize gate.
        Previously: [0.60, 0.98] — skipped anything below 0.60, including
        forecasts near 0.5 that most need pushing.
        Now: [0.02, 0.98] excluding the exact midpoint — anything with genuine
        signal (not exactly 0.5) is eligible for extremization.
        The logit function naturally produces ~zero effect at 0.5, so there is
        no risk of over-pushing truly uncertain forecasts.
        """
        p = float(p)
        return 0.02 < p < 0.98 and p != 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Red-team  (IMPROVEMENT 6)
    # ──────────────────────────────────────────────────────────────────────────

    async def _red_team_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        initial_pred: float,
        trace: ReasoningTrace,
    ) -> float:
        if not self.flags.enable_red_team:
            trace.add("Red-team", "SKIPPED (flag disabled)")
            return initial_pred
        self._ensure_some_research_or_raise(research)
        llm = self.get_llm("red_team", "llm")
        try:
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a skeptical red-team forecaster. Your job is to find the SINGLE STRONGEST
argument that the current forecast is WRONG.

Question: {question.question_text}

Current forecast: {initial_pred:.2%}

Research (most relevant excerpts — focus on disconfirming evidence):
{research[:2500]}

Step 1 — State the single strongest counter-argument to the current forecast in
          1-2 sentences. Be specific: cite a figure, date, or mechanism.
Step 2 — Explain why this counter-argument should move the probability.
Step 3 — Output a revised probability that incorporates this counter-argument.

End with ONLY this JSON on the last line:
{{"revised_prediction_in_decimal": 0.XX, "counter_argument": "one sentence summary"}}
"""
                )
            )
            # Capture the full red-team narrative
            trace.add("Red-team narrative", (raw or "").strip()[:800])

            # Parse the JSON from the last line
            last_line = [l.strip() for l in raw.splitlines() if l.strip()][-1]
            parsed = json.loads(sanitize_llm_json(last_line))
            val = float(parsed.get("revised_prediction_in_decimal", initial_pred))
            counter = parsed.get("counter_argument", "")
            result = float(np.clip(val, 0.0, 1.0))
            trace.add(
                "Red-team result",
                f"revised={result:.4f} (Δ={result - initial_pred:+.4f}) | "
                f'counter: "{counter}"',
            )
            return result
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            trace.add("Red-team", f"FAILED ({e}); keeping initial={initial_pred:.4f}")
            return initial_pred

    # ──────────────────────────────────────────────────────────────────────────
    # Consistency check  (IMPROVEMENT 5)
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_consistency(
        self,
        question: MetaculusQuestion,
        proposed_pred: float,
        trace: ReasoningTrace,
    ) -> bool:
        """
        IMPROVEMENT 5: Only binary question probabilities are compared.
        Numeric question medians are excluded — they are not on the same
        scale and produce meaningless consistency judgements.
        """
        if not self.flags.enable_consistency_check:
            trace.add("Consistency check", "SKIPPED (flag disabled)")
            return True
        if len(self._recent_binary_predictions) < 2:
            trace.add(
                "Consistency check",
                f"SKIPPED (only {len(self._recent_binary_predictions)} prior binary predictions, need ≥2)",
            )
            return True
        recent_summary = "\n".join(
            [f"Q: {qt} → Pred: {p:.2%}" for qt, p in self._recent_binary_predictions[-3:]]
        )
        llm = self.get_llm("parser", "llm")
        prompt = f"""
Is this new binary forecast logically consistent with the prior binary forecasts below?
Consider whether the implied world-state is coherent across questions.

New forecast: {question.question_text} → {proposed_pred:.2%}

Prior binary forecasts:
{recent_summary}

Answer YES or NO only. Do not explain.
""".strip()
        try:
            response = await llm.invoke(prompt)
            result = "YES" in (response or "").upper()
            trace.add(
                "Consistency check (binary-only)",
                f"{'PASSED' if result else 'FAILED — applying consistency shrink'} | "
                f"compared against {len(self._recent_binary_predictions)} prior binary predictions",
            )
            return result
        except Exception:
            trace.add("Consistency check", "ERROR — treating as consistent")
            return True

    # ──────────────────────────────────────────────────────────────────────────
    # Numeric parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(
            f"""
Extract a numeric forecast distribution from the text.

Output MUST be a list of objects with fields:
  - percentile
  - value

Percentile can be:
  - 10,20,40,60,80,90
  OR
  - 0.1,0.2,0.4,0.6,0.8,0.9

Values:
  - MUST be in units: {question.unit_of_measure}
  - Never use scientific notation.

Rules:
  - Required percentiles are exactly those six.
  - Values must be strictly increasing with percentile.
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
        lines = []
        for line in (text or "").splitlines():
            if re.search(
                r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, flags=re.IGNORECASE
            ):
                lines.append(line.strip())
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_raw_percentiles(raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p = p / 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        return out

    @staticmethod
    def _require_standard_percentiles(pcts: List[Percentile]) -> List[Percentile]:
        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        by = {round(float(p.percentile), 3): p for p in pcts}
        missing = [r for r in required if round(r, 3) not in by]
        if missing:
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
        pcts = [
            Percentile(percentile=p, value=lo + (hi - lo) * w[p])
            for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        ]
        return Dezzy._enforce_monotone(pcts)

    @staticmethod
    def _median_from_40_60(pcts: List[Percentile]) -> float:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        if 0.4 in by and 0.6 in by:
            return 0.5 * (by[0.4] + by[0.6])
        return (
            float(sorted(pcts, key=lambda x: x.percentile)[len(pcts) // 2].value)
            if pcts
            else 0.0
        )

    @staticmethod
    def _p10_p90(pcts: List[Percentile]) -> Tuple[Optional[float], Optional[float]]:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        return by.get(0.1), by.get(0.9)

    @staticmethod
    def _format_pcts(pcts: List[Percentile]) -> str:
        return " | ".join(
            f"P{int(round(float(p.percentile) * 100))}={p.value:.6g}" for p in pcts
        )

    async def _parse_numeric_percentiles_robust(
        self, question: NumericQuestion, text: str, stage: str
    ) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
        instructions = self._numeric_parsing_instructions(question)

        for attempt, source in enumerate([text, self._extract_percentile_block(text)], 1):
            if not source:
                continue
            try:
                raw: List[RawPercentile] = await structure_output(
                    source,
                    list[RawPercentile],
                    model=parser_llm,
                    additional_instructions=instructions,
                    num_validation_samples=1,
                )
                normalized = self._normalize_raw_percentiles(raw)
                std = self._require_standard_percentiles(normalized)
                if std:
                    return self._enforce_monotone(std)
            except Exception as e:
                logger.warning(f"[{stage}] numeric parse attempt {attempt} failed: {e}")

        # Attempt 3: ask the LLM to reformat
        try:
            reform_prompt = clean_indents(
                f"""
Rewrite into EXACTLY these 6 lines (no extra text):

Percentile 10: <number>
Percentile 20: <number>
Percentile 40: <number>
Percentile 60: <number>
Percentile 80: <number>
Percentile 90: <number>

Rules:
- Values in units: {question.unit_of_measure}
- No scientific notation.
- Strictly increasing.

Text:
{text}
"""
            )
            reformatted = await parser_llm.invoke(reform_prompt)
            rb = self._extract_percentile_block(reformatted) or reformatted or ""
            raw3: List[RawPercentile] = await structure_output(
                rb,
                list[RawPercentile],
                model=parser_llm,
                additional_instructions=instructions,
                num_validation_samples=1,
            )
            p3 = self._normalize_raw_percentiles(raw3)
            std3 = self._require_standard_percentiles(p3)
            if std3:
                return self._enforce_monotone(std3)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 3 failed: {e}")

        logger.warning(f"[{stage}] all parse attempts failed; using bounds fallback.")
        return self._bounds_fallback(question)

    # ──────────────────────────────────────────────────────────────────────────
    # Numeric regimes
    # ──────────────────────────────────────────────────────────────────────────

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
        return any(c in r for c in cues) and self._extract_date_range_generic(
            question.question_text or ""
        ) is not None

    def _detect_numeric_regime(
        self, question: NumericQuestion, research: str
    ) -> NumericRegime:
        if not self.flags.enable_numeric_regimes:
            return NumericRegime.GENERIC
        if self._has_partial_observations(research, question):
            return NumericRegime.PARTIAL_REVEAL_SUM
        dr = self._extract_date_range_generic(question.question_text or "")
        if dr:
            start, end = dr
            horizon = (end - start).days + 1
            if 2 <= horizon <= 31:
                return NumericRegime.STRUCTURED_TS
        return NumericRegime.GENERIC

    async def _llm_extract_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> PartialRevealExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(
            f"""
Return JSON only:
{{"known_subtotal": null, "known_parts": null, "total_parts": null, "notes": null}}

Question:
{question.question_text}

Research:
{research}

Extract:
- known_subtotal if research states a subtotal/sum
- known_parts and total_parts if inferable
"""
        )
        raw = await parser.invoke(prompt)
        return safe_model(PartialRevealExtract, raw)  # type: ignore[return-value]

    async def _llm_extract_reference_class(
        self, question: NumericQuestion, research: str
    ) -> ReferenceClassExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(
            f"""
Return JSON only:
{{"reference_totals": [], "trend_multiplier": null, "notes": null}}

Question:
{question.question_text}

Research:
{research}

Extract comparable reference totals and an optional trend multiplier (0.85-1.15).
"""
        )
        raw = await parser.invoke(prompt)
        return safe_model(ReferenceClassExtract, raw)  # type: ignore[return-value]

    async def _bounded_multiplier(
        self,
        question: NumericQuestion,
        research: str,
        baseline: float,
        *,
        lo: float,
        hi: float,
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(
            f"""
Return JSON only: {{"multiplier": 1.00}}

Question: {question.question_text}
Baseline: {baseline}

Research:
{research}

Rules:
- multiplier must be within [{lo:.6f}, {hi:.6f}]
- Output only JSON.
"""
        )
        raw = await critic.invoke(prompt)
        model = safe_model(BoundedMultiplier, raw)  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "multiplier")), lo, hi))

    def _mult_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        if h <= 21:
            return (0.98, 1.02)
        if h <= 60:
            return (0.96, 1.04)
        return (0.92, 1.08)

    def _horizon_days_from_text(self, question: NumericQuestion) -> Optional[int]:
        dr = self._extract_date_range_generic(question.question_text or "")
        if not dr:
            return None
        start, end = dr
        return (end - start).days + 1

    @staticmethod
    def _normal_percentiles_from_mean_sd(mean: float, sd: float) -> List[Percentile]:
        z = {0.1: -1.2816, 0.2: -0.8416, 0.4: -0.2533, 0.6: 0.2533, 0.8: 0.8416, 0.9: 1.2816}
        out: List[Percentile] = [
            Percentile(percentile=p, value=float(mean + z[p] * sd))
            for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        ]
        return Dezzy._enforce_monotone(out)

    # ──────────────────────────────────────────────────────────────────────────
    # Model calls + multi-run  (IMPROVEMENT 1: narrative captured per run)
    # ──────────────────────────────────────────────────────────────────────────

    async def _single_model_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        run_index: int,
        trace: ReasoningTrace,
    ) -> Any:
        """
        IMPROVEMENT 1: Returns (result, narrative_text) so callers can
        embed the full LLM chain-of-thought into the ReasoningTrace.
        """
        self._ensure_some_research_or_raise(research)
        temp = self._get_temperature(question)
        llm = GeneralLlm(
            model=self._llm_config_defaults()["default"], temperature=temp
        )

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a calibrated superforecaster. Think step by step before giving your answer.

Question: {question.question_text}

Resolution criteria:
{question.resolution_criteria}

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Step 1 — What is the BASE RATE for this type of event?
Step 2 — What evidence from the research SUPPORTS YES?
Step 3 — What evidence from the research SUPPORTS NO?
Step 4 — What is the STATUS QUO if nothing changes before resolution?
Step 5 — Synthesise: what probability is best calibrated given all of the above?

OUTPUT ONLY VALID JSON on the very last line:
{{"prediction_in_decimal": 0.50}}
"""
                )
            )
            # Capture narrative (everything before the final JSON line)
            narrative = "\n".join(
                line for line in (raw or "").splitlines()
                if not line.strip().startswith("{")
            ).strip()
            trace.add_narrative(run_index, narrative)

            result = await structure_output(
                sanitize_llm_json(raw),
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            return result

        if isinstance(question, MultipleChoiceQuestion):
            schema_example = json.dumps(
                {
                    "predicted_options": [
                        {"option_name": opt, "probability": round(1 / len(question.options), 3)}
                        for opt in question.options
                    ]
                }
            )
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a calibrated superforecaster.

Question: {question.question_text}
Options: {question.options}

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Step 1 — What does the BASE RATE suggest for each option?
Step 2 — Which option is favoured by the current evidence?
Step 3 — What is the STATUS QUO option?
Step 4 — Assign calibrated probabilities summing to exactly 1.0.

OUTPUT ONLY VALID JSON on the very last line:
{schema_example}
"""
                )
            )
            narrative = "\n".join(
                line for line in (raw or "").splitlines()
                if not line.strip().startswith("{")
            ).strip()
            trace.add_narrative(run_index, narrative)

            result = await structure_output(
                sanitize_llm_json(raw),
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            return result

        if isinstance(question, NumericQuestion):
            units = question.unit_of_measure or "Not stated"
            upper = (
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )
            lower = (
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a calibrated superforecaster.

Question:
{question.question_text}

Units: {units}
Bounds: [{lower}, {upper}]

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Step 1 — What is the REFERENCE CLASS / historical base rate for this quantity?
Step 2 — What TREND does the research suggest?
Step 3 — What are the key UPSIDE risks?
Step 4 — What are the key DOWNSIDE risks?
Step 5 — How WIDE should the uncertainty interval be given the evidence?

The LAST thing you write is EXACTLY these 6 lines and nothing else after them:
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"""
                )
            )
            # Capture narrative (everything before the final percentile block)
            narrative_lines = []
            for line in (raw or "").splitlines():
                if re.match(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, re.IGNORECASE):
                    break
                narrative_lines.append(line)
            trace.add_narrative(run_index, "\n".join(narrative_lines).strip())

            return await self._parse_numeric_percentiles_robust(
                question, raw, stage=f"run{run_index}"
            )

        raise TypeError(f"Unsupported question type: {type(question)}")

    async def _multi_run(
        self,
        question: MetaculusQuestion,
        research: str,
        trace: ReasoningTrace,
    ) -> List[Any]:
        """Sequential runs — friendlier to free-tier rate limits."""
        outs: List[Any] = []
        for i in range(self.runs_per_question):
            try:
                result = await self._single_model_forecast(question, research, i + 1, trace)
                outs.append(result)
            except Exception as e:
                logger.warning(f"run {i+1}/{self.runs_per_question} failed: {e}")
                trace.add(f"Run {i+1}", f"FAILED: {e}")
        return outs

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Binary
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)

        # IMPROVEMENT 7: Research summary as first trace step
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)

        src = self._search_footprint(research)
        quality = self._research_quality_weight(research)
        trace.add("Research sources", f"{src} | quality_weight={quality:.2f}")

        # Multi-run  (narratives auto-added inside _multi_run → _single_model_forecast)
        runs = await self._multi_run(question, research, trace)
        if not runs:
            raise RuntimeError("All binary runs failed.")

        probs = [float(r.prediction_in_decimal) for r in runs]
        run_med = self._median(probs)
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        trace.add(
            f"Multi-run aggregation ({len(probs)} runs)",
            f"individual={[f'{p:.4f}' for p in probs]} | median={run_med:.4f} | spread={spread:.4f}",
        )

        applied: List[str] = []

        # Conservative shrink
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
            f"alpha={shrink:.2f} | {run_med:.4f} → {base_p:.4f} | triggers=[{', '.join(applied) or 'none'}]",
        )

        # Red-team blend
        red_p = await self._red_team_forecast(question, research, base_p, trace)
        combined = 0.6 * base_p + 0.4 * red_p
        applied.append("blend(red-team)")
        trace.add(
            "Red-team blend",
            f"0.6×{base_p:.4f} + 0.4×{red_p:.4f} = {combined:.4f}",
        )

        # Consistency check (binary-only)
        if not await self._check_consistency(question, combined, trace):
            before = combined
            combined = 0.5 * combined + 0.5 * 0.5
            applied.append("consistency-shrink")
            trace.add("Consistency shrink", f"{before:.4f} → {combined:.4f}")

        # Extremize (corrected gate)
        gate_hit = self.flags.enable_extremize and self._extremize_gate(combined)
        if gate_hit:
            ext_strength = self._extremize_strength(research, probs + [combined], question)
            p_ext = ForecastingPrinciples.extremize_logit(combined, ext_strength)
            applied.append(f"extremize(x{ext_strength:.2f})")
            trace.add(
                "Extremize",
                f"gate=OPEN (p={combined:.4f} ∈ (0.02, 0.98)) | "
                f"strength={ext_strength:.3f} | {combined:.4f} → {p_ext:.4f}",
            )
        else:
            p_ext = combined
            reason = (
                "flag disabled"
                if not self.flags.enable_extremize
                else f"p={combined:.4f} at/outside gate bounds"
            )
            applied.append("extremize(gated-off)")
            trace.add("Extremize", f"gate=CLOSED ({reason}) | p unchanged={p_ext:.4f}")

        # Time decay (softened)
        p_time = ForecastingPrinciples.apply_time_decay(
            p_ext, getattr(question, "close_time", None)
        )
        if abs(p_time - p_ext) > 1e-6:
            applied.append("time-decay")
            close_days = (
                (question.close_time - datetime.now(timezone.utc)).days
                if getattr(question, "close_time", None)
                else "N/A"
            )
            trace.add("Time decay (softened)", f"{p_ext:.4f} → {p_time:.4f} | days_to_close={close_days}")
        else:
            trace.add("Time decay", "no change (close_time not set or ≤90 days)")

        # Bayesian calibration
        try:
            if hasattr(self, "apply_bayesian_calibration"):
                p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
                if abs(p_cal - p_time) > 1e-6:
                    applied.append("bayes-calibration")
                    trace.add("Bayes calibration", f"{p_time:.4f} → {p_cal:.4f}")
                else:
                    trace.add("Bayes calibration", "no change")
            else:
                p_cal = p_time
                trace.add("Bayes calibration", "method not available on parent class")
        except Exception:
            p_cal = p_time
            trace.add("Bayes calibration", "FAILED — keeping previous value")

        final_p = float(np.clip(p_cal, 0.01, 0.99))
        trace.add("Final clamp [0.01, 0.99]", f"{p_cal:.6f} → {final_p:.4f}")
        trace.add("Pipeline summary", f"controls applied: {', '.join(applied)}")
        trace.add("★ FINAL PREDICTION", f"{final_p:.4f}  ({final_p:.1%})")

        # IMPROVEMENT 5: only store binary predictions for consistency checks
        self._recent_binary_predictions.append((question.question_text[:120], final_p))
        if len(self._recent_binary_predictions) > 20:
            self._recent_binary_predictions.pop(0)

        return ReasonedPrediction(prediction_value=final_p, reasoning=trace.render())

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Multiple choice
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)

        # IMPROVEMENT 7: Research summary
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)

        quality = self._research_quality_weight(research)
        trace.add(
            "Research sources",
            f"{self._search_footprint(research)} | quality_weight={quality:.2f}",
        )
        trace.add("Options", str(list(question.options)))

        runs = await self._multi_run(question, research, trace)
        if not runs:
            raise RuntimeError("All MC runs failed.")

        opt_names = list(question.options)
        per_opt: Dict[str, List[float]] = {o: [] for o in opt_names}
        for r in runs:
            try:
                cur = {o.option_name: float(o.probability) for o in r.predicted_options}
            except Exception:
                continue
            for o in opt_names:
                per_opt[o].append(float(cur.get(o, 0.0)))

        med_probs = {
            o: self._median(per_opt[o]) if per_opt[o] else 0.0 for o in opt_names
        }
        trace.add(
            f"Multi-run medians ({len(runs)} runs)",
            " | ".join(f"{o}={v:.4f}" for o, v in med_probs.items()),
        )

        uniform = 1.0 / max(1, len(opt_names))
        alpha = 0.10 if quality >= 0.75 else 0.18
        shrunk = {o: (1 - alpha) * med_probs[o] + alpha * uniform for o in opt_names}
        trace.add(
            "Shrink to uniform",
            f"alpha={alpha:.2f} | uniform={uniform:.4f}",
        )
        trace.add(
            "Shrunk probs",
            " | ".join(f"{o}={v:.4f}" for o, v in shrunk.items()),
        )

        total = float(sum(max(0.0, v) for v in shrunk.values()))
        if total <= 0:
            final = [{"option_name": o, "probability": uniform} for o in opt_names]
        else:
            final = [
                {
                    "option_name": o,
                    "probability": float(np.clip(shrunk[o] / total, 0.0, 1.0)),
                }
                for o in opt_names
            ]

        trace.add(
            "Normalized probs",
            " | ".join(f"{x['option_name']}={x['probability']:.4f}" for x in final),
        )
        trace.add(
            "★ FINAL PREDICTION",
            " | ".join(f"{x['option_name']}={x['probability']:.1%}" for x in final),
        )

        final_val = safe_model(PredictedOptionList, {"predicted_options": final})  # type: ignore[assignment]
        return ReasonedPrediction(prediction_value=final_val, reasoning=trace.render())

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Numeric (generic)
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_forecast_on_numeric_generic(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        trace = ReasoningTrace(question.question_text, self.bot_name)

        # IMPROVEMENT 7: Research summary
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)

        trace.add("Numeric regime", "GENERIC")
        trace.add(
            "Research sources",
            f"{self._search_footprint(research)} | quality_weight={self._research_quality_weight(research):.2f}",
        )

        runs = await self._multi_run(question, research, trace)
        if not runs:
            raise RuntimeError("All numeric runs failed.")

        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_pct: Dict[float, List[float]] = {p: [] for p in required}

        for r in runs:
            try:
                for pct in r:
                    p = float(pct.percentile)
                    v = float(pct.value)
                    if p > 1.0:
                        p = p / 100.0
                    p = round(p, 3)
                    if p in per_pct and np.isfinite(v):
                        per_pct[p].append(v)
            except Exception:
                continue

        agg: List[Percentile] = []
        for p in required:
            vals = per_pct.get(round(p, 3), [])
            if vals:
                agg.append(Percentile(percentile=p, value=float(self._median(vals))))
            else:
                trace.add(
                    "Fallback triggered",
                    f"no values for p={p} — using bounds-based fallback",
                )
                pcts = self._bounds_fallback(question)
                dist = NumericDistribution.from_question(pcts, question)
                trace.add("★ FINAL (bounds fallback)", self._format_pcts(pcts))
                return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

        agg = self._enforce_monotone(agg)
        trace.add(
            f"Aggregated percentiles ({len(runs)} runs)",
            self._format_pcts(agg),
        )
        trace.add("Monotone enforced", "yes")

        med = self._median_from_40_60(agg)
        p10, p90 = self._p10_p90(agg)
        trace.add(
            "Distribution summary",
            f"median≈{med:.6g} | P10={p10:.6g} | P90={p90:.6g}",
        )
        trace.add("★ FINAL PREDICTION", self._format_pcts(agg))

        dist = NumericDistribution.from_question(agg, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Numeric regimes
    # ──────────────────────────────────────────────────────────────────────────

    async def _forecast_numeric_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        trace.add("Numeric regime", "PARTIAL_REVEAL_SUM")

        try:
            ex = await self._llm_extract_partial_reveal(question, research)
        except Exception:
            trace.add("Partial-reveal extract", "FAILED — falling back to generic")
            return await self._run_forecast_on_numeric_generic(question, research)

        if ex.known_subtotal is None:
            trace.add("Partial-reveal extract", "known_subtotal=None — falling back to generic")
            return await self._run_forecast_on_numeric_generic(question, research)

        known = float(ex.known_subtotal)
        if not np.isfinite(known) or known <= 0:
            trace.add(
                "Partial-reveal extract",
                f"known_subtotal={known} invalid — falling back to generic",
            )
            return await self._run_forecast_on_numeric_generic(question, research)

        trace.add("Known subtotal", f"{known:.6g} | notes={ex.notes}")

        remainder_baseline = 0.75 * known
        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(
            question, research, remainder_baseline, lo=lo_m, hi=hi_m
        )
        total_mean = known + remainder_baseline * mult
        sd = max(0.10 * total_mean, 0.05 * known)

        trace.add(
            "Remainder estimate",
            f"baseline={remainder_baseline:.6g} × mult={mult:.4f} | "
            f"total_mean={total_mean:.6g} | sd={sd:.6g}",
        )

        pcts = self._normal_percentiles_from_mean_sd(total_mean, sd)
        for p in pcts:
            if p.value < known:
                p.value = known
        pcts = self._enforce_monotone(pcts)
        trace.add("★ FINAL PREDICTION", self._format_pcts(pcts))

        dist = NumericDistribution.from_question(pcts, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

    async def _forecast_numeric_structured_ts(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        trace.add("Numeric regime", "STRUCTURED_TS")

        baseline = 0.5 * (float(question.lower_bound) + float(question.upper_bound))
        try:
            ref = await self._llm_extract_reference_class(question, research)
            refs = [
                float(x)
                for x in (ref.reference_totals or [])
                if np.isfinite(float(x)) and float(x) > 0
            ]
            if refs:
                baseline = float(np.median(refs))
                trace.add(
                    "Reference class",
                    f"totals={refs} | median_baseline={baseline:.6g}",
                )
                if ref.trend_multiplier is not None and np.isfinite(float(ref.trend_multiplier)):
                    tm = float(ref.trend_multiplier)
                    if 0.85 <= tm <= 1.15:
                        baseline *= tm
                        trace.add(
                            "Trend multiplier",
                            f"×{tm:.4f} → adjusted_baseline={baseline:.6g}",
                        )
            else:
                trace.add(
                    "Reference class",
                    f"no usable totals — using midpoint baseline={baseline:.6g}",
                )
        except Exception as e:
            trace.add(
                "Reference class extract",
                f"FAILED ({e}) — using midpoint baseline={baseline:.6g}",
            )

        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(question, research, baseline, lo=lo_m, hi=hi_m)
        mean = baseline * mult

        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        width = hi - lo if np.isfinite(hi - lo) and hi > lo else max(1.0, abs(mean))
        sd = float(
            np.clip(0.10 * abs(mean) + 0.05 * width, 1e-9, 0.35 * abs(mean) + 1e-9)
        )

        trace.add(
            "Final mean & sd",
            f"baseline={baseline:.6g} × mult={mult:.4f} → mean={mean:.6g} | sd={sd:.6g}",
        )

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        for p in pcts:
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                p.value = float(np.clip(p.value, lo, hi))
        pcts = self._enforce_monotone(pcts)
        trace.add("★ FINAL PREDICTION", self._format_pcts(pcts))

        dist = NumericDistribution.from_question(pcts, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=trace.render())

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        if not self.flags.enable_numeric_regimes:
            return await self._run_forecast_on_numeric_generic(question, research)

        regime = self._detect_numeric_regime(question, research)
        if regime == NumericRegime.PARTIAL_REVEAL_SUM:
            try:
                return await self._forecast_numeric_partial_reveal(question, research)
            except Exception as e:
                logger.warning(f"Partial-reveal regime failed; fallback: {e}")
                return await self._run_forecast_on_numeric_generic(question, research)
        if regime == NumericRegime.STRUCTURED_TS:
            try:
                return await self._forecast_numeric_structured_ts(question, research)
            except Exception as e:
                logger.warning(f"Structured TS regime failed; fallback: {e}")
                return await self._run_forecast_on_numeric_generic(question, research)

        return await self._run_forecast_on_numeric_generic(question, research)

    async def _run_forecast_on_numeric_wrapper(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_forecast_on_numeric(question, research)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="dezzy: Tavily+Exa, OpenRouter free router, multi-run, full reasoning trace"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument("--bot-name", type=str, default="dezzy")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent LLM runs to aggregate per question (sequential)",
    )
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
        raise RuntimeError(
            "Set at least one of TAVILY_API_KEY or EXA_API_KEY in your environment."
        )

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
            seasonal_task = bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
            minibench_task = bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
            seasonal, minibench = await asyncio.gather(seasonal_task, minibench_task)
            return seasonal + minibench

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )

        bot.skip_previously_forecasted_questions = False
        EXAMPLE_QUESTION_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [
            client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTION_URLS
        ]
        single_reports_task = bot.forecast_questions(questions, return_exceptions=True)
        market_pulse_task = bot.forecast_on_tournament(
            "market-pulse-26q1", return_exceptions=True
        )
        single_reports, market_pulse_reports = await asyncio.gather(
            single_reports_task, market_pulse_task
        )
        return single_reports + market_pulse_reports

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
