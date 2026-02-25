import argparse
import asyncio
import logging
import os
import re
import json
from dataclasses import dataclass
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
    AskNewsSearcher, # unused but may exist in your env; safe to keep? -> we'll NOT import extras to avoid confusion
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


# ---------------------------
# Helpers
# ---------------------------
def sanitize_llm_json(text: str) -> str:
    """
    Cleans up common LLM JSON issues:
      - removes numeric underscores
      - converts quoted numeric fields to numbers when possible
      - strips ```json fences
    """
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


# ---------------------------
# Research providers (ONLY: Tavily + Exa)
# ---------------------------
class ExaSearcher:
    """
    Uses EXA_API_KEY only.
    """
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


# ---------------------------
# Forecasting principles
# ---------------------------
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
            return 0.3 * prob + 0.7 * 0.5
        if days > 180:
            return 0.5 * prob + 0.5 * 0.5
        if days > 90:
            return 0.7 * prob + 0.3 * 0.5
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
        strength = float(np.clip(strength, 0.5, 1.8)) # conservative cap
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.0, 1.0))


# ---------------------------
# Schemas / Regimes
# ---------------------------
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


# ---------------------------
# Feature flags
# ---------------------------
@dataclass
class BotFeatureFlags:
    enable_extremize: bool = True
    enable_decomposition: bool = True
    enable_numeric_regimes: bool = True
    enable_red_team: bool = True
    enable_consistency_check: bool = True


# ---------------------------
# Bot
# ---------------------------
class SpringAdvancedForecastingBot(ForecastBot):
    """
    Full version:
      - Research: Tavily + Exa only
      - LLM: OpenRouter free router only
      - Multi-run per question
      - Binary: median -> conservative shrink -> red-team -> (extremize gated 0.60-0.98) -> time-decay -> bayes calibration
      - MC: median-per-option -> shrink-to-uniform -> normalize
      - Numeric: median-per-percentile -> monotone -> regimes (partial-reveal / structured-ts) (conservative)
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

        self._recent_predictions: list[tuple[MetaculusQuestion, float]] = []

    def _llm_config_defaults(self) -> Dict[str, str]:
        # Only OpenRouter free router
        free = "openrouter/openrouter/free"
        return {
            "default": free,
            "parser": free,
            "query_optimizer": free,
            "critic": free,
            "red_team": free,
            "decomposer": free,
        }

    # ---------------------------
    # Research
    # ---------------------------
    def _search_footprint(self, research: str) -> str:
        used: list[str] = []

        def ok(tag: str, fail_markers: list[str]) -> bool:
            return (tag in research) and (not any(m in research for m in fail_markers))

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

    async def _decompose_question(self, question: MetaculusQuestion) -> Optional[DecompositionOutput]:
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
            return safe_model(DecompositionOutput, raw) # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            return None

    async def _optimize_search_query(self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]) -> List[str]:
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
                [f"Source: {r.get('url','')}\nContent: {r.get('content','')}" for r in response.get("results", [])]
            )
            return f"[Tavily Data]\n{context}" if context.strip() else "[Tavily search failed]"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def run_research(self, question: MetaculusQuestion) -> str:
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

        research = f"""{ForecastingPrinciples.get_generic_base_rate()}

{ForecastingPrinciples.get_generic_fermi_prompt()}

{combined}"""

        self._ensure_some_research_or_raise(research)
        return research

    # ---------------------------
    # Core utilities (conservative)
    # ---------------------------
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
        # conservative temps for stability
        if not getattr(question, "close_time", None):
            return 0.15
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        return 0.20 if days_to_close > 180 else 0.10

    def _agreement_strength(self, probs: List[float]) -> float:
        if not probs:
            return 0.0
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return float(np.clip(1.0 - (spread / 0.30), 0.0, 1.0))

    def _extremize_strength(self, research: str, probs: List[float], question: MetaculusQuestion) -> float:
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
        # REQUIRED: extremize only for 60% - 98%
        return 0.60 <= float(p) <= 0.98

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float) -> float:
        if not self.flags.enable_red_team:
            return initial_pred
        self._ensure_some_research_or_raise(research)
        llm = self.get_llm("red_team", "llm")
        try:
            raw = await llm.invoke(
                clean_indents(
                    f"""
You are a skeptical red teamer. Look for base-rate neglect, missing disconfirming evidence, and resolution pitfalls.

Question: {question.question_text}

Research:
{research}

Current forecast: {initial_pred:.2%}

Output ONLY JSON:
{{"revised_prediction_in_decimal": 0.XX}}
"""
                )
            )
            parsed = await structure_output(
                sanitize_llm_json(raw),
                dict,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            val = float(parsed.get("revised_prediction_in_decimal"))
            return float(np.clip(val, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            return initial_pred

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float) -> bool:
        if not self.flags.enable_consistency_check:
            return True
        if len(self._recent_predictions) < 2:
            return True
        recent_summary = "\n".join(
            [f"Q: {getattr(q, 'question_text', '')} → Pred: {p:.2%}" for q, p in self._recent_predictions[-3:]]
        )
        llm = self.get_llm("parser", "llm")
        prompt = f"""
Is this new forecast logically consistent with prior forecasts?

New: {question.question_text} → {proposed_pred:.2%}

Prior:
{recent_summary}

Answer YES or NO only.
""".strip()
        try:
            response = await llm.invoke(prompt)
            return "YES" in (response or "").upper()
        except Exception:
            return True

    # ---------------------------
    # Numeric parsing
    # ---------------------------
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
            if re.search(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, flags=re.IGNORECASE):
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
        pcts = [Percentile(percentile=p, value=lo + (hi - lo) * w[p]) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]
        return SpringAdvancedForecastingBot._enforce_monotone(pcts)

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

    async def _parse_numeric_percentiles_robust(self, question: NumericQuestion, text: str, stage: str) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
        instructions = self._numeric_parsing_instructions(question)

        try:
            raw1: List[RawPercentile] = await structure_output(
                text,
                list[RawPercentile],
                model=parser_llm,
                additional_instructions=instructions,
                num_validation_samples=1,
            )
            p1 = self._normalize_raw_percentiles(raw1)
            std1 = self._require_standard_percentiles(p1)
            if std1:
                return self._enforce_monotone(std1)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 1 failed: {e}")

        block = self._extract_percentile_block(text)
        if block:
            try:
                raw2: List[RawPercentile] = await structure_output(
                    block,
                    list[RawPercentile],
                    model=parser_llm,
                    additional_instructions=instructions,
                    num_validation_samples=1,
                )
                p2 = self._normalize_raw_percentiles(raw2)
                std2 = self._require_standard_percentiles(p2)
                if std2:
                    return self._enforce_monotone(std2)
            except Exception as e:
                logger.warning(f"[{stage}] numeric parse attempt 2 failed: {e}")

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
            rb = self._extract_percentile_block(reformatted) or (reformatted or "")
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

        logger.warning(f"[{stage}] numeric parsing failed; using bounds fallback.")
        return self._bounds_fallback(question)

    # ---------------------------
    # Numeric regimes (conservative)
    # ---------------------------
    def _extract_date_range_generic(self, text: str) -> Optional[Tuple[date, date]]:
        m = re.search(
            r"\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*-\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*",
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
        return (any(c in r for c in cues) and self._extract_date_range_generic(question.question_text or "") is not None)

    def _detect_numeric_regime(self, question: NumericQuestion, research: str) -> NumericRegime:
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

    async def _llm_extract_partial_reveal(self, question: NumericQuestion, research: str) -> PartialRevealExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"known_subtotal": null, "known_parts": null, "total_parts": null, "notes": null}}

Question:
{question.question_text}

Research:
{research}

Extract:
- known_subtotal if research states a subtotal/sum
- known_parts and total_parts if inferable
""")
        raw = await parser.invoke(prompt)
        return safe_model(PartialRevealExtract, raw) # type: ignore[return-value]

    async def _llm_extract_reference_class(self, question: NumericQuestion, research: str) -> ReferenceClassExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"reference_totals": [], "trend_multiplier": null, "notes": null}}

Question:
{question.question_text}

Research:
{research}

Extract comparable reference totals and an optional trend multiplier (0.85-1.15).
""")
        raw = await parser.invoke(prompt)
        return safe_model(ReferenceClassExtract, raw) # type: ignore[return-value]

    async def _bounded_multiplier(self, question: NumericQuestion, research: str, baseline: float, *, lo: float, hi: float) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"multiplier": 1.00}}

Question: {question.question_text}
Baseline: {baseline}

Research:
{research}

Rules:
- multiplier must be within [{lo:.6f}, {hi:.6f}]
- Output only JSON.
""")
        raw = await critic.invoke(prompt)
        model = safe_model(BoundedMultiplier, raw) # type: ignore[arg-type]
        m = float(getattr(model, "multiplier"))
        return float(np.clip(m, lo, hi))

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
        out: List[Percentile] = []
        for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
            out.append(Percentile(percentile=p, value=float(mean + z[p] * sd)))
        return SpringAdvancedForecastingBot._enforce_monotone(out)

    # ---------------------------
    # Model calls (OpenRouter free router only) + multi-run
    # ---------------------------
    async def _single_model_forecast(self, question: MetaculusQuestion, research: str) -> Any:
        self._ensure_some_research_or_raise(research)
        temp = self._get_temperature(question)
        model_name = self._llm_config_defaults()["default"]
        llm = GeneralLlm(model=model_name, temperature=temp)

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(
                clean_indents(
                    f"""
Question: {question.question_text}

Research:
{research}

OUTPUT ONLY VALID JSON:
{{"prediction_in_decimal": 0.50}}
"""
                )
            )
            return await structure_output(
                sanitize_llm_json(raw),
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )

        if isinstance(question, MultipleChoiceQuestion):
            schema_example = json.dumps(
                {"predicted_options": [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]}
            )
            raw = await llm.invoke(
                clean_indents(
                    f"""
Question: {question.question_text}
Options: {question.options}

Research:
{research}

OUTPUT ONLY VALID JSON:
{schema_example}
"""
                )
            )
            return await structure_output(
                sanitize_llm_json(raw),
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )

        if isinstance(question, NumericQuestion):
            units = question.unit_of_measure if question.unit_of_measure else "Not stated"
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            reasoning = await llm.invoke(
                clean_indents(
                    f"""
Question:
{question.question_text}

Units: {units}
Bounds: [{lower}, {upper}]

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

The LAST thing you write is EXACTLY:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""
                )
            )
            return await self._parse_numeric_percentiles_robust(question, reasoning, stage="model_numeric")

        raise TypeError(f"Unsupported question type: {type(question)}")

    async def _multi_run(self, question: MetaculusQuestion, research: str) -> List[Any]:
        # Sequential: friendlier to free-tier rate limits
        outs: List[Any] = []
        for i in range(self.runs_per_question):
            try:
                outs.append(await self._single_model_forecast(question, research))
            except Exception as e:
                logger.warning(f"run {i+1}/{self.runs_per_question} failed: {e}")
        return outs

    # ---------------------------
    # Reasoning helpers
    # ---------------------------
    def _methodology_header(self, research: str) -> str:
        src = self._search_footprint(research)
        return (
            f"[{self.bot_name}] methodology: research({src}); openrouter/free runs={self.runs_per_question}; "
            f"median aggregation + conservative shrink; red-team; extremize gated to p∈[0.60,0.98]."
        )

    def _short_reasoning_binary(
        self,
        research: str,
        final_p: float,
        run_med: float,
        red_p: float,
        p_ext: float,
        spread: float,
        quality: float,
        applied: List[str],
    ) -> str:
        applied_txt = ", ".join(applied) if applied else "none"
        return (
            f"{self._methodology_header(research)} "
            f"Binary: run_med={run_med:.3f} spread={spread:.3f} q={quality:.2f}; "
            f"red={red_p:.3f} ext={p_ext:.3f}; controls({applied_txt}); final={final_p:.3f}."
        )

    def _short_reasoning_mc(self, research: str, alpha: float) -> str:
        return f"{self._methodology_header(research)} MC: median-per-option; shrink-to-uniform(alpha={alpha:.2f}); normalized."

    def _short_reasoning_numeric(self, research: str, pcts: List[Percentile], regime: str) -> str:
        med = self._median_from_40_60(pcts)
        p10, p90 = self._p10_p90(pcts)
        tail = f"median≈{med:.6g}" + (f", 10–90≈[{p10:.6g},{p90:.6g}]" if (p10 is not None and p90 is not None) else "")
        return f"{self._methodology_header(research)} Numeric({regime}): monotone enforced; {tail}."

    # ---------------------------
    # Forecasting: Binary
    # ---------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        runs = await self._multi_run(question, research)
        if not runs:
            raise RuntimeError("All binary runs failed.")

        probs = [float(r.prediction_in_decimal) for r in runs]
        run_med = self._median(probs)
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        quality = self._research_quality_weight(research)

        applied: List[str] = []

        # Conservative shrink based on instability + evidence weakness
        shrink = 0.12
        if spread >= 0.20:
            shrink = 0.28
            applied.append("shrink(high-spread)")
        if quality < 0.70:
            shrink = max(shrink, 0.22)
            applied.append("shrink(low-research)")

        base_p = self._shrink_to_half(run_med, shrink)

        red_p = await self._red_team_forecast(question, research, base_p)
        combined = 0.6 * base_p + 0.4 * red_p
        applied.append("blend(red-team)")

        if not await self._check_consistency(question, combined):
            combined = 0.5 * combined + 0.5 * 0.5
            applied.append("consistency-shrink")

        # Extremize ONLY if combined is in [0.60, 0.98]
        if self.flags.enable_extremize and self._extremize_gate(combined):
            ext_strength = self._extremize_strength(research, probs + [combined], question)
            p_ext = ForecastingPrinciples.extremize_logit(combined, ext_strength)
            if abs(ext_strength - 1.0) > 0.03:
                applied.append(f"extremize(x{ext_strength:.2f})")
        else:
            p_ext = combined
            applied.append("extremize(gated-off)")

        p_time = ForecastingPrinciples.apply_time_decay(p_ext, getattr(question, "close_time", None))
        if p_time != p_ext:
            applied.append("time-decay")

        try:
            p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
            if p_cal != p_time:
                applied.append("bayes-calibration")
        except Exception:
            p_cal = p_time

        final_p = float(np.clip(p_cal, 0.01, 0.99))
        self._recent_predictions.append((question, final_p))

        reasoning = self._short_reasoning_binary(
            research=research,
            final_p=final_p,
            run_med=run_med,
            red_p=red_p,
            p_ext=p_ext,
            spread=spread,
            quality=quality,
            applied=applied,
        )
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    # ---------------------------
    # Forecasting: Multiple choice
    # ---------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        runs = await self._multi_run(question, research)
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

        med_probs = {o: self._median(per_opt[o]) if per_opt[o] else 0.0 for o in opt_names}

        # Conservative shrink towards uniform (more if research weaker)
        quality = self._research_quality_weight(research)
        uniform = 1.0 / max(1, len(opt_names))
        alpha = 0.10 if quality >= 0.75 else 0.18
        shrunk = {o: (1 - alpha) * med_probs[o] + alpha * uniform for o in opt_names}

        total = float(sum(max(0.0, v) for v in shrunk.values()))
        if total <= 0:
            final = [{"option_name": o, "probability": uniform} for o in opt_names]
        else:
            final = [{"option_name": o, "probability": float(np.clip(shrunk[o] / total, 0.0, 1.0))} for o in opt_names]

        final_val = safe_model(PredictedOptionList, {"predicted_options": final}) # type: ignore[assignment]
        reasoning = self._short_reasoning_mc(research, alpha)
        self._recent_predictions.append((question, float(np.mean([x["probability"] for x in final]))))
        return ReasonedPrediction(prediction_value=final_val, reasoning=reasoning)

    # ---------------------------
    # Forecasting: Numeric (generic aggregation)
    # ---------------------------
    async def _run_forecast_on_numeric_generic(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        runs = await self._multi_run(question, research)
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
                pcts = self._bounds_fallback(question)
                dist = NumericDistribution.from_question(pcts, question)
                reasoning = f"{self._methodology_header(research)} Numeric fallback: bounds-based percentiles; monotone enforced."
                return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

        agg = self._enforce_monotone(agg)
        dist = NumericDistribution.from_question(agg, question)
        reasoning = self._short_reasoning_numeric(research, agg, regime="generic")
        med = self._median_from_40_60(agg)
        self._recent_predictions.append((question, float(med / (abs(med) + 1.0)) if med else 0.0))
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _forecast_numeric_partial_reveal(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        # Conservative: only if extractable; else generic
        try:
            ex = await self._llm_extract_partial_reveal(question, research)
        except Exception:
            return await self._run_forecast_on_numeric_generic(question, research)

        if ex.known_subtotal is None:
            return await self._run_forecast_on_numeric_generic(question, research)

        known = float(ex.known_subtotal)
        if not np.isfinite(known) or known <= 0:
            return await self._run_forecast_on_numeric_generic(question, research)

        remainder_baseline = 0.75 * known
        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(question, research, remainder_baseline, lo=lo_m, hi=hi_m)

        total_mean = known + remainder_baseline * mult
        sd = max(0.10 * total_mean, 0.05 * known)
        pcts = self._normal_percentiles_from_mean_sd(total_mean, sd)
        for p in pcts:
            if p.value < known:
                p.value = known
        pcts = self._enforce_monotone(pcts)

        dist = NumericDistribution.from_question(pcts, question)
        reasoning = self._short_reasoning_numeric(research, pcts, regime="partial_reveal_sum")
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _forecast_numeric_structured_ts(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        baseline = 0.5 * (float(question.lower_bound) + float(question.upper_bound))
        try:
            ref = await self._llm_extract_reference_class(question, research)
            refs = [float(x) for x in (ref.reference_totals or []) if np.isfinite(float(x)) and float(x) > 0]
            if refs:
                baseline = float(np.median(refs))
                if ref.trend_multiplier is not None and np.isfinite(float(ref.trend_multiplier)):
                    tm = float(ref.trend_multiplier)
                    if 0.85 <= tm <= 1.15:
                        baseline *= tm
        except Exception:
            pass

        horizon = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult = await self._bounded_multiplier(question, research, baseline, lo=lo_m, hi=hi_m)
        mean = baseline * mult

        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        width = hi - lo if np.isfinite(hi - lo) and hi > lo else max(1.0, abs(mean))
        sd = float(np.clip(0.10 * abs(mean) + 0.05 * width, 1e-9, 0.35 * abs(mean) + 1e-9))

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        for p in pcts:
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                p.value = float(np.clip(p.value, lo, hi))
        pcts = self._enforce_monotone(pcts)

        dist = NumericDistribution.from_question(pcts, question)
        reasoning = self._short_reasoning_numeric(research, pcts, regime="structured_ts")
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
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

    async def _run_forecast_on_numeric_wrapper(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_forecast_on_numeric(question, research)


# ---------------------------
# CLI / Runs
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="dezzy: Tavily+Exa, OpenRouter free router, multi-run, extremize gated 60-98%"
    )
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    parser.add_argument("--bot-name", type=str, default="dezzy")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs to aggregate per question (sequential)")

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
        raise RuntimeError("Set at least one of TAVILY_API_KEY or EXA_API_KEY in your environment.")

    bot = SpringAdvancedForecastingBot(
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
            seasonal_task = bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
            minibench_task = bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
            seasonal, minibench = await asyncio.gather(seasonal_task, minibench_task)
            return seasonal + minibench

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)

        bot.skip_previously_forecasted_questions = False
        EXAMPLE_QUESTION_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTION_URLS]
        single_reports_task = bot.forecast_questions(questions, return_exceptions=True)
        market_pulse_task = bot.forecast_on_tournament("market-pulse-26q1", return_exceptions=True)
        single_reports, market_pulse_reports = await asyncio.gather(single_reports_task, market_pulse_task)
        return single_reports + market_pulse_reports

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
