import argparse
import asyncio
import logging
import os
import re
import json
import math
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

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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

def _fetch_yfinance_data_sync(ticker: str) -> str:
    if not YFINANCE_AVAILABLE: return ""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty: return ""
        spot = hist['Close'].iloc[-1]
        high_52 = tk.info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = tk.info.get('fiftyTwoWeekLow', 'N/A')
        vol = hist['Close'].pct_change().dropna().std() * math.sqrt(252)
        monthly_vol = vol * math.sqrt(21/252)
        rw_p10 = spot * math.exp(-1.28 * monthly_vol)
        rw_p90 = spot * math.exp(1.28 * monthly_vol)
        return (f"--- LIVE MARKET DATA ({ticker}) ---\n"
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

        self._research_cache: Dict[str, str] = {}
        self._recent_binary_predictions: List[Tuple[str, float]] = []
        self._active_tournament: str = ""

    def set_active_tournament(self, tid: str) -> None:
        self._active_tournament = str(tid).strip().lower()
        logger.info(f"[{self.bot_name}] Active tournament set to: '{self._active_tournament}'")

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
    # Research & YFinance
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
            raise RuntimeError("No research evidence available (Tavily and Exa both failed or not configured).")

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none": return 0.25
        return {1: 0.65, 2: 0.82}.get(len(srcs.split(",")), 0.7)

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
                    fin_data = await loop.run_in_executor(None, _fetch_yfinance_data_sync, ticker)
            except Exception as e:
                logger.warning(f"Ticker extraction failed: {e}")

        decomp = await self._decompose_question(question)
        queries = await self._optimize_search_query(question, decomp)
        optimized_query = " OR ".join(queries)

        results = await asyncio.gather(self._run_tavily_search(optimized_query), self._run_exa_search(optimized_query), return_exceptions=True)
        cleaned = [f"[Search failed: {str(res)}]" if isinstance(res, Exception) else res for res in results]
        
        research = (
            f"{fin_data}"
            f"{ForecastingPrinciples.get_generic_base_rate()}\n\n"
            f"{ForecastingPrinciples.get_generic_fermi_prompt()}\n\n"
            f"{chr(10).join(cleaned).strip()}"
        )
        self._ensure_some_research_or_raise(research)
        self._research_cache[cache_key] = research
        return research

    # ──────────────────────────────────────────────────────────────────────────
    # Core Aggregation & Confidence Gate
    # ──────────────────────────────────────────────────────────────────────────

    def _check_spring_ai_confidence(self, trace: ReasoningTrace, spread: float, quality: float):
        is_spring_ai = self._active_tournament in ["32916", str(MetaculusClient().CURRENT_AI_COMPETITION_ID)]
        if not is_spring_ai: return

        trace.add("Spring AI Confidence Gate", f"Evaluating... spread={spread:.4f}, quality={quality:.2f}")
        
        if spread > 0.20:
            raise RuntimeError(f"Low Confidence (Spring AI): Model spread too high ({spread:.2f} > 0.20). Skipping.")
        if quality < 0.65:
            raise RuntimeError(f"Low Confidence (Spring AI): Research quality too low ({quality:.2f} < 0.65). Skipping.")
            
        trace.add("Spring AI Confidence Gate", "PASSED. High confidence adjudged.")

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

    def _minibench_extremize_binary(self, blend: float, probs: List[float], research: str) -> Tuple[float, float, str]:
        agree = all(p > 0.5 for p in probs) or all(p < 0.5 for p in probs) if probs else False
        strong = len(self._CONVICTION_RE.findall(research or "")) >= 2
        in_zone = 0.44 <= blend <= 0.52

        if in_zone and agree and strong:
            pos = blend > 0.50
            result = 0.82 if pos else 0.18
            return result, 7.0, f"T5({'pos' if pos else 'neg'} {blend:.3f}->{result:.3f})"

        k = 5.0
        triggers = ["T1(base)"]
        if agree:
            k = min(k + 1.0, 7.0); triggers.append("T2(agree)")
        if strong:
            k = min(k + 1.0, 7.0); triggers.append("T3(research)")

        result = ForecastingPrinciples.extremize_logit(blend, k)
        if 0.40 <= result <= 0.60:
            result = float(np.clip(ForecastingPrinciples.sigmoid(1.5 * ForecastingPrinciples.logit(result)), 0.01, 0.99))
            triggers.append("T4(gate)")
        return result, k, "+".join(triggers)

    # ──────────────────────────────────────────────────────────────────────────
    # Red-team & Consistency
    # ──────────────────────────────────────────────────────────────────────────

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float, trace: ReasoningTrace) -> float:
        if not self.flags.enable_red_team:
            trace.add("Red-team", "SKIPPED")
            return initial_pred
        self._ensure_some_research_or_raise(research)
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

    async def _single_model_forecast(self, question: MetaculusQuestion, research: str, run_index: int, trace: ReasoningTrace) -> Any:
        self._ensure_some_research_or_raise(research)
        llm = GeneralLlm(model=self._llm_config_defaults()["default"], temperature=self._get_temperature(question))

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster. Think step by step before giving your answer.
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Research: {research}
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                OUTPUT ONLY VALID JSON on the very last line: {{"prediction_in_decimal": 0.50}}
            """))
            trace.add_narrative(run_index, "\n".join(line for line in (raw or "").splitlines() if not line.strip().startswith("{")).strip())
            return await structure_output(sanitize_llm_json(raw), BinaryPrediction, model=self.get_llm("parser", "llm"), num_validation_samples=1)

        if isinstance(question, MultipleChoiceQuestion):
            schema_example = json.dumps({"predicted_options": [{"option_name": opt, "probability": round(1 / len(question.options), 3)} for opt in question.options]})
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster.
                Question: {question.question_text}
                Options: {question.options}
                Research: {research}
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
                Question: {question.question_text}
                Units: {question.unit_of_measure or "Not stated"} | Bounds: [{lower}, {upper}]
                Research: {research}
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

    async def _multi_run(self, question: MetaculusQuestion, research: str, trace: ReasoningTrace) -> List[Any]:
        outs: List[Any] = []
        for i in range(self.runs_per_question):
            try: outs.append(await self._single_model_forecast(question, research, i + 1, trace))
            except Exception as e:
                logger.warning(f"run {i+1}/{self.runs_per_question} failed: {e}")
                trace.add(f"Run {i+1}", f"FAILED: {e}")
        return outs

    # ──────────────────────────────────────────────────────────────────────────
    # Forecasting: Aggregations & Logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)
        trace = ReasoningTrace(question.question_text, self.bot_name)

        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        quality = self._research_quality_weight(research)
        trace.add("Research sources", f"{self._search_footprint(research)} | quality_weight={quality:.2f}")

        runs = await self._multi_run(question, research, trace)
        if not runs: raise RuntimeError("All binary runs failed.")

        probs = [float(r.prediction_in_decimal) for r in runs]
        run_med = self._median(probs)
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        
        # Spring AI Confidence Gate
        self._check_spring_ai_confidence(trace, spread, quality)

        trace.add(f"Multi-run aggregation ({len(probs)} runs)", f"individual={[f'{p:.4f}' for p in probs]} | median={run_med:.4f} | spread={spread:.4f}")
        applied: List[str] = []

        shrink = 0.28 if spread >= 0.20 else (0.22 if quality < 0.70 else 0.12)
        base_p = self._shrink_to_half(run_med, shrink)
        applied.append(f"shrink(alpha={shrink:.2f})")

        red_p = await self._red_team_forecast(question, research, base_p, trace)
        combined = 0.6 * base_p + 0.4 * red_p
        applied.append("blend(red-team)")

        if not await self._check_consistency(question, combined, trace):
            combined = 0.5 * combined + 0.5 * 0.5
            applied.append("consistency-shrink")

        # Dynamic Extremize (Minibench or Standard)
        if self.flags.enable_extremize:
            if "minibench" in self._active_tournament:
                p_ext, eff_k, trigs = self._minibench_extremize_binary(combined, probs, research)
                applied.append(f"extremize(mb: {trigs})")
                trace.add("Minibench Extremize", f"k={eff_k} triggers=[{trigs}] | {combined:.4f} -> {p_ext:.4f}")
            elif self._extremize_gate(combined):
                ext_strength = self._extremize_strength(research, probs + [combined], question)
                p_ext = ForecastingPrinciples.extremize_logit(combined, ext_strength)
                applied.append(f"extremize(x{ext_strength:.2f})")
                trace.add("Extremize", f"gate=OPEN | strength={ext_strength:.3f} | {combined:.4f} -> {p_ext:.4f}")
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
        self._ensure_some_research_or_raise(research)
        trace = ReasoningTrace(question.question_text, self.bot_name)

        trace.add("Research summary", await self._summarize_research(question, research))
        quality = self._research_quality_weight(research)

        runs = await self._multi_run(question, research, trace)
        if not runs: raise RuntimeError("All MC runs failed.")

        opt_names = list(question.options)
        per_opt: Dict[str, List[float]] = {o: [] for o in opt_names}
        for r in runs:
            try: cur = {o.option_name: float(o.probability) for o in r.predicted_options}
            except Exception: continue
            for o in opt_names: per_opt[o].append(float(cur.get(o, 0.0)))

        med_probs = {o: self._median(per_opt[o]) if per_opt[o] else 0.0 for o in opt_names}
        
        # Calculate MC spread for Confidence Gate
        max_spread = max([(max(per_opt[o]) - min(per_opt[o])) for o in opt_names if per_opt[o]] + [0])
        self._check_spring_ai_confidence(trace, max_spread, quality)

        uniform = 1.0 / max(1, len(opt_names))
        alpha = 0.10 if quality >= 0.75 else 0.18
        shrunk = {o: (1 - alpha) * med_probs[o] + alpha * uniform for o in opt_names}

        total = float(sum(max(0.0, v) for v in shrunk.values()))
        final = [{"option_name": o, "probability": uniform} for o in opt_names] if total <= 0 else [{"option_name": o, "probability": float(np.clip(shrunk[o] / total, 0.0, 1.0))} for o in opt_names]

        trace.add("★ FINAL PREDICTION", " | ".join(f"{x['option_name']}={x['probability']:.1%}" for x in final))
        return ReasonedPrediction(prediction_value=safe_model(PredictedOptionList, {"predicted_options": final}), reasoning=trace.render())

    async def _run_forecast_on_numeric_generic(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)
        trace = ReasoningTrace(question.question_text, self.bot_name)

        trace.add("Research summary", await self._summarize_research(question, research))
        quality = self._research_quality_weight(research)

        runs = await self._multi_run(question, research, trace)
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
        
        # Calculate Numeric relative spread for Spring AI Gate
        p10, p90 = self._p10_p90(agg)
        med = self._median_from_40_60(agg)
        rel_spread = (p90 - p10) / abs(med) if p10 is not None and p90 is not None and med != 0 else 0.0
        self._check_spring_ai_confidence(trace, rel_spread, quality)

        trace.add("★ FINAL PREDICTION", self._format_pcts(agg))
        return ReasonedPrediction(prediction_value=NumericDistribution.from_question(agg, question), reasoning=trace.render())

    async def _forecast_numeric_partial_reveal(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        trace.add("Research summary", await self._summarize_research(question, research))
        
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
        self._ensure_some_research_or_raise(research)
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

    parser = argparse.ArgumentParser(description="dezzy: Tavily+Exa, OpenRouter, YFinance, Minibench Extremize, Spring AI Gate")
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    parser.add_argument("--bot-name", type=str, default="dezzy")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent agent runs to aggregate per question")
    parser.add_argument("--no-extremize", action="store_true")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-numeric-regimes", action="store_true")
    parser.add_argument("--no-red-team", action="store_true")
    parser.add_argument("--no-consistency", action="store_true")

    args = parser.parse_args()
    flags = BotFeatureFlags(
        enable_extremize=not args.no_extremize, enable_decomposition=not args.no_decomposition,
        enable_numeric_regimes=not args.no_numeric_regimes, enable_red_team=not args.no_red_team,
        enable_consistency_check=not args.no_consistency,
    )

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("EXA_API_KEY"):
        raise RuntimeError("Set at least one of TAVILY_API_KEY or EXA_API_KEY in your environment.")

    bot = Dezzy(
        research_reports_per_question=1, predictions_per_research_report=1,
        publish_reports_to_metaculus=True, skip_previously_forecasted_questions=True,
        bot_name=args.bot_name, flags=flags, runs_per_question=max(1, int(args.runs)),
    )

    client = MetaculusClient()

    async def run_all():
        if args.mode == "tournament":
            bot.set_active_tournament(str(client.CURRENT_AI_COMPETITION_ID))
            seasonal_task = bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
            
            bot.set_active_tournament(str(client.CURRENT_MINIBENCH_ID))
            minibench_task = bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
            
            seasonal, minibench = await asyncio.gather(seasonal_task, minibench_task)
            return seasonal + minibench

        if args.mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            bot.set_active_tournament(str(client.CURRENT_METACULUS_CUP_ID))
            return await bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)

        bot.skip_previously_forecasted_questions = False
        bot.set_active_tournament("market-pulse-26q1")
        return await bot.forecast_on_tournament("market-pulse-26q1", return_exceptions=True)

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
