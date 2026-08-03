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
    DateQuestion,
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
            # Without this the API returns title + url only, so every snippet was
            # empty: the source counted toward research_quality while contributing
            # nothing but headlines. See https://exa.ai/docs/reference/search
            "contents": {"text": {"maxCharacters": 1200}},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = []
                empty_bodies = 0
                for r in data.get("results", []):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = (r.get("text") or r.get("summary") or "").strip()[:900]
                    if not snippet:
                        empty_bodies += 1
                        continue
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                if empty_bodies:
                    logger.warning(f"Exa returned {empty_bodies} result(s) with no text body")
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

@dataclass(frozen=True)
class SpreadScale:
    """Every threshold keyed to run-to-run spread, plus the regime it was set in.

    `spread` means different things depending on how the forecast passes are
    configured. With all passes on one model at temperature 0.1-0.2 it is
    within-model sampling variance. With passes on different models it is
    between-model disagreement, a structurally larger quantity. A bare 0.20 in
    the code cannot tell you which it was calibrated against, so it silently
    outlives its regime. This carries the regime with the numbers.
    """
    regime: str
    gate_armed: bool
    gate_limit: Optional[float]
    heavy_shrink_at: Optional[float]
    agreement_full_disagreement_at: Optional[float]
    extremize_max: float
    # Agreement value used when spread cannot be interpreted. 1.0 keeps extremize
    # driven by research quality alone (which is calibrated independently of
    # spread) under the lower extremize_max ceiling. Set to 0.0 to turn extremize
    # off entirely while uncalibrated. This is the one judgement call in the
    # cross-model preset rather than a disarm; both options are one edit.
    agreement_when_uncalibrated: float = 1.0

    @property
    def spread_is_calibrated(self) -> bool:
        return self.gate_limit is not None

# Calibrated against o3-only passes at temperature 0.10-0.20. These are the values
# that shipped before forecast passes were split across models.
SAME_MODEL_SAMPLING = SpreadScale(
    regime="same-model sampling variance (o3 x N, temp 0.10-0.20)",
    gate_armed=True,
    gate_limit=0.20,
    heavy_shrink_at=0.20,
    agreement_full_disagreement_at=0.30,
    extremize_max=1.60,
)

# Passes on different models. No distribution of cross-model spread exists yet, so
# every spread-keyed threshold is disarmed rather than guessed at. #calibration is
# collecting spread with model_set per row; the gate is re-armed from that data.
CROSS_MODEL_UNCALIBRATED = SpreadScale(
    regime="cross-model disagreement (UNCALIBRATED - thresholds disarmed)",
    gate_armed=False,
    gate_limit=None,
    heavy_shrink_at=None,
    agreement_full_disagreement_at=None,
    # Spread can no longer suppress over-sharpening, so cap sharpening harder
    # until it can. Conservative, not a re-tune.
    extremize_max=1.20,
    agreement_when_uncalibrated=1.0,
)

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

    # Smoke-test target. https://www.metaculus.com/tournament/bot-testing-area/
    TEST_TOURNAMENT_ID = "bot-testing-area"

    @staticmethod
    def default_tournament_ids() -> List[str]:
        return ["33022", "market-pulse-26q2"]

    # The forecast passes. _single_model_forecast used to hardcode
    # "openrouter/openai/o3" for every pass, so three "independent runs" were one
    # model at temperature 0.10-0.20 and `spread` measured sampling noise. These
    # are one per vendor family so spread measures genuine disagreement.
    #
    # Ranking source: FutureSearch BTF-3, June-July 2026, pooled Brier lower-better
    # (https://evals.futuresearch.ai/): Claude Opus 5 xhigh 0.118, Opus 4.8 xhigh
    # 0.130, Fable 5 0.131, GPT-5.5 agent-SDK 0.134, GPT-5.6 Sol 0.135, GPT-5.5
    # 0.143, Sonnet 5 0.154. o3 does not appear on that board at all.
    # Metaculus' own FutureEval model leaderboard could not be read (Cloudflare),
    # so this is justified on BTF-3 alone.
    # Every slug below was checked against
    # https://openrouter.ai/api/v1/models/<id>/endpoints and resolves.
    FORECASTER_MODELS: List[str] = [
        "openrouter/anthropic/claude-opus-5",          # BTF-3 best single model, 0.116
        "openrouter/openai/gpt-5.6-sol",               # BTF-3 best available OpenAI, 0.135
        "openrouter/anthropic/claude-opus-4.8",        # BTF-3 0.130; see note below
    ]
    # Probed by --model-check so reachability of alternates is measured rather than
    # assumed. google/gemini-3.1-pro-preview would give a third vendor family, but
    # it returned 429 on this account and Google has no non-preview pro slug on
    # OpenRouter, which is not something to hang a 35-minute cron on. Pass 3 is
    # therefore a different Anthropic generation: measured on BTF-3 and stable, at
    # the cost of more correlation with pass 1 than a third family would give.
    FORECASTER_CANDIDATES: List[str] = [
        "openrouter/google/gemini-3.1-pro-preview",
        "openrouter/x-ai/grok-4.5",
        "openrouter/anthropic/claude-fable-5",
        "openrouter/openai/gpt-5.5",
        "openrouter/anthropic/claude-sonnet-5",
    ]

    def _llm_config_defaults(self) -> Dict[str, str]:
        return {
            "default":         "openrouter/anthropic/claude-opus-5",
            "parser":          "openrouter/openai/gpt-4.1-mini",
            "query_optimizer": "openrouter/anthropic/claude-sonnet-5",
            "critic":          "openrouter/openai/gpt-5.6-sol",
            "red_team":        "openrouter/anthropic/claude-opus-5",
            "decomposer":      "openrouter/anthropic/claude-sonnet-5",
            "summarizer":      "openrouter/openai/gpt-4.1-mini",
            # gpt-oss-120b 404s on this OpenRouter account ("No allowed providers
            # are available for the selected model"), which is why the recall step
            # logged "GPT-OSS research failed" on live runs. Verified reachable.
            "researcher":      "openrouter/openai/gpt-4.1-mini",
            "online_researcher": "openrouter/openai/gpt-4.1-mini",
            "research_synthesizer": "openrouter/openai/gpt-4.1-mini",
        }

    @property
    def spread_scale(self) -> SpreadScale:
        distinct = len({m for m in self.FORECASTER_MODELS[: self.runs_per_question]})
        return SAME_MODEL_SAMPLING if distinct <= 1 else CROSS_MODEL_UNCALIBRATED

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
        "synthesis": "[Model Synthesis]",
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

    @staticmethod
    def _asknews_articles(response: Any) -> List[Any]:
        """asknews_sdk returns SearchResponse(as_dicts=[...], as_string=str).

        The old code read `response.articles`, which does not exist on any
        version of the SDK, so every AskNews call raised
        AttributeError and was swallowed into "[AskNews search failed]".
        Checked against asknews_sdk.dto.news.SearchResponse. `articles` is
        tolerated first in case a future version adds it.
        """
        for attr in ("articles", "as_dicts"):
            items = getattr(response, attr, None)
            if items:
                return list(items)
        return []

    async def _run_asknews_search(self, query: str) -> str:
        if not self.asknews: return "[AskNews not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.asknews.news.search_news(
                    query=query, n_articles=6, hours_back=24 * 7, strategy="latest news",
                    # Defaults to "string", which leaves as_dicts empty and gives us
                    # no article URLs. Tournament rules require a reasoning comment,
                    # so citable URLs matter. "both" keeps as_string as a fallback.
                    return_type="both",
                ),
            )
            results = []
            for article in self._asknews_articles(response):
                # Article fields are article_url / eng_title / summary, not url / title.
                url = getattr(article, "article_url", None) or getattr(article, "url", "")
                title = getattr(article, "eng_title", None) or getattr(article, "title", "") or "No title"
                points = getattr(article, "key_points", None)
                body = " ".join(str(x) for x in points) if isinstance(points, (list, tuple)) and points \
                    else (getattr(article, "summary", "") or "")
                pub = getattr(article, "pub_date", "")
                results.append(f"Title: {title}\nURL: {url}\nPublished: {pub}\nSnippet: {str(body)[:900]}")
            if results:
                return "[AskNews Results]\n" + "\n\n".join(results)
            # No structured items: fall back to the flattened string form rather
            # than reporting a failure on a request that actually succeeded.
            as_string = getattr(response, "as_string", None)
            if as_string:
                return f"[AskNews Results]\n{str(as_string)[:6000]}"
            logger.warning("AskNews returned no articles for query: %s", query[:120])
            return "[AskNews search failed]"
        except Exception as e:
            logger.error(f"AskNews search failed: {type(e).__name__}: {e}")
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

    async def _run_model_synthesis(self, question: MetaculusQuestion, research: str) -> str:
        """Second pass over the retrieved evidence by the `researcher` role.

        Was _run_gptoss_research with the model hardcoded to gpt-oss-120b, which
        404s on this account, so it returned "[GPT-OSS research failed]" on every
        live run. It is now routed through the configured role, and the tag no
        longer names a model it does not use. It is a synthesis step, not a source:
        see _WEB_SOURCE_TAGS.
        """
        try:
            llm = self.get_llm("researcher", "llm")
            prompt = clean_indents(f"""
                You are a research assistant. Research this forecasting question using the Tavily results and your knowledge and provide:
                1. Key factual findings.
                2. Signals supporting YES/higher outcome.
                3. Signals supporting NO/lower outcome.
                Question: {question.question_text}
                Existing research: {research[:2000] if research else 'None'}
            """)
            response = await llm.invoke(prompt)
            return f"[Model Synthesis]\n{response.strip()}"
        except Exception as e:
            logger.error(f"Model synthesis failed: {type(e).__name__}: {e}")
            return "[Model synthesis failed]"

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
        recall = await self._run_model_synthesis(question, web_text)
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

        scale = self.spread_scale
        alpha = 0.0
        reasons: List[str] = []
        if not scale.gate_armed:
            trace.add(
                "Spring AI Confidence Gate",
                f"spread={spread:.4f} recorded but NOT gated: regime is \"{scale.regime}\". "
                f"The 0.20 limit was calibrated against same-model sampling variance and does not "
                f"transfer to cross-model disagreement. Re-arm from a measured distribution.",
            )
        elif scale.gate_limit is not None and spread > scale.gate_limit:
            over = (spread - scale.gate_limit) / scale.gate_limit
            alpha += 0.15 * over
            reasons.append(f"spread {spread:.2f} > {scale.gate_limit:.2f}")
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
        """Agreement in [0,1], used to scale extremize strength.

        The old form was `1.0 - spread/0.30`, with 0.30 calibrated against
        same-model sampling variance. Under cross-model spread it returns 0 on
        almost every question, which drives _extremize_strength to exactly 1.0 and
        turns extremize off across the board with no log line saying so. When the
        scale is uncalibrated this returns neutral agreement instead, and the
        sharpening ceiling is lowered via SpreadScale.extremize_max to compensate.
        """
        if not probs: return 0.0
        scale = self.spread_scale
        denom = scale.agreement_full_disagreement_at
        if denom is None:
            return float(np.clip(scale.agreement_when_uncalibrated, 0.0, 1.0))
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return float(np.clip(1.0 - (spread / denom), 0.0, 1.0))

    def _extremize_strength(self, research: str, probs: List[float], question: MetaculusQuestion) -> float:
        if not self.flags.enable_extremize: return 1.0
        quality = self._research_quality_weight(research)
        agree = self._agreement_strength(probs)
        base = 1.0 + 0.45 * (quality - 0.5) * 2.0 * agree
        close_time = getattr(question, "close_time", None)
        if close_time:
            days = (close_time - datetime.now(timezone.utc)).days
            if days < 60: base = 1.0 + (base - 1.0) * 0.6
        return float(np.clip(base, 0.95, self.spread_scale.extremize_max))

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

    # Percentiles the pipeline always works in.
    STANDARD_PERCENTILES: List[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    @staticmethod
    def _date_bounds_ts(question: DateQuestion) -> Tuple[float, float]:
        """DateQuestion bounds are datetimes. NumericDistribution.from_question
        converts them with .timestamp() and sets is_date=True, so every percentile
        value on this path must be a POSIX timestamp."""
        return float(question.lower_bound.timestamp()), float(question.upper_bound.timestamp())

    @staticmethod
    def _clip_to_date_bounds(pcts: List[Percentile], question: DateQuestion) -> List[Percentile]:
        """Clip only the sides the question declares CLOSED.

        open_upper_bound=True means the event may occur after upper_bound, and
        NumericDistribution represents that with cdf[-1] < 1.0. Clipping such a
        forecast to the bound converts an honest "possibly much later" into a point
        mass sitting on the bound, which is exactly what happened on
        https://www.metaculus.com/questions/43324 in run 30796095656: one pass
        correctly put P90 in 2036 and clipping flattened it onto 2027-05-28.
        """
        lo, hi = Dezzy._date_bounds_ts(question)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return pcts
        low_clip = lo if not question.open_lower_bound else -np.inf
        high_clip = hi if not question.open_upper_bound else np.inf
        for p in pcts:
            p.value = float(np.clip(float(p.value), low_clip, high_clip))
        return Dezzy._enforce_monotone(pcts)

    @staticmethod
    def _min_representable_gap(lo: float, hi: float, cdf_size: int) -> float:
        """One CDF bucket. A distribution with cdf_size buckets over [lo, hi] cannot
        represent mass narrower than this, so percentiles closer together than one
        bucket are not a confident forecast, they are an unrepresentable one."""
        buckets = max(2, int(cdf_size)) - 1
        span = float(hi) - float(lo)
        return (span / buckets) if (np.isfinite(span) and span > 0) else 0.0

    @staticmethod
    def _enforce_min_spacing(pcts: List[Percentile], min_gap: float) -> List[Percentile]:
        """Push adjacent percentiles at least min_gap apart, upward from the lowest.

        _enforce_monotone separates ties by 1e-6, which is a sensible epsilon on a
        unitless numeric axis and meaningless on a timestamp axis where the unit is
        one second: six percentiles a microsecond apart is a point mass. Models do
        collapse onto a single date - two of three passes on question 43324 returned
        P20 through P90 as the identical date.
        """
        if min_gap <= 0 or not pcts:
            return pcts
        out = sorted(pcts, key=lambda x: float(x.percentile))
        for i in range(1, len(out)):
            need = float(out[i - 1].value) + min_gap
            if float(out[i].value) < need:
                out[i].value = need
        return out

    @staticmethod
    def _date_bounds_fallback(question: DateQuestion) -> List[Percentile]:
        lo, hi = Dezzy._date_bounds_ts(question)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            now = datetime.now(timezone.utc).timestamp()
            lo, hi = now, now + 365 * 86400.0
        w = {0.1: 0.05, 0.2: 0.15, 0.4: 0.40, 0.6: 0.60, 0.8: 0.85, 0.9: 0.95}
        return Dezzy._enforce_monotone(
            [Percentile(percentile=q, value=lo + (hi - lo) * w[q]) for q in Dezzy.STANDARD_PERCENTILES]
        )

    _DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")

    @staticmethod
    def _parse_one_date(text: str) -> Optional[float]:
        """Parse a single date to a POSIX timestamp, or None."""
        raw = (text or "").strip().strip(".,;")
        if not raw:
            return None
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        candidates = [m.group(0)] if m else []
        candidates.append(raw)
        for cand in candidates:
            for fmt in Dezzy._DATE_FORMATS:
                try:
                    return datetime.strptime(cand, fmt).replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    continue
        return None

    @staticmethod
    def _format_date_pcts(pcts: List[Percentile]) -> str:
        out = []
        for p in sorted(pcts, key=lambda x: float(x.percentile)):
            try:
                shown = datetime.fromtimestamp(float(p.value), tz=timezone.utc).strftime("%Y-%m-%d")
            except (OverflowError, OSError, ValueError):
                shown = f"{float(p.value):.0f}"
            out.append(f"P{int(round(float(p.percentile) * 100))}={shown}")
        return " | ".join(out)

    async def _parse_date_percentiles(self, question: DateQuestion, text: str, stage: str) -> List[Percentile]:
        """Pull six `Percentile NN: <date>` lines out of the model output.

        Deliberately regex-first rather than going straight to the parser LLM: the
        expected shape is fixed, and a local parse is cheaper and cannot hallucinate
        a date the forecaster did not write.
        """
        def scan(src: str) -> List[Percentile]:
            found: Dict[float, float] = {}
            for line in (src or "").splitlines():
                m = re.match(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
                if not m:
                    continue
                ts = self._parse_one_date(m.group(2))
                if ts is not None:
                    found[round(int(m.group(1)) / 100.0, 3)] = ts
            if all(round(q, 3) in found for q in self.STANDARD_PERCENTILES):
                return self._enforce_monotone(
                    [Percentile(percentile=q, value=found[round(q, 3)]) for q in self.STANDARD_PERCENTILES]
                )
            return []

        got = scan(text) or scan(self._extract_percentile_block(text) or "")
        if got:
            return got
        try:
            parser_llm = self.get_llm("parser", "llm")
            reformatted = await parser_llm.invoke(
                "Rewrite the following into EXACTLY these 6 lines and nothing else, each date as YYYY-MM-DD:\n"
                "Percentile 10: YYYY-MM-DD\nPercentile 20: YYYY-MM-DD\nPercentile 40: YYYY-MM-DD\n"
                "Percentile 60: YYYY-MM-DD\nPercentile 80: YYYY-MM-DD\nPercentile 90: YYYY-MM-DD\n\n"
                f"Text:\n{text}"
            )
            got = scan(reformatted)
            if got:
                return got
        except Exception as e:
            logger.warning(f"date percentile reformat failed at {stage}: {type(e).__name__}: {e}")
        logger.warning(f"date percentile parse failed at {stage}; using bounds fallback")
        return self._date_bounds_fallback(question)

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
        # Only clip the closed sides. Same defect as the date path: clipping a
        # forecast to a bound the question declares open destroys the tail mass that
        # NumericDistribution represents with cdf[-1] < 1.0.
        low_clip = lo_f if not getattr(question, "open_lower_bound", False) else -np.inf
        high_clip = hi_f if not getattr(question, "open_upper_bound", False) else np.inf
        for p in pcts:
            p.value = float(np.clip(float(p.value), low_clip, high_clip))
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
        model = self.FORECASTER_MODELS[(max(1, run_index) - 1) % len(self.FORECASTER_MODELS)]
        # Logged, not traced: add_narrative deliberately anonymises model identity in
        # the published rationale, and that intent is preserved here. #calibration
        # records model_set on its own rows.
        logger.info(f"[{self.bot_name}] forecast pass {run_index} model={model}")
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

        if isinstance(question, DateQuestion):
            lo_dt = question.lower_bound.strftime("%Y-%m-%d")
            hi_dt = question.upper_bound.strftime("%Y-%m-%d")
            raw = await llm.invoke(clean_indents(f"""
                You are a calibrated superforecaster estimating WHEN an event occurs.
                {self._grounding_instructions()}
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria or "Not stated"}
                The answer is a DATE. The question's stated range is {lo_dt} to {hi_dt}.
                {"The UPPER bound is OPEN: the event may occur later than " + hi_dt + ", and if you believe that is likely you MUST put your upper percentiles beyond it. Do not compress them onto " + hi_dt + "." if question.open_upper_bound else "The event cannot occur after " + hi_dt + "; do not give a date beyond it."}
                {"The LOWER bound is OPEN: the event may already have occurred before " + lo_dt + "." if question.open_lower_bound else "The event cannot occur before " + lo_dt + "; do not give a date before it."}
                Your six dates must be strictly increasing and genuinely spread out.
                Do not repeat the same date across percentiles: if you are unsure, widen.
                Context:
                {context}
                Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
                Reason about the mechanism and timing first: what has to happen, how long each
                step historically takes, and what would delay it. Keep the interval wide unless
                the evidence is strong; late resolution is more common than early.
                The LAST thing you write is EXACTLY these 6 lines, each date as YYYY-MM-DD,
                in increasing order:
                Percentile 10: YYYY-MM-DD
                Percentile 20: YYYY-MM-DD
                Percentile 40: YYYY-MM-DD
                Percentile 60: YYYY-MM-DD
                Percentile 80: YYYY-MM-DD
                Percentile 90: YYYY-MM-DD
            """))
            narrative_lines = []
            for line in (raw or "").splitlines():
                if re.match(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, re.IGNORECASE): break
                narrative_lines.append(line)
            trace.add_narrative(run_index, "\n".join(narrative_lines).strip())
            return await self._parse_date_percentiles(question, raw, stage=f"run{run_index}")

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

        scale = self.spread_scale
        if scale.heavy_shrink_at is not None and spread >= scale.heavy_shrink_at:
            shrink = 0.28
        else:
            # Uncalibrated regime: spread cannot select a shrink rung, so fall back to
            # the research-quality ladder, which is calibrated independently of spread.
            shrink = 0.22 if quality < 0.70 else 0.12
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
                # A strength of 1.0 makes extremize_logit an identity function. That
                # used to happen silently whenever _agreement_strength returned 0,
                # so the config claimed extremize was on while it did nothing.
                if abs(ext_strength - 1.0) < 1e-3:
                    trace.add(
                        "Extremize",
                        f"resolved to strength {ext_strength:.4f} - identity, no sharpening applied "
                        f"(agreement={self._agreement_strength(probs):.3f}, "
                        f"quality={self._research_quality_weight(research):.2f}, "
                        f"regime=\"{self.spread_scale.regime}\")",
                    )
                    logger.warning(
                        f"[{self.bot_name}] extremize resolved to identity (strength={ext_strength:.4f}) "
                        f"under regime: {self.spread_scale.regime}"
                    )
            else:
                p_ext = combined
                applied.append("extremize(gated-off)")
                trace.add("Extremize", f"SKIPPED - p={combined:.4f} outside the (0.02, 0.98) gate or exactly 0.5.")
        else:
            p_ext = combined
            trace.add("Extremize", "SKIPPED - disabled by --no-extremize.")

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

    async def _run_forecast_on_date(self, question: DateQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        """Forecast a date question as a distribution of POSIX timestamps.

        The library dispatches DateQuestion here (forecast_bot.py:511) and expects
        a NumericDistribution of timestamps. Dezzy never implemented it, so every
        date question raised NotImplementedError. After the confidence-gate change
        that failure no longer stops the run, which made it a silent loss of one
        forecast per date question.

        Deliberately NOT reusing the numeric regime detection: PARTIAL_REVEAL_SUM
        and STRUCTURED_TS both key off numeric magnitudes and a bounded multiplier,
        neither of which means anything on a timestamp axis.
        """
        trace = ReasoningTrace(question.question_text, self.bot_name)
        _web, no_evidence = self._trace_research_footprint(trace, research)
        quality = self._research_quality_weight(research)

        lo_ts, hi_ts = self._date_bounds_ts(question)
        trace.add(
            "Question bounds",
            f"{question.lower_bound.strftime('%Y-%m-%d')} to {question.upper_bound.strftime('%Y-%m-%d')} "
            f"| open_lower={question.open_lower_bound} open_upper={question.open_upper_bound}",
        )

        premortem = await self._run_premortem_analysis(question, research)
        trace.add("Premortem", premortem[:1200])
        grounded_context = self._build_grounded_context(question, research, premortem)

        runs = await self._multi_run(question, research, trace, grounded_context=grounded_context)
        runs = [r for r in runs if r]
        if not runs:
            pcts = self._date_bounds_fallback(question)
            trace.add("Fallback prediction", "All independent date runs failed; returning a wide in-bounds distribution.")
            trace.add("★ FINAL PREDICTION", self._format_date_pcts(pcts))
            return ReasonedPrediction(
                prediction_value=NumericDistribution.from_question(pcts, question), reasoning=trace.render()
            )

        for i, r in enumerate(runs, 1):
            trace.add(f"Run {i} percentiles", self._format_date_pcts(r))

        # Median per percentile across runs.
        by_q: Dict[float, List[float]] = {round(q, 3): [] for q in self.STANDARD_PERCENTILES}
        for r in runs:
            for pc in r:
                key = round(float(pc.percentile), 3)
                if key in by_q:
                    by_q[key].append(float(pc.value))
        agg = [
            Percentile(percentile=q, value=float(np.median(by_q[round(q, 3)])))
            for q in self.STANDARD_PERCENTILES
            if by_q[round(q, 3)]
        ]
        if len(agg) < len(self.STANDARD_PERCENTILES):
            agg = self._date_bounds_fallback(question)
        agg = self._enforce_monotone(agg)
        trace.add(f"Aggregated across {len(runs)} run(s)", self._format_date_pcts(agg))

        min_gap = self._min_representable_gap(lo_ts, hi_ts, question.cdf_size)
        spaced = self._enforce_min_spacing([Percentile(percentile=p.percentile, value=p.value) for p in agg], min_gap)
        if self._format_date_pcts(spaced) != self._format_date_pcts(agg):
            trace.add(
                "Minimum spacing",
                f"adjacent percentiles were closer than one CDF bucket "
                f"({min_gap / 86400.0:.1f} days over a {question.cdf_size}-bucket grid); "
                f"spread to the representable minimum -> {self._format_date_pcts(spaced)}",
            )
        agg = spaced

        if no_evidence:
            agg = self._clip_to_date_bounds(self._widen_percentiles(agg, self.NO_EVIDENCE_WIDEN), question)
            trace.add(
                "No-evidence handling",
                f"research_footprint=none; interval widened x{self.NO_EVIDENCE_WIDEN:.2f} about the median "
                f"-> {self._format_date_pcts(agg)}",
            )

        # Spread is RECORDED, never gated, on this path. The P10-P90 span of a date
        # distribution is a duration in seconds; it is not a probability spread and
        # not the numeric relative width either, so no threshold from either regime
        # transfers to it. See SpreadScale.
        p10, p90 = self._p10_p90(agg)
        span_days = ((p90 - p10) / 86400.0) if (p10 is not None and p90 is not None) else 0.0
        bounds_days = ((hi_ts - lo_ts) / 86400.0) if np.isfinite(hi_ts - lo_ts) else 0.0
        frac = (span_days / bounds_days) if bounds_days > 0 else 0.0
        trace.add(
            "Spread (recorded, not gated)",
            f"P10-P90 span={span_days:.1f} days over a {bounds_days:.1f} day window "
            f"({frac:.1%} of the range); quality={quality:.2f}. A date span is a duration, "
            f"not a probability spread, so it is recorded only.",
        )

        agg = self._clip_to_date_bounds(agg, question)
        trace.add("★ FINAL PREDICTION", self._format_date_pcts(agg))
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(agg, question), reasoning=trace.render()
        )

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
    parser.add_argument("--model-check", action="store_true",
                        help="Send a one-token prompt to every configured model and exit nonzero if any is unreachable. No Metaculus calls.")
    parser.add_argument("--research-check", action="store_true",
                        help="Execute the web search clients only, print what each returned, and exit nonzero if a configured client returns no evidence. No Metaculus or LLM calls.")
    parser.add_argument("--research-check-query", type=str,
                        default="US Federal Reserve interest rate decision September 2026",
                        help="Query used by --research-check.")

    args = parser.parse_args()

    if args.model_check:
        # Sends a minimal prompt to every model this bot is configured to use, so a
        # bad or retired slug surfaces here rather than as a silently swallowed
        # exception inside a try/except during a live run.
        probe = Dezzy.__new__(Dezzy)
        Dezzy.__init__(
            probe,
            research_reports_per_question=1, predictions_per_research_report=1,
            publish_reports_to_metaculus=False, bot_name="model-check",
        )
        targets: List[Tuple[str, str]] = [
            (f"forecaster_{i + 1}", m) for i, m in enumerate(probe.FORECASTER_MODELS)
        ] + sorted(probe._llm_config_defaults().items()) + [
            (f"candidate", m) for m in probe.FORECASTER_CANDIDATES
        ]

        async def _ping() -> int:
            failures: List[str] = []
            seen: Dict[str, str] = {}
            print("\n=== model check ===\n")
            for role, model in targets:
                if model in seen:
                    print(f"  {role:22} {model:46} reuses {seen[model]}")
                    continue
                try:
                    out = await GeneralLlm(model=model, temperature=0.0).invoke(
                        "Reply with the single word: OK"
                    )
                    text = (out or "").strip().replace("\n", " ")[:60]
                    print(f"  {role:22} {model:46} OK    <- {text!r}")
                    seen[model] = role
                except Exception as e:
                    print(f"  {role:22} {model:46} FAIL  {type(e).__name__}: {str(e)[:180]}")
                    if role != "candidate":
                        failures.append(f"{role}={model}")
            if failures:
                print(f"\nGATE FAIL: {len(failures)} model(s) unreachable: {failures}")
                return 1
            print(f"\nGATE PASS: {len(seen)} distinct model(s) reachable.")
            return 0

        raise SystemExit(asyncio.run(_ping()))

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

    # Test runs must not publish. Previously test_questions shared the tournament
    # config and would have posted forecasts to whatever it was pointed at.
    publish = args.mode != "test_questions"
    bot = Dezzy(
        research_reports_per_question=1, predictions_per_research_report=1,
        publish_reports_to_metaculus=publish, skip_previously_forecasted_questions=False,
        bot_name=args.bot_name, flags=flags, runs_per_question=max(1, int(args.runs)),
    )
    logger.info(
        f"[{args.bot_name}] mode={args.mode} publish={publish} runs={bot.runs_per_question} "
        f"spread_regime=\"{bot.spread_scale.regime}\" gate_armed={bot.spread_scale.gate_armed}"
    )
    logger.info(
        f"[{args.bot_name}] forecast models: {bot.FORECASTER_MODELS[: bot.runs_per_question]}"
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

        # test_questions. PR #1 ("minibench removed") rewrote
        # default_tournament_ids() but left this fallback hardcoded to
        # market-pulse-26q2, which closed at the end of June 2026 and returns 0
        # questions. That is why "Test Bot" has been a silent no-op ever since.
        # bot-testing-area is the target the official template uses for smoke
        # tests and contains every question type.
        bot.skip_previously_forecasted_questions = False
        bot.set_active_tournament(Dezzy.TEST_TOURNAMENT_ID)
        return await bot.forecast_on_tournament(Dezzy.TEST_TOURNAMENT_ID, return_exceptions=True)

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
