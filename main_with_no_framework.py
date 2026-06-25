# main.py
# Hybrid Pre-Mortem Forecasting Bot — Tournament-Only, Validation-Compliant

import argparse
import asyncio
import json
import logging
import os
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    ReasonedPrediction,
    clean_indents,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
)

from tavily import TavilyClient
from newsapi import NewsApiClient

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HybridPreMortemBot")

# -----------------------------
# Suppress noisy warnings
# -----------------------------
warnings.filterwarnings("ignore", message=".*does not support cost tracking.*")


class HybridPreMortemBot(ForecastBot):
    """
    Tournament-only bot with strict compliance to forecasting_tools schema.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/openai/gpt-4o-search-preview",
            "online_researcher": "openrouter/perplexity/sonar",
            "research_synthesizer": "openrouter/openai/gpt-4o",
            "pre_mortem_agent": "openrouter/openai/gpt-5",
            "pre_parade_agent": "openrouter/openai/gpt-4.1",
            "advocate_agent": "openrouter/openai/gpt-4.1-nano",
            "risk_synthesizer": "openrouter/openai/gpt-5",
            "synthesizer_1": "openrouter/openai/o3",
            "synthesizer_2": "openrouter/openai/gpt-5",
            "synthesizer_3": "openrouter/anthropic/claude-sonnet-4",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.serpapi_key = SERPAPI_API_KEY
        logger.info("Initialized HybridPreMortemBot in tournament-only mode.")

    def _get_synth_models_for_question(self, question: MetaculusQuestion) -> List[str]:
        text = question.question_text.lower()
        if "extinction" in text or "population" in text:
            return ["synthesizer_2", "synthesizer_1"]
        elif any(kw in text for kw in ["yield", "spread", "stock", "nvidia", "apple", "nasdaq", "s&p", "futures", "oil"]):
            return ["synthesizer_3", "synthesizer_2"]
        else:
            return ["synthesizer_1", "synthesizer_2", "synthesizer_3"]

    async def _run_premortem_analysis(self, question: MetaculusQuestion, research: str) -> str:
        prompt = clean_indents(f"""
        Imagine the correct answer is the opposite of your best guess.
        List 3 plausible, evidence-based reasons why you could be wrong.

        Question: {question.question_text}
        Research: {research[:1000]}...
        """)
        return await self.get_llm("pre_mortem_agent", "llm").invoke(prompt)

    # -----------------------------
    # External Research
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key: return "Tavily search not performed."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return "NewsAPI search not performed."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI search failed: {e}"

    def call_serpapi(self, query: str) -> str:
        if not self.serpapi_key: return "SerpApi search not performed."
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": self.serpapi_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            snippets = [result.get('snippet', '') for result in data.get('organic_results', [])]
            return "\n".join([f"- {s}" for s in snippets if s])
        except Exception as e: return f"SerpApi search failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            logger.info(f"--- Starting All-Source Research for: {question.question_text} ---")
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text),
                "serpapi": loop.run_in_executor(None, self.call_serpapi, question.question_text),
                "perplexity": self.get_llm("online_researcher", "llm").invoke(question.question_text)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            source_names = list(tasks.keys())
            raw_research_dump = ""
            for i, result in enumerate(results):
                raw_research_dump += f"--- RAW DATA FROM: {source_names[i].upper()} ---\n{result}\n\n"
            synthesis_prompt = clean_indents(f"""
            Synthesize into a clean intelligence briefing.

            Raw Data Dump:
            {raw_research_dump}

            Synthesized Intelligence Briefing:
            """)
            return await self.get_llm("research_synthesizer", "llm").invoke(synthesis_prompt)

    # -----------------------------
    # Forecasting Pipelines — STRICTLY COMPLIANT
    # -----------------------------
    async def _single_binary_forecast(self, synthesizer_key: str, question: BinaryQuestion, context: str) -> Dict[str, Any]:
        prompt = clean_indents(f"""
        Estimate probability (0.0–1.0) that the event occurs.

        Question: {question.question_text}
        Context: {context}

        Respond ONLY with JSON: {{"probability": 0.75, "reasoning": "..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            prob = max(0.0, min(1.0, float(parsed["probability"])))
            return {"probability": prob, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"Binary parsing failed: {e}. Fallback to 0.5.")
            return {"probability": 0.5, "reasoning": "Fallback"}

    async def _single_mc_forecast(self, synthesizer_key: str, question: MultipleChoiceQuestion, context: str) -> Dict[str, Any]:
        options_str = "\n".join([f"- {opt}" for opt in question.options])
        prompt = clean_indents(f"""
        Assign probabilities to each option (sum to 1.0).

        Question: {question.question_text}
        Options:
        {options_str}
        Context: {context}

        Respond ONLY with JSON: {{"probabilities": [0.2, 0.5, 0.3], "reasoning": "..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            probs = [float(p) for p in parsed["probabilities"]]
            total = sum(probs)
            if total == 0:
                probs = [1.0 / len(probs)] * len(probs)
            else:
                probs = [p / total for p in probs]
            return {"probabilities": probs, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"MC parsing failed: {e}. Uniform fallback.")
            uniform = [1.0 / len(question.options)] * len(question.options)
            return {"probabilities": uniform, "reasoning": "Fallback"}

    async def _single_numeric_forecast(self, synthesizer_key: str, question: NumericQuestion, context: str) -> Dict[str, Any]:
        unit_info = getattr(question, 'unit', getattr(question, 'units', 'Not specified'))
        prompt = clean_indents(f"""
        Provide 10th, 50th, 90th percentiles.

        Question: {question.question_text}
        Unit/format: {unit_info}
        Context: {context}

        Respond ONLY with JSON: {{"percentiles": [{{"percentile": 10, "value": -2.1}}, ...], "reasoning": "..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            percentiles = []
            for p in parsed["percentiles"]:
                percentiles.append({"percentile": int(p["percentile"]), "value": float(p["value"])})
            return {"percentiles": percentiles, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"Numeric parsing failed: {e}. Fallback.")
            if "exceed" in question.question_text.lower():
                return {"percentiles": [{"percentile":10,"value":-5},{"percentile":50,"value":0},{"percentile":90,"value":5}], "reasoning": "Fallback"}
            else:
                return {"percentiles": [{"percentile":10,"value":0.1},{"percentile":50,"value":0.5},{"percentile":90,"value":0.9}], "reasoning": "Fallback"}

    async def _forecast_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction:
        premortem = await self._run_premortem_analysis(question, research)
        context = f"RESEARCH:\n{research}\n\nPRE-MORTEM:\n{premortem}"
        synthesizers = self._get_synth_models_for_question(question)
        forecasts = await asyncio.gather(*[self._single_binary_forecast(k, question, context) for k in synthesizers])
        median_prob = float(np.median([f["probability"] for f in forecasts]))
        reasoning = " | ".join([f["reasoning"] for f in forecasts])
        # ✅ CORRECT FIELD NAME: prediction_in_decimal
        forecast = BinaryPrediction(prediction_in_decimal=median_prob)
        return ReasonedPrediction(forecast=forecast, reasoning=reasoning)

    async def _forecast_mc(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction:
        premortem = await self._run_premortem_analysis(question, research)
        context = f"RESEARCH:\n{research}\n\nPRE-MORTEM:\n{premortem}"
        synthesizers = self._get_synth_models_for_question(question)
        forecasts = await asyncio.gather(*[self._single_mc_forecast(k, question, context) for k in synthesizers])
        all_probs = np.array([f["probabilities"] for f in forecasts])
        median_probs = np.median(all_probs, axis=0)
        if median_probs.sum() > 0:
            median_probs = median_probs / median_probs.sum()
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        forecast_list = [{"option": opt, "probability": float(p)} for opt, p in zip(question.options, median_probs)]
        forecast = PredictedOptionList(forecast_list)
        reasoning = " | ".join([f["reasoning"] for f in forecasts])
        return ReasonedPrediction(forecast=forecast, reasoning=reasoning)

    async def _forecast_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction:
        premortem = await self._run_premortem_analysis(question, research)
        context = f"RESEARCH:\n{research}\n\nPRE-MORTEM:\n{premortem}"
        synthesizers = self._get_synth_models_for_question(question)
        forecasts = await asyncio.gather(*[self._single_numeric_forecast(k, question, context) for k in synthesizers])
        target_percentiles = [10, 50, 90]
        aggregated_percentiles = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f["percentiles"]:
                    if item["percentile"] == p:
                        values.append(item["value"])
                        break
                else:
                    values.append(0.0)
            median_val = float(np.median(values))
            aggregated_percentiles.append(Percentile(percentile=p / 100.0, value=median_val))
        
        # ✅ Infer bounds from question semantics
        question_text = question.question_text.lower()
        if "yield" in question_text or "spread" in question_text:
            open_lower_bound = False
            lower_bound: Optional[float] = 0.0
            open_upper_bound = True
            upper_bound: Optional[float] = None
        elif "exceed" in question_text and ("return" in question_text or "stock" in question_text):
            open_lower_bound = True
            open_upper_bound = True
            lower_bound = None
            upper_bound = None
        else:
            open_lower_bound = True
            open_upper_bound = True
            lower_bound = None
            upper_bound = None

        # ✅ CORRECT CONSTRUCTION: ALL 5 FIELDS, CORRECT NAMES
        distribution = NumericDistribution(
            declared_percentiles=aggregated_percentiles,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        reasoning = " | ".join([f["reasoning"] for f in forecasts])
        return ReasonedPrediction(forecast=distribution, reasoning=reasoning)

    # -----------------------------
    # Abstract method implementations
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_binary(question, research)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_mc(question, research)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_numeric(question, research)


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Pre-Mortem Bot on tournaments.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["33022", "market-pulse-26q2", MetaculusApi.CURRENT_MINIBENCH_ID],
        help="Tournament IDs to forecast on",
    )
    args = parser.parse_args()

    bot = HybridPreMortemBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        logger.info("Starting tournament-mode forecasting...")
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Critical error during run: {e}", exc_info=True)
