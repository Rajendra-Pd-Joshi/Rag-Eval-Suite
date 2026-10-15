"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from model.nemotron_model import NemotronLLM
# from langchain_openai import ChatOpenAI
from src.generator import generate   # your generator: generate(query, context) -> answer
from evals.harness import load_goldens, summarize_by_metric, print_summary
from deepeval.evaluate.configs import AsyncConfig 

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = NemotronLLM()  # judge model for faithfulness and relevancy
THRESHOLD = 0.7


import json, os

CACHE = "goldens/generated_answers.json"


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH, required=("query", "ideal_context"))

    # 2. LOAD any answers generated on earlier runs
    answers = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            answers = json.load(f)

    # 3. GENERATE only the answers we don't have yet (on the GOLDEN context, for isolation)
    for g in goldens:
        if g["query"] in answers:
            continue
        try:
            answers[g["query"]] = generate(g["query"], g["ideal_context"])
        except Exception as e:
            print(f"generate failed for: {g['query']}\n  {e}")
            continue

        # save after every answer so a crash later doesn't lose the earlier ones
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)

    # 4. BUILD the test cases, skipping any question that failed to generate
    test_cases = [
        LLMTestCase(
            input=g["query"],
            actual_output=answers[g["query"]],
            retrieval_context=g["ideal_context"],
        )
        for g in goldens
        if g["query"] in answers
    ]

    # 5. THE METRICS
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    ]

    # 6. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics,async_config=AsyncConfig(run_async=False))
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())