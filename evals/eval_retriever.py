# evals/eval_retriever.py
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig   # add this
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary
from model.nemotron_model import NemotronLLM

load_dotenv()   # needs NVIDIA_API_KEY in your .env

# --- the two golden sets you want to compare ---
GOLDEN_SETS = {
    "llm_drafted_human_reviewed": "goldens/retriever_goldens.json"
    # "deepeval_synthesizer":       "goldens/retriever_deepeval_goldens.json",
}

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 20
TOP_K = 5

JUDGE = NemotronLLM()          # judge model (replaces gpt-4o-mini)
THRESHOLD = 0.7


def run(retriever, golden_path):
    # 1. LOAD the golden set --- the fixed, human-reviewed truth
    goldens = load_goldens(golden_path)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    #    async_mode=False -> one call at a time, friendlier to API rate limits.
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=JUDGE, include_reason=True, async_mode=False
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=JUDGE, include_reason=True, async_mode=False
        ),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),   # add this: sequential test cases  
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": EMBED_MODEL,
            "reranker_model": RERANK_MODEL,
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "fetch_k": getattr(retriever, "fetch_k", FETCH_K),
            "top_k": getattr(retriever, "top_k", TOP_K),
            "judge_model": JUDGE.get_model_name(),
            "golden_set": golden_path,
        },
        
    )
    return summarize_by_metric(result)


def run_local(golden_path=GOLDEN_SETS["llm_drafted_human_reviewed"]):
    """Standalone convenience: build the retriever, then run on one golden set."""
    return run(RerankingRetriever(fetch_k=FETCH_K, top_k=TOP_K), golden_path)


if __name__ == "__main__":
    retriever = RerankingRetriever(fetch_k=FETCH_K, top_k=TOP_K)   # build once, reuse

    for name, path in GOLDEN_SETS.items():
        print(f"\n===== golden set: {name} ({path}) =====")
        print_summary(f"retriever [{name}]", run(retriever, path))