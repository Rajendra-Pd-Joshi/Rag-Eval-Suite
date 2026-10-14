import time
import numpy as np
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

QUERY = "When is the beta shipping?"
PASSAGES = [
    "We agreed to ship the beta on Friday after QA signs off.",   # relevant
    "The weather was nice during lunch today.",                   # irrelevant
    "Marketing wants the launch blog post ready by next week.",   # loosely related
    "Let's move the retro to Thursday afternoon.",                # irrelevant
]


def test_embeddings():
    emb = NVIDIAEmbeddings(model=EMBED_MODEL)
    q = np.array(emb.embed_query(QUERY))
    docs = np.array(emb.embed_documents(PASSAGES))
    sims = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))
    print(f"\n[embeddings] dim={len(q)}")
    for s, p in sorted(zip(sims, PASSAGES), reverse=True):
        print(f"  {s:.3f}  {p}")
    assert PASSAGES[int(np.argmax(sims))] == PASSAGES[0]


def test_rerankers():
    for name in RERANKERS:
        t = time.time()
        ce = CrossEncoder(name)
        scores = ce.predict([(QUERY, p) for p in PASSAGES])
        print(f"\n[rerank] {name}  ({time.time() - t:.1f}s incl. load)")
        for s, p in sorted(zip(scores, PASSAGES), reverse=True):
            print(f"  {s:8.3f}  {p}")
        assert int(np.argmax(scores)) == 0, f"{name} ranked the wrong passage first"


if __name__ == "__main__":
    test_embeddings()
    test_rerankers()
    print("\nAll good.")