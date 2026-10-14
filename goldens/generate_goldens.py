import os
import re
import glob
import json
import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()   # needs NVIDIA_API_KEY in your .env


# --- wrapper so DeepEval can use ChatOpenAI (pointed at NVIDIA's endpoint) ---
class NemotronLLM(DeepEvalBaseLLM):

    def __init__(self, model="nvidia/nemotron-3-ultra-550b-a55b"):
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
            temperature=0.2,
        )

    def load_model(self):
        return self.llm

    # ---- helpers ----
    def _with_schema(self, prompt, schema):
        return prompt + (
            "\n\nRespond with ONLY a valid JSON object matching this schema, "
            "no extra text:\n" + json.dumps(schema.model_json_schema())
        )

    def _parse(self, text, schema):
        # drop <think>...</think> reasoning blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # drop markdown JSON fences
        text = re.sub(r"```(?:json)?", "", text).strip()

        # extract the JSON object
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Model did not return valid JSON:\n{text}")

        return schema.model_validate_json(text[start:end + 1])

    # ---- sync ----
    def generate(self, prompt: str, schema=None):
        if not schema:
            return self.llm.invoke(prompt).content

        prompt = self._with_schema(prompt, schema)
        for attempt in range(2):                      # one retry on bad JSON
            try:
                return self._parse(self.llm.invoke(prompt).content, schema)
            except Exception:
                if attempt == 1:
                    raise

    # ---- async ----
    async def a_generate(self, prompt: str, schema=None):
        if not schema:
            return (await self.llm.ainvoke(prompt)).content

        prompt = self._with_schema(prompt, schema)
        for attempt in range(2):
            try:
                return self._parse((await self.llm.ainvoke(prompt)).content, schema)
            except Exception:
                if attempt == 1:
                    raise

    def get_model_name(self):
        return self.model_name


# --- reuse your VTT cleaning + chunking ---
def load_chunks():
    texts = []

    for path in glob.glob("data/*.vtt"):
        with open(path, encoding="utf-8") as f:
            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return splitter.split_text("\n\n".join(texts))


# --- generate ---
random.seed(42)

chunks = load_chunks()

# skip very short chunks -- they tend to produce vague questions
chunks = [c for c in chunks if len(c) >= 300]

sample = random.sample(chunks, min(15, len(chunks)))

contexts = [[c] for c in sample]


synthesizer = Synthesizer(
    model=NemotronLLM(),
    async_mode=False,
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


# --- convert to your schema ---
rows = []

for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",
    })


os.makedirs("goldens", exist_ok=True)

out = "goldens/retriever_deepeval_goldens.json"

with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)


print(f"wrote {len(rows)} DRAFT goldens -> {out}")
print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)