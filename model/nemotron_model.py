# model/nemotron_model.py
import os
import re
import json

from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM


class NemotronLLM(DeepEvalBaseLLM):

    def __init__(self, model="nvidia/nemotron-3-ultra-550b-a55b"):
        self.model_name = model
        self.llm = ChatOpenAI(
            model=model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
            temperature=0
        )

    def load_model(self):
        return self.llm

    def _with_schema(self, prompt, schema):
        return prompt + (
            "\n\nRespond with ONLY a valid JSON object matching this schema, "
            "no extra text:\n" + json.dumps(schema.model_json_schema())
        )

    def _parse(self, text, schema):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Model did not return valid JSON:\n{text}")
        return schema.model_validate_json(text[start:end + 1])

    def generate(self, prompt: str, schema=None):
        if not schema:
            return self.llm.invoke(prompt).content
        prompt = self._with_schema(prompt, schema)
        for attempt in range(2):
            try:
                return self._parse(self.llm.invoke(prompt).content, schema)
            except ValueError:          # bad JSON only; rate-limit errors are not retried here
                if attempt == 1:
                    raise

    async def a_generate(self, prompt: str, schema=None):
        if not schema:
            return (await self.llm.ainvoke(prompt)).content
        prompt = self._with_schema(prompt, schema)
        for attempt in range(2):
            try:
                return self._parse((await self.llm.ainvoke(prompt)).content, schema)
            except ValueError:
                if attempt == 1:
                    raise

    def get_model_name(self):
        return self.model_name