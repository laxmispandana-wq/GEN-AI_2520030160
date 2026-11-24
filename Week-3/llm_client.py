import os
import time
import requests

from dataclasses import dataclass
from google import genai
from google.genai import types


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    provider: str


class LLMClient:

    def __init__(self, provider, usage_logger):
        self.provider = provider
        self.usage_logger = usage_logger

        if provider == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

        elif provider == "ollama":
            self.client = None

        else:
            raise ValueError("Unsupported provider")

    def generate(self, prompt):

        start_time = time.time()

        for attempt in range(3):

            try:

                if self.provider == "gemini":

                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            http_options=types.HttpOptions(
                                timeout=60000
                            )
                        )
                    )

                    text = response.text or ""

                    input_tokens = (
                        response.usage_metadata.prompt_token_count
                        if response.usage_metadata else 0
                    )

                    output_tokens = (
                        response.usage_metadata.candidates_token_count
                        if response.usage_metadata else 0
                    )

                    input_price = float(
                        os.getenv(
                            "GEMINI_INPUT_PRICE_PER_MILLION", "0"
                        )
                    )

                    output_price = float(
                        os.getenv(
                            "GEMINI_OUTPUT_PRICE_PER_MILLION", "0"
                        )
                    )

                    cost = (
                        input_tokens * input_price
                        + output_tokens * output_price
                    ) / 1_000_000

                else:

                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2",
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=(5, 60)
                    )

                    response.raise_for_status()
                    data = response.json()

                    text = data.get("response", "")
                    input_tokens = data.get("prompt_eval_count", 0)
                    output_tokens = data.get("eval_count", 0)
                    cost = 0.0

                latency_ms = (time.time() - start_time) * 1000

                result = LLMResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost=cost,
                    provider=self.provider
                )

                self.usage_logger.log_call({
                    "provider": self.provider,
                    "prompt": prompt,
                    "response": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "cost": cost,
                    "status": "success"
                })

                return result

            except Exception as error:

                if attempt < 2:
                    time.sleep(2 ** attempt)

                else:

                    self.usage_logger.log_call({
                        "provider": self.provider,
                        "prompt": prompt,
                        "error": str(error),
                        "status": "failed"
                    })

                    raise