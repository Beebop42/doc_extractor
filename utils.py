import json
import logging
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
def setup_logger(log_file: str = "pipeline.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

log = setup_logger()

# ── OpenRouter client ──────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

client = get_client()

# ── LLM call ───────────────────────────────────────────────────────────────
def call_llm(prompt: str, 
             image_data: str, 
             model: str = "google/gemini-2.0-flash-001", 
             model_temperature:float = 1.0, 
             mime_type: str = "image/png") -> tuple[str, dict]:
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                },
                {"type": "text", "text": prompt}
            ]
        }],
        temperature=model_temperature
    )

    elapsed = time.perf_counter() - start
    usage = response.usage
    choice0 = response.choices[0]

    # OpenRouter/SDKs do not always provide `finish_reason` on the choice object.
    # Keep this key stable for downstream code.
    finish_reason = getattr(choice0, "finish_reason", None)
    if finish_reason is None:
        try:
            finish_reason = choice0.get("finish_reason")  # type: ignore[attr-defined]
        except Exception:
            finish_reason = None
    if not finish_reason:
        finish_reason = "stop"

    stats = {
        "latency_ms":        round(elapsed * 1000),
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens":      usage.total_tokens,
        "finish_reason":     finish_reason
    }

    return choice0.message.content.strip(), stats


# ── Parse JSON from LLM response ───────────────────────────────────────────
def parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    print(raw)
    return json.loads(raw.strip())