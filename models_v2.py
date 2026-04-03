"""
models_v2.py — LLM, embedding, and vision model functions using Ollama.
Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
No API keys or internet connection required after models are pulled.
"""

import asyncio
import logging
import numpy as np
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

from config_v2 import (
    LLM_MODEL_V2,
    VISION_MODEL_V2,
    EMBEDDING_MODEL_V2,
    EMBEDDING_DIM_V2,
    DOMAIN_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Limit concurrent LLM calls to 2 so qwen2.5:7b doesn't time out under
# parallel load. LightRAG default is 4, which overwhelms a single GPU model.
_llm_semaphore = asyncio.Semaphore(2)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # required by the openai library but ignored by Ollama
        )
    return _client


async def llm_model_func_v2(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    combined_system = DOMAIN_SYSTEM_PROMPT
    if system_prompt:
        combined_system = f"{DOMAIN_SYSTEM_PROMPT}\n\n{system_prompt}"

    messages = [{"role": "system", "content": combined_system}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    async with _llm_semaphore:
        response = await get_client().chat.completions.create(
            model=LLM_MODEL_V2,
            messages=messages,
            temperature=0.1,
        )
    return response.choices[0].message.content


async def _raw_embedding_func_v2(texts: list[str]) -> list[list[float]]:
    # nomic-embed-text has a ~2048 token context limit per text.
    # Truncate at 6000 chars (~1500 tokens) to stay safely within limit.
    MAX_CHARS = 6000
    texts = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
    response = await get_client().embeddings.create(
        model=EMBEDDING_MODEL_V2,
        input=texts,
        encoding_format="float",
    )
    return np.array([item.embedding for item in response.data])


embedding_func_v2 = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM_V2,
    max_token_size=2048,
    func=_raw_embedding_func_v2,
)


async def vision_model_func_v2(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    image_data: str = None,
    **kwargs,
) -> str:
    # No vision model installed — fall back to text-only for all calls.
    # Pull a vision model (e.g. `ollama pull minicpm-v`) and add image
    # handling here to enable full multimodal support.
    if image_data:
        logger.debug("Vision model not available; processing image/table as text-only.")
    return await llm_model_func_v2(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
    )
