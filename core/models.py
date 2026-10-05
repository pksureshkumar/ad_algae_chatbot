import re
import base64
import logging
import numpy as np
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

from config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_VISION_MODEL,
    OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_MAX_TOKENS,
    EMBEDDING_BATCH_SIZE, OLLAMA_THINKING,
    DOMAIN_SYSTEM_PROMPT,
)

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# --- Ollama client (LLM, vision, and embeddings — all local) ---
_ollama_client: AsyncOpenAI | None = None


def get_ollama_client() -> AsyncOpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't validate the key
            timeout=600.0,     # local generation is slower than a hosted API
            max_retries=3,
        )
    return _ollama_client


def _strip_thinking(text: str) -> str:
    """Remove qwen3 <think>...</think> blocks from responses.

    Recent Ollama versions return reasoning in a separate field rather than
    inline, so this is usually a no-op — kept for older servers and for models
    that still emit the tags inline.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Ollama accepts `think` as a non-standard field on the OpenAI-compatible
# endpoint; the SDK forwards unknown keys only via extra_body.
_EXTRA_BODY = {} if OLLAMA_THINKING else {"think": False}


async def llm_model_func(
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

    response = await get_ollama_client().chat.completions.create(
        model=OLLAMA_LLM_MODEL,
        messages=messages,
        temperature=0.1,
        extra_body=_EXTRA_BODY,
    )
    return _strip_thinking(response.choices[0].message.content or "")


async def _raw_embedding_func(texts: list[str]) -> np.ndarray:
    # Ollama serialises embedding requests anyway, and large batches make it
    # stall on long scientific chunks — so send fixed-size sub-batches.
    client = get_ollama_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(
            model=OLLAMA_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        # Ollama does not guarantee response ordering matches input ordering.
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)

    arr = np.array(vectors, dtype=np.float32)
    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"{OLLAMA_EMBEDDING_MODEL} returned dim={arr.shape[1]} but config "
            f"declares EMBEDDING_DIM={EMBEDDING_DIM}. Fix config.py before ingesting — "
            "a mismatch corrupts the vector store."
        )
    return arr


# LightRAG requires the embedding callable to be wrapped in EmbeddingFunc
# so it knows the vector dimension and max token size up front.
embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=EMBEDDING_MAX_TOKENS,
    func=_raw_embedding_func,
)


async def vision_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    image_data: str = None,  # RAGAnything passes extracted images here
    **kwargs,
) -> str:
    # If no image data was provided, fall back to the plain LLM.
    if not image_data:
        return await llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
        )

    # Build the image content block — accept data URIs, HTTP URLs, local paths,
    # or raw base64 strings (RAGAnything passes base64 without a data: prefix).
    if image_data.startswith(("data:", "http://", "https://")):
        image_url = image_data
    elif len(image_data) > 260 or not any(c in image_data for c in ("/", "\\", ".")):
        # Looks like a raw base64 string rather than a file path.
        # JPEG base64 starts with /9j/; PNG with iVBOR; default to jpeg.
        if image_data.startswith("iVBOR"):
            mime = "png"
        elif image_data.startswith("R0lGOD"):
            mime = "gif"
        elif image_data.startswith("UklGR"):
            mime = "webp"
        else:
            mime = "jpeg"
        image_url = f"data:image/{mime};base64,{image_data}"
    else:
        # Local file path — encode to a base64 data URI.
        try:
            with open(image_data, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = image_data.rsplit(".", 1)[-1].lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                    "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
            image_url = f"data:image/{mime};base64,{b64}"
        except Exception as e:
            logger.warning(f"Could not load image {image_data[:80]}...: {e}. Falling back to text-only.")
            return await llm_model_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
            )

    combined_system = DOMAIN_SYSTEM_PROMPT
    if system_prompt:
        combined_system = f"{DOMAIN_SYSTEM_PROMPT}\n\n{system_prompt}"

    messages = [{"role": "system", "content": combined_system}]
    messages.extend(history_messages)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    })

    response = await get_ollama_client().chat.completions.create(
        model=OLLAMA_VISION_MODEL,
        messages=messages,
        max_tokens=2048,
        extra_body=_EXTRA_BODY,
    )
    return _strip_thinking(response.choices[0].message.content or "")
