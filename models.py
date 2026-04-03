import os
import base64
import logging
import numpy as np
from openai import AsyncAzureOpenAI
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

from config import LLM_MODEL, VISION_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM, DOMAIN_SYSTEM_PROMPT

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Defer client creation so importing this module never raises even when
# credentials are not yet in the environment (e.g. during test imports).
_client: AsyncAzureOpenAI | None = None


def get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_key or not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must both be set. "
                "Copy .env.example to .env and fill in your Azure credentials."
            )
        _client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    return _client


async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    # Always prepend the domain system prompt so every RAG-internal
    # LLM call (entity extraction, summarisation, final answer) is
    # aware of the AD/algae research context.
    combined_system = DOMAIN_SYSTEM_PROMPT
    if system_prompt:
        combined_system = f"{DOMAIN_SYSTEM_PROMPT}\n\n{system_prompt}"

    messages = [{"role": "system", "content": combined_system}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


async def _raw_embedding_func(texts: list[str]) -> list[list[float]]:
    response = await get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    return np.array([item.embedding for item in response.data])


# LightRAG requires the embedding callable to be wrapped in EmbeddingFunc
# so it knows the vector dimension and max token size up front.
embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
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

    response = await get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content
