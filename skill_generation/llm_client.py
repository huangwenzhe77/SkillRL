"""Build an OpenAI-compatible or Azure OpenAI client from environment variables.

Custom providers should expose an OpenAI-compatible ``/v1`` Chat Completions
endpoint.  Set ``LLM_API_KEY``, ``LLM_BASE_URL`` and ``LLM_MODEL``.  The
``OPENAI_*`` names are accepted as aliases; Azure variables remain supported
for users who want the original setup.
"""

import os
from typing import Tuple


DEFAULT_MODEL = "gpt-5.6-terra"


def normalize_base_url(base_url: str) -> str:
    """Return a base URL suitable for OpenAI SDK Chat Completions calls."""
    normalized = base_url.strip().rstrip("/")
    if normalized and not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def create_client(default_model: str = DEFAULT_MODEL) -> Tuple[object, str]:
    """Return ``(client, model_name)`` using the configured provider."""
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
    )
    model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or default_model
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise EnvironmentError(
            "Set LLM_API_KEY (and LLM_BASE_URL for a custom OpenAI-compatible provider), "
            "or set OPENAI_API_KEY/AZURE_OPENAI_API_KEY."
        )

    # A custom base URL selects the standard OpenAI-compatible client.  This
    # works with providers such as OpenRouter, DeepSeek, SiliconFlow and local
    # vLLM servers when they implement /v1/chat/completions.
    if base_url:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=normalize_base_url(base_url)), model

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        from openai import AzureOpenAI
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        ), model

    # A normal OpenAI endpoint can be used without setting a base URL.
    from openai import OpenAI
    return OpenAI(api_key=api_key), model
