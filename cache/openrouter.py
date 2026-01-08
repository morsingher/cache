"""
OpenRouter integration for LLM queries.

Provides helpers to:
- List available free models
- Check API usage/limits
- Run chat completions
"""

import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default free models (fallback if API fetch fails)
DEFAULT_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]


@dataclass
class OpenRouterLimits:
    """API key limits and usage info from OpenRouter."""
    label: str | None
    credit_limit: float | None
    credits_remaining: float | None
    usage_daily: float | None
    rate_limit_requests: int | None
    rate_limit_interval: str | None


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error: str | None = None


def get_api_key() -> str | None:
    """Get OpenRouter API key from Streamlit secrets."""
    return st.secrets.get("OPENROUTER_API_KEY", "").strip() or None


def fetch_free_models(api_key: str | None = None) -> list[str]:
    """
    Fetch list of free models from OpenRouter.
    
    Returns model IDs ending with ':free'.
    Falls back to DEFAULT_FREE_MODELS if fetch fails.
    """
    if OpenAI is None:
        return DEFAULT_FREE_MODELS
    
    key = api_key or get_api_key()
    if not key:
        return DEFAULT_FREE_MODELS
    
    try:
        client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
        models = client.models.list().data
        free_models = [m.id for m in models if m.id.endswith(":free")]
        # Sort alphabetically for consistent display
        free_models.sort()
        return free_models if free_models else DEFAULT_FREE_MODELS
    except Exception:
        return DEFAULT_FREE_MODELS


def fetch_limits(api_key: str | None = None) -> OpenRouterLimits | None:
    """
    Fetch API key limits and usage from OpenRouter.
    
    Returns None if fetch fails.
    """
    key = api_key or get_api_key()
    if not key:
        return None
    
    try:
        resp = requests.get(
            f"{OPENROUTER_BASE_URL}/auth/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        
        return OpenRouterLimits(
            label=data.get("label"),
            credit_limit=data.get("limit"),
            credits_remaining=data.get("limit_remaining"),
            usage_daily=data.get("usage"),
            rate_limit_requests=data.get("rate_limit", {}).get("requests"),
            rate_limit_interval=data.get("rate_limit", {}).get("interval"),
        )
    except Exception:
        return None


def chat_completion(
    prompt: str,
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    api_key: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
) -> ChatResponse:
    """
    Run a chat completion via OpenRouter.
    
    Args:
        prompt: The full prompt (treated as user message).
        model: Model ID to use.
        api_key: OpenRouter API key (uses env var if not provided).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    
    Returns:
        ChatResponse with content or error.
    """
    return chat_completion_messages(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def chat_completion_messages(
    messages: list[dict[str, str]],
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    api_key: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
) -> ChatResponse:
    """
    Run a multi-turn chat completion via OpenRouter.
    
    Args:
        messages: List of message dicts with "role" and "content" keys.
                  Roles can be "system", "user", or "assistant".
        model: Model ID to use.
        api_key: OpenRouter API key (uses env var if not provided).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    
    Returns:
        ChatResponse with content or error.
    """
    if OpenAI is None:
        return ChatResponse(
            content="",
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error="OpenAI library not installed. Run: pip install openai",
        )
    
    key = api_key or get_api_key()
    if not key:
        return ChatResponse(
            content="",
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error="OPENROUTER_API_KEY environment variable is not set.",
        )
    
    try:
        client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        return ChatResponse(
            content=content,
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
    
    except Exception as e:
        error_msg = str(e)
        # Try to extract more useful error message
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            error_msg = "Rate limit exceeded. Please wait a moment before trying again."
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            error_msg = "Invalid API key. Please check your OPENROUTER_API_KEY."
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            error_msg = f"Model '{model}' not found or not available. Try a different model."
        
        return ChatResponse(
            content="",
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error=error_msg,
        )

