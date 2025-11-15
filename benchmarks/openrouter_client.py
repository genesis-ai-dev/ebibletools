#!/usr/bin/env python3
"""
OpenRouter client helper - Provides OpenAI client configured for OpenRouter API
"""

import os
from openai import OpenAI
from typing import Optional, Dict, Any


# Model name mapping for backward compatibility
# Maps old LiteLLM model names to OpenRouter format (developer/model-name)
MODEL_NAME_MAP = {
    # OpenAI models
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    
    # Anthropic models
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku-20240307",
    "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514",
    
    # Google models
    "gemini-pro": "google/gemini-pro",
    "gemini/gemini-pro": "google/gemini-pro",
    
    # Meta models (for use with different providers)
    "llama-2-7b-chat": "meta-llama/llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/llama-2-70b-chat-hf",
    "llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
    "llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
}


def convert_model_name(model_name: str) -> str:
    """
    Convert model name to OpenRouter format (developer/model-name).
    
    OpenRouter uses the format: developer/model-name (e.g., meta-llama/llama-3-70b-instruct)
    The provider (e.g., cerebras) is specified separately via the X-Provider header.
    
    If the model name is already in OpenRouter format (contains /), it's returned unchanged.
    Otherwise, it's looked up in MODEL_NAME_MAP or assumed to be OpenAI format.
    
    Args:
        model_name: Model name in either old format or OpenRouter format
        
    Returns:
        Model name in OpenRouter format (developer/model-name)
    """
    # If already in OpenRouter format (contains /), return as-is
    if "/" in model_name:
        return model_name
    
    # Check mapping
    if model_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model_name]
    
    # Default to OpenAI if no mapping found and no provider specified
    return f"openai/{model_name}"


def get_openrouter_client(provider: Optional[str] = None) -> OpenAI:
    """
    Get OpenAI client configured for OpenRouter.
    
    Args:
        provider: Optional provider name (e.g., "cerebras") to set X-Provider header
        
    Returns:
        OpenAI client instance configured with OpenRouter base URL and API key
        
    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in environment
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment. "
            "Please set OPENROUTER_API_KEY environment variable."
        )
    
    # Prepare default headers if provider is specified
    default_headers = {}
    if provider:
        default_headers["X-Provider"] = provider
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=default_headers if default_headers else None,
    )


def completion(**kwargs) -> Any:
    """
    Wrapper function that matches litellm.completion() signature.
    This allows drop-in replacement for litellm.completion() calls.
    
    Args:
        **kwargs: Arguments passed to OpenAI chat.completions.create()
                 - model: Model name (will be converted to OpenRouter format: developer/model-name)
                 - messages: List of message dicts
                 - temperature: Optional temperature setting
                 - provider: Optional provider name (e.g., "cerebras") to route request via X-Provider header
                 - Other OpenAI API parameters
                 
    Returns:
        OpenAI completion response object (compatible with LiteLLM response format)
        
    Example:
        # Use Meta model via Cerebras provider
        completion(
            model="meta-llama/llama-3-70b-instruct",
            provider="cerebras",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    # Extract and convert model name
    model = kwargs.pop("model", "openai/gpt-4o")
    model = convert_model_name(model)
    
    # Extract provider if specified (for X-Provider header)
    provider = kwargs.pop("provider", None)
    
    # Get client with provider header if specified
    client = get_openrouter_client(provider=provider)
    
    # Create completion
    response = client.chat.completions.create(
        model=model,
        **kwargs
    )
    
    return response

