#!/usr/bin/env python3
"""
Shared utilities for benchmark scripts
"""

import re
from typing import Optional


def extract_xml_content(text: str, tag: str) -> str:
    """
    Extract content from XML tags in LLM responses, filtering out reasoning content
    
    Args:
        text: The full response text
        tag: The XML tag to extract (without brackets)
    
    Returns:
        The content inside the XML tags, or the filtered text if tags not found
    """
    # First, remove any content within <think></think> tags
    # This filters out reasoning that might contain our target XML tags
    filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Now extract the target XML content from the filtered text
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, filtered_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else filtered_text.strip()


def format_xml_prompt(base_prompt: str, tag: str, description: str) -> str:
    """
    Add XML formatting instructions to a prompt
    
    Args:
        base_prompt: The main prompt text
        tag: The XML tag to use
        description: Description of what goes in the tag
    
    Returns:
        Formatted prompt with XML instructions
    """
    return f"{base_prompt}\n\nProvide your answer in this format:\n<{tag}>{description}</{tag}>" 