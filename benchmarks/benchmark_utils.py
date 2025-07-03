#!/usr/bin/env python3
"""
Shared utilities for benchmark scripts
"""

import re
from typing import Optional


def extract_xml_content(text: str, tag: str) -> str:
    """
    Extract content from XML tags in LLM responses
    
    Args:
        text: The full response text
        tag: The XML tag to extract (without brackets)
    
    Returns:
        The content inside the XML tags, or the full text if tags not found
    """
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


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