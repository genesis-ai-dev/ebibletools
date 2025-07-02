#!/usr/bin/env python3
"""
Example demonstrating how to use the translation benchmark with different LLM providers via liteLLM.
"""

import os
from dotenv import load_dotenv
from translation_benchmark import TranslationBenchmark

def main():
    load_dotenv()
    
    print("🌍 Translation Benchmark with liteLLM")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Example 1: Using OpenAI GPT-4o (default)
    print("\n1. Using OpenAI GPT-4o...")
    benchmark_openai = TranslationBenchmark(
        api_key=api_key,
        corpus_dir="../Corpus",
        source_file="eng-engULB.txt",
        query_method="context",
        model="gpt-4o"  # OpenAI model
    )
    
    # Run a small benchmark
    benchmark_openai.run_benchmark(
        num_target_files=1,
        num_tests_per_file=3,
        example_counts=[0, 3],
        output_file="openai_results.json"
    )
    
    print("\n" + "="*50)
    print("✅ Example completed!")
    print("\nTo use other providers, simply change the model parameter:")
    print("  - Anthropic: model='anthropic/claude-3-sonnet-20240229'")
    print("  - Google: model='gemini/gemini-pro'")
    print("  - Groq: model='groq/llama2-70b-4096'")
    print("  - OpenRouter: model='openrouter/anthropic/claude-3-haiku'")
    print("\nMake sure to set the appropriate API keys as environment variables.")
    print("See liteLLM documentation for full provider list: https://docs.litellm.ai/docs/providers")

if __name__ == "__main__":
    main() 