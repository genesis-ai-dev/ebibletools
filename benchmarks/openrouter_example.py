#!/usr/bin/env python3
"""
Example demonstrating how to use benchmarks with different LLM providers via OpenRouter.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.biblical_recall_benchmark import BiblicalRecallBenchmark

def main():
    load_dotenv()
    
    print("🌍 Biblical Recall Benchmark with OpenRouter")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   Get your API key from: https://openrouter.ai/docs/api-keys")
        return
    
    # Use project root's Corpus directory
    default_corpus = str(Path(__file__).parent.parent / "Corpus")
    
    # Example 1: Using OpenAI GPT-4o via OpenRouter
    print("\n1. Using OpenAI GPT-4o via OpenRouter...")
    benchmark_openai = BiblicalRecallBenchmark(
        corpus_dir=default_corpus,
        source_file="eng-engULB.txt",
        models=["openai/gpt-4o"]  # OpenRouter format: provider/model-name
    )
    
    # Run a small benchmark
    benchmark_openai.run_benchmark(
        num_tests=5,
        output_file="openai_results.json"
    )
    
    # Example 2: Using Anthropic Claude via OpenRouter
    print("\n2. Using Anthropic Claude-3-Haiku via OpenRouter...")
    benchmark_claude = BiblicalRecallBenchmark(
        corpus_dir=default_corpus,
        source_file="eng-engULB.txt",
        models=["anthropic/claude-3-haiku-20240307"]
    )
    
    benchmark_claude.run_benchmark(
        num_tests=5,
        output_file="claude_results.json"
    )
    
    # Example 3: Multi-model comparison
    print("\n3. Comparing multiple models via OpenRouter...")
    benchmark_multi = BiblicalRecallBenchmark(
        corpus_dir=default_corpus,
        source_file="eng-engULB.txt",
        models=[
            "openai/gpt-4o",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku-20240307"
        ]
    )
    
    benchmark_multi.run_benchmark(
        num_tests=5,
        output_file="multi_model_results.json"
    )
    
    print("\n" + "="*50)
    print("✅ Example completed!")
    print("\nOpenRouter Model Format: provider/model-name")
    print("\nExamples:")
    print("  - OpenAI: 'openai/gpt-4o', 'openai/gpt-3.5-turbo'")
    print("  - Anthropic: 'anthropic/claude-3-haiku-20240307'")
    print("  - Google: 'google/gemini-pro'")
    print("  - Groq: 'groq/llama-3.1-70b-versatile'")
    print("\nNote: Old format model names (e.g., 'gpt-4o') are automatically")
    print("      converted to OpenRouter format (e.g., 'openai/gpt-4o')")
    print("\nSee OpenRouter models: https://openrouter.ai/models")

if __name__ == "__main__":
    main()

