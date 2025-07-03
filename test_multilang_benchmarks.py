#!/usr/bin/env python3
"""
Test script for the new multi-language benchmark functionality
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory and benchmarks to path for proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'benchmarks'))

# Now import the benchmarks
from benchmarks.context_corrigibility_benchmark import ContextCorrigibilityBenchmark
from benchmarks.true_source_benchmark import TrueSourceBenchmark
from benchmarks.power_prompt_benchmark import PowerPromptBenchmark

def main():
    print("🧪 Testing Multi-Language Benchmark Updates")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv('benchmarks/.env')
    
    # Configuration
    corpus_dir = "Corpus"
    source_file = "eng-engULB.txt"
    model = "gpt-4o-mini"  # Using smaller model for testing
    
    # Check if we have files
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print("❌ Corpus directory not found!")
        print("Please run the ebible downloader first to get some test files.")
        return
    
    source_path = corpus_path / source_file
    if not source_path.exists():
        print(f"❌ Source file {source_file} not found!")
        return
    
    # Find available target files
    target_files = [f for f in corpus_path.glob('*.txt') if f != source_path]
    if not target_files:
        print("❌ No target language files found!")
        print("Please download some target language files first.")
        return
    
    print(f"✅ Found source file: {source_file}")
    print(f"✅ Found {len(target_files)} target language files:")
    for f in target_files[:5]:  # Show first 5
        print(f"   - {f.name}")
    if len(target_files) > 5:
        print(f"   ... and {len(target_files) - 5} more")
    print()
    
    # Test 1: Context Corrigibility Benchmark
    print("🔄 Testing Context Corrigibility Benchmark (multi-language)")
    try:
        benchmark = ContextCorrigibilityBenchmark(corpus_dir, source_file, model=model)
        results = benchmark.run_benchmark(num_tests=2, example_counts=[0, 3], output_file="test_context.json")
        
        if results:
            print(f"✅ Success! Tested {len(results['languages_tested'])} languages")
            print(f"📊 Overall results available in: summary.overall")
            print(f"📍 Per-language results available in: summary.per_language")
        else:
            print("❌ No results returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test 2: True Source Benchmark
    print("🎯 Testing True Source Benchmark (multi-language)")
    try:
        benchmark = TrueSourceBenchmark(corpus_dir, source_file, model=model)
        results = benchmark.run_benchmark(num_tests=2, output_file="test_source.json")
        
        if results:
            print(f"✅ Success! Tested {len(results['languages_tested'])} languages")
            print(f"📊 Overall results available in: summary.overall")
            print(f"📍 Per-language results available in: summary.per_language")
        else:
            print("❌ No results returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test 3: Power Prompt Benchmark
    print("💪 Testing Power Prompt Benchmark (multi-language)")
    try:
        benchmark = PowerPromptBenchmark(corpus_dir, source_file, model=model)
        results = benchmark.run_benchmark(num_tests=2, output_file="test_prompt.json")
        
        if results:
            print(f"✅ Success! Tested {len(results['languages_tested'])} languages")
            print(f"📊 Overall results available in: summary.overall")
            print(f"📍 Per-language results available in: summary.per_language")
        else:
            print("❌ No results returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Multi-language benchmark testing completed!")
    print("\nKey improvements:")
    print("✅ All translation benchmarks now test on ALL available languages")
    print("✅ Results are organized by language with per-language and overall statistics")
    print("✅ Much more comprehensive coverage than single random language")
    print("✅ Better insights into model performance across different languages")

if __name__ == "__main__":
    main() 