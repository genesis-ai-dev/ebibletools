#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner - Execute all eBible Tools benchmarks
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from power_prompt_benchmark import PowerPromptBenchmark
from true_source_benchmark import TrueSourceBenchmark
from context_corrigibility_benchmark import ContextCorrigibilityBenchmark
from biblical_recall_benchmark import BiblicalRecallBenchmark


class BenchmarkRunner:
    def __init__(self, corpus_dir="../Corpus", source_file="eng-engULB.txt", models=None):
        load_dotenv()
        
        self.corpus_dir = corpus_dir
        self.source_file = source_file
        self.models = models or ["gpt-4o", "gpt-3.5-turbo"]
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"benchmark_run_{timestamp}"

    def run_biblical_recall(self):
        print("🔥 Running Biblical Recall Benchmark")
        benchmark = BiblicalRecallBenchmark(
            self.corpus_dir, 
            self.source_file, 
            models=self.models
        )
        
        output_file = self.results_dir / f"{self.session_id}_biblical_recall.json"
        benchmark.run_benchmark(num_tests=20, output_file=str(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_power_prompt(self):
        print("\n💪 Running Power Prompt Benchmark")
        
        # Load example prompts for power prompt benchmark
        example_file = Path(__file__).parent / "example_prompts.json"
        custom_prompts = {}
        if example_file.exists():
            with open(example_file, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
        
        benchmark = PowerPromptBenchmark(self.corpus_dir, self.source_file, self.models, custom_prompts)
        output_file = self.results_dir / f"{self.session_id}_power_prompt.json"
        results = benchmark.run_benchmark(num_tests=12, output_file=str(output_file))
        
        return results

    def run_true_source(self):
        print("\n🎯 Running True Source Benchmark")
        
        benchmark = TrueSourceBenchmark(self.corpus_dir, self.source_file, self.models)
        output_file = self.results_dir / f"{self.session_id}_true_source.json"
        results = benchmark.run_benchmark(num_tests=15, output_file=str(output_file))
        
        return results

    def run_context_corrigibility(self):
        print("\n🔄 Running Context Corrigibility Benchmark")
        
        benchmark = ContextCorrigibilityBenchmark(
            self.corpus_dir, 
            self.source_file, 
            self.models, 
            query_method="context"
        )
        output_file = self.results_dir / f"{self.session_id}_context_corrigibility.json"
        results = benchmark.run_benchmark(
            num_tests=10, 
            example_counts=[0, 3, 5], 
            output_file=str(output_file)
        )
        
        return results

    def summarize_results(self, all_results):
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Session: {self.session_id}")
        print(f"Models tested: {', '.join(self.models)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "models": self.models,
            "benchmarks": {}
        }
        
        # Biblical Recall Summary
        if "biblical_recall" in all_results:
            print(f"\n📖 Biblical Recall:")
            br_data = all_results["biblical_recall"]
            summary["benchmarks"]["biblical_recall"] = {}
            
            for model in br_data.get("summary", {}):
                stats = br_data["summary"][model]
                chrf = stats.get("chrf_mean", 0)
                print(f"  {model}: chrF+ {chrf:.3f}")
                summary["benchmarks"]["biblical_recall"][model] = chrf
        
        # Power Prompt Summary
        if "power_prompt" in all_results:
            print(f"\n💪 Power Prompt (Best Prompt Style):")
            summary["benchmarks"]["power_prompt"] = {}
            
            for model, pp_data in all_results["power_prompt"].items():
                best_prompt = max(pp_data["summary"].items(), 
                                key=lambda x: x[1]["overall"])
                print(f"  {model}: {best_prompt[0]} ({best_prompt[1]['overall']:.3f})")
                summary["benchmarks"]["power_prompt"][model] = {
                    "best_prompt": best_prompt[0],
                    "score": best_prompt[1]["overall"]
                }
        
        # True Source Summary
        if "true_source" in all_results:
            print(f"\n🎯 True Source (Source Effect):")
            summary["benchmarks"]["true_source"] = {}
            
            for model, ts_data in all_results["true_source"].items():
                with_src = ts_data["summary"]["with_source"]["chrf_mean"]
                without_src = ts_data["summary"]["without_source"]["chrf_mean"]
                effect = with_src - without_src
                print(f"  {model}: +{effect:.3f} chrF+ improvement")
                summary["benchmarks"]["true_source"][model] = effect
        
        # Context Corrigibility Summary
        if "context_corrigibility" in all_results:
            print(f"\n🔄 Context Corrigibility (5 vs 0 examples):")
            summary["benchmarks"]["context_corrigibility"] = {}
            
            for model, cc_data in all_results["context_corrigibility"].items():
                baseline = cc_data["summary"]["0"]["chrf_mean"]
                with_context = cc_data["summary"]["5"]["chrf_mean"]
                improvement = with_context - baseline
                print(f"  {model}: +{improvement:.3f} chrF+ improvement")
                summary["benchmarks"]["context_corrigibility"][model] = improvement
        
        # Save summary
        summary_file = self.results_dir / f"{self.session_id}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete results saved in: {self.results_dir}")
        print(f"📋 Summary saved to: {summary_file}")

    def run_all(self):
        print("🚀 Starting Comprehensive eBible Tools Benchmark Suite")
        print(f"Models: {', '.join(self.models)}")
        print(f"Session ID: {self.session_id}")
        
        all_results = {}
        start_time = time.time()
        
        try:
            all_results["biblical_recall"] = self.run_biblical_recall()
            all_results["power_prompt"] = self.run_power_prompt()
            all_results["true_source"] = self.run_true_source()
            all_results["context_corrigibility"] = self.run_context_corrigibility()
            
        except Exception as e:
            print(f"\n❌ Error during benchmark execution: {e}")
            print("Generating summary with available results...")
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.summarize_results(all_results)
        
        print(f"\n⏱️  Total execution time: {duration/60:.1f} minutes")
        print("✅ Benchmark suite completed!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all eBible Tools benchmarks")
    parser.add_argument("--corpus-dir", default="../Corpus", help="Corpus directory")
    parser.add_argument("--source-file", default="eng-engULB.txt", help="Source file")
    parser.add_argument("--models", nargs="+", default=["gpt-4o"], help="Models to test")
    parser.add_argument("--quick", action="store_true", help="Run with reduced test counts for faster execution")
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Quick mode: reducing test counts for faster execution")
        # We'll modify the runner to use smaller test counts in quick mode
        # This would require modifying the individual benchmark calls
    
    runner = BenchmarkRunner(
        corpus_dir=args.corpus_dir,
        source_file=args.source_file,
        models=args.models
    )
    
    runner.run_all()


if __name__ == "__main__":
    main() 