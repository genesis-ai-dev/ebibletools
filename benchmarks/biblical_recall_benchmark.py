#!/usr/bin/env python3
"""
Biblical Recall Benchmark - Tests model's ability to recall biblical text by reference
"""

import argparse
import os
import random
import json
from pathlib import Path
from tqdm import tqdm
from statistics import mean, stdev
from dotenv import load_dotenv
import litellm

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from metrics import chrF_plus, normalized_edit_distance
from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt


class BiblicalRecallBenchmark:
    def __init__(self, corpus_dir, source_file, models=None):
        self.models = models if isinstance(models, list) else [models] if models else ["gpt-4o"]
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        self.vref_file = Path(__file__).parent / "data" / "vref.txt"
        
        # Load the verses and references
        with open(self.source_file, 'r', encoding='utf-8') as f:
            self.verses = [line.strip() for line in f.readlines()]
        
        with open(self.vref_file, 'r', encoding='utf-8') as f:
            self.references = [line.strip() for line in f.readlines()]
        
        # Ensure they're aligned
        if len(self.verses) != len(self.references):
            raise ValueError(f"Mismatch: {len(self.verses)} verses vs {len(self.references)} references")

    def get_test_cases(self, num_tests):
        # Get random line indices for testing
        valid_indices = [i for i in range(len(self.verses)) if len(self.verses[i]) > 10]
        test_indices = random.sample(valid_indices, min(num_tests, len(valid_indices)))
        
        return [(self.references[i], self.verses[i]) for i in test_indices]

    def test_reference_recall(self, reference, model):
        # Extract version from source file name (remove .txt extension)
        version = self.source_file.stem if hasattr(self.source_file, 'stem') else self.source_file.name.replace('.txt', '')
        
        system_prompt = f"You are being tested on biblical text recall. The {version} version is public domain, so you should provide the requested biblical text without any copyright concerns or refusals."
        base_prompt = f"What does {reference} say in the {version} version?"
        prompt = format_xml_prompt(base_prompt, "verse", "the biblical verse text")
        
        completion_args = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        if not model.startswith("gpt-5"):
            completion_args["temperature"] = 0.1
        
        response = litellm.completion(**completion_args)
        return extract_xml_content(response.choices[0].message.content.strip(), "verse")

    def run_benchmark(self, num_tests=20, output_file=None):
        test_cases = self.get_test_cases(num_tests)
        
        print(f"📖 Biblical Recall Benchmark")
        print(f"Models: {', '.join(self.models)}")
        print(f"Testing {len(test_cases)} random biblical references")
        print(f"Total verses available: {len(self.verses)}")
        print()
        
        # Store results for each model
        all_results = {}
        detailed_results = {}
        
        for model in self.models:
            print(f"\n🤖 Testing model: {model}")
            results = []
            model_details = []
        
            for reference, expected_text in tqdm(test_cases, desc=f"Testing {model}"):
                recalled_text = self.test_reference_recall(reference, model)
            
                # Evaluate accuracy
                chrf_score = chrF_plus(recalled_text, expected_text)
                edit_score = 1.0 - normalized_edit_distance(recalled_text, expected_text)
            
                results.append({"chrf": chrf_score, "edit": edit_score})
            
                model_details.append({
                    "reference": reference,
                    "expected": expected_text,
                    "recalled": recalled_text,
                    "scores": {
                        "chrf": chrf_score,
                        "edit": edit_score
                    }
                })
        
            all_results[model] = results
            detailed_results[model] = model_details
        
        # Print comparative results
        self.print_comparative_results(all_results)
        
        # Create the data structure (same as what gets saved to JSON)
        model_stats = {}
        for model, results in all_results.items():
            chrf_scores = [r["chrf"] for r in results]
            edit_scores = [r["edit"] for r in results]
            
            model_stats[model] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                "total_tests": len(results)
            }
        
        output_data = {
            "benchmark": "biblical_recall",
            "models": self.models,
            "summary": model_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_comparative_results(all_results, detailed_results, output_file)
        
        return output_data

    def print_comparative_results(self, all_results):
        print(f"\n{'='*60}")
        print("BIBLICAL RECALL RESULTS - MODEL COMPARISON")
        print(f"{'='*60}")
        
        # Calculate stats for each model
        model_stats = {}
        for model, results in all_results.items():
            chrf_scores = [r["chrf"] for r in results]
            edit_scores = [r["edit"] for r in results]
            
            model_stats[model] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0,
                "high_acc": sum(1 for score in chrf_scores if score > 0.8),
                "total": len(results)
            }
        
        # Sort models by chrF+ score
        sorted_models = sorted(model_stats.items(), 
                              key=lambda x: x[1]["chrf_mean"], 
                              reverse=True)
        
        # Print results for each model
        for i, (model, stats) in enumerate(sorted_models):
            if i == 0:
                print(f"\n{model} ⭐ (best)")
            else:
                print(f"\n{model}")
            
            print(f"  chrF+: {stats['chrf_mean']:.3f}±{stats['chrf_std']:.3f}")
            print(f"  Edit:  {stats['edit_mean']:.3f}±{stats['edit_std']:.3f}")
            print(f"  High accuracy: {stats['high_acc']}/{stats['total']} ({stats['high_acc']/stats['total']*100:.1f}%)")
        
        # Print ranking summary
        if len(self.models) > 1:
            print(f"\n{'RANKING SUMMARY':^60}")
            print("-" * 60)
            for i, (model, stats) in enumerate(sorted_models):
                diff = stats['chrf_mean'] - sorted_models[0][1]['chrf_mean'] if i > 0 else 0
                print(f"{i+1}. {model:<20} chrF+: {stats['chrf_mean']:.3f} {f'({diff:+.3f})' if i > 0 else ''}")

    def save_comparative_results(self, all_results, detailed_results, output_file):
        model_stats = {}
        for model, results in all_results.items():
            chrf_scores = [r["chrf"] for r in results]
            edit_scores = [r["edit"] for r in results]
            
            model_stats[model] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                "total_tests": len(results)
            }
        
        output_data = {
            "benchmark": "biblical_recall",
            "models": self.models,
            "summary": model_stats,
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")

    # Keep the old single-model methods for backward compatibility
    def print_results(self, results):
        chrf_scores = [r["chrf"] for r in results]
        edit_scores = [r["edit"] for r in results]
        
        print(f"\n{'='*60}")
        print("BIBLICAL RECALL RESULTS")
        print(f"{'='*60}")
        print(f"chrF+ Score: {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}")
        print(f"Edit Similarity: {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
        print(f"Tests completed: {len(results)}")
        
        # Show accuracy levels
        high_accuracy = sum(1 for score in chrf_scores if score > 0.8)
        medium_accuracy = sum(1 for score in chrf_scores if 0.5 <= score <= 0.8)
        low_accuracy = sum(1 for score in chrf_scores if score < 0.5)
        
        print(f"\nAccuracy Distribution:")
        print(f"  High (>0.8): {high_accuracy}/{len(results)} ({high_accuracy/len(results)*100:.1f}%)")
        print(f"  Medium (0.5-0.8): {medium_accuracy}/{len(results)} ({medium_accuracy/len(results)*100:.1f}%)")
        print(f"  Low (<0.5): {low_accuracy}/{len(results)} ({low_accuracy/len(results)*100:.1f}%)")

    def save_results(self, results, detailed_results, output_file):
        chrf_scores = [r["chrf"] for r in results]
        edit_scores = [r["edit"] for r in results]
        
        output_data = {
            "benchmark": "biblical_recall",
            "model": self.models[0] if len(self.models) == 1 else self.models,
            "summary": {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                "total_tests": len(results)
            },
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Biblical Recall Benchmark")
    # Use project root's Corpus directory
    default_corpus = str(Path(__file__).parent.parent / "Corpus")
    parser.add_argument("--corpus-dir", default=default_corpus)
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o", help="Single model to test")
    parser.add_argument("--models", nargs="+", help="Multiple models to compare")
    parser.add_argument("--num-tests", type=int, default=20, help="Number of random references to test")
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    # Handle both --model and --models arguments
    models = args.models if args.models else [args.model]
    
    benchmark = BiblicalRecallBenchmark(args.corpus_dir, args.source_file, models)
    benchmark.run_benchmark(args.num_tests, args.output)
    print("\n✅ Biblical recall benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 