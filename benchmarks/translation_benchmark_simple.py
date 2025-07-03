#!/usr/bin/env python3
"""
Simplified Translation Benchmark - Tests translation quality with varying context examples
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

import sys
sys.path.append(str(Path(__file__).parent.parent))
from query import Query
from metrics import chrF_plus, normalized_edit_distance
from benchmark_utils import extract_xml_content, format_xml_prompt


class SimpleTranslationBenchmark:
    def __init__(self, corpus_dir, source_file, model="gpt-4o", query_method="context"):
        self.model = model
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        self.query_method = query_method
        
        # Find target files (exclude source)
        self.target_files = [f for f in self.corpus_dir.glob('*.txt') if f != self.source_file]

    def load_file_pair(self, target_file):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        with open(target_file, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        return source_lines, target_lines

    def get_examples(self, query_obj, query_text, num_examples):
        if num_examples == 0:
            return []
            
        results = query_obj.search_by_text(query_text, top_k=num_examples * 2)
        examples = []
        
        for _, src, tgt, _ in results:
            if len(src) > 10 and len(tgt) > 10 and src != query_text:
                examples.append((src.strip(), tgt.strip()))
                if len(examples) >= num_examples:
                    break
        
        return examples

    def translate(self, text, examples=None):
        if examples:
            base_prompt = "Translate from source to target language. Examples:\n\n"
            for src, tgt in examples:
                base_prompt += f"Source: {src}\nTarget: {tgt}\n\n"
            base_prompt += f"Now translate:\nSource: {text}"
        else:
            base_prompt = f"Translate this text: {text}"
        
        prompt = format_xml_prompt(base_prompt, "translation", "your translation here")
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return extract_xml_content(response.choices[0].message.content.strip(), "translation")

    def run_benchmark(self, num_tests=10, example_counts=None, output_file=None):
        example_counts = example_counts or [0, 3, 5]
        
        # Select random target file
        target_file = random.choice(self.target_files)
        source_lines, target_lines = self.load_file_pair(target_file)
        
        # Find valid test cases
        valid_indices = [
            i for i in range(min(len(source_lines), len(target_lines)))
            if len(source_lines[i].strip()) > 10 and len(target_lines[i].strip()) > 10
        ]
        test_indices = random.sample(valid_indices, min(num_tests, len(valid_indices)))
        
        print(f"🌍 Simple Translation Benchmark")
        print(f"Model: {self.model}")
        print(f"Query method: {self.query_method}")
        print(f"Target: {target_file.name}")
        print(f"Tests: {len(test_indices)}, Example counts: {example_counts}")
        print()
        
        # Initialize results
        results = {count: {"chrf": [], "edit": []} for count in example_counts}
        query_obj = Query(str(self.source_file), str(target_file), method=self.query_method)
        
        # Run tests
        total_tests = len(test_indices) * len(example_counts)
        with tqdm(total=total_tests, desc="Testing translations") as pbar:
            for idx in test_indices:
                source_text = source_lines[idx].strip()
                reference = target_lines[idx].strip()
                
                for count in example_counts:
                    examples = self.get_examples(query_obj, source_text, count)
                    translation = self.translate(source_text, examples)
                    
                    # Calculate scores
                    chrf = chrF_plus(translation, reference)
                    edit = 1.0 - normalized_edit_distance(translation, reference)
                    
                    results[count]["chrf"].append(chrf)
                    results[count]["edit"].append(edit)
                    
                    pbar.update(1)
        
        # Print results
        self.print_results(results, example_counts)
        
        if output_file:
            self.save_results(results, example_counts, output_file)

    def print_results(self, results, example_counts):
        print(f"\n{'='*50}")
        print("TRANSLATION BENCHMARK RESULTS")
        print(f"{'='*50}")
        
        # Print scores for each example count
        for count in example_counts:
            chrf_mean = mean(results[count]["chrf"])
            chrf_std = stdev(results[count]["chrf"]) if len(results[count]["chrf"]) > 1 else 0
            edit_mean = mean(results[count]["edit"])
            edit_std = stdev(results[count]["edit"]) if len(results[count]["edit"]) > 1 else 0
            
            print(f"\n{count} examples:")
            print(f"  chrF+: {chrf_mean:.3f}±{chrf_std:.3f}")
            print(f"  Edit:  {edit_mean:.3f}±{edit_std:.3f}")
        
        # Show improvement analysis
        if len(example_counts) > 1 and 0 in example_counts:
            print(f"\nIMPROVEMENT ANALYSIS:")
            print("-" * 25)
            baseline_chrf = mean(results[0]["chrf"])
            
            for count in example_counts[1:]:
                improvement = mean(results[count]["chrf"]) - baseline_chrf
                print(f"{count} examples: {improvement:+.3f} chrF+ improvement")

    def save_results(self, results, example_counts, output_file):
        output_data = {
            "benchmark": "simple_translation",
            "model": self.model,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "summary": {}
        }
        
        for count in example_counts:
            output_data["summary"][count] = {
                "chrf_mean": mean(results[count]["chrf"]),
                "chrf_std": stdev(results[count]["chrf"]) if len(results[count]["chrf"]) > 1 else 0,
                "edit_mean": mean(results[count]["edit"]),
                "edit_std": stdev(results[count]["edit"]) if len(results[count]["edit"]) > 1 else 0
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Simple Translation Benchmark")
    parser.add_argument("--corpus-dir", default="../Corpus")
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--query-method", default="context", choices=["bm25", "tfidf", "context"])
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--example-counts", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    benchmark = SimpleTranslationBenchmark(
        args.corpus_dir, args.source_file, args.model, args.query_method
    )
    benchmark.run_benchmark(args.num_tests, args.example_counts, args.output)
    print("\n✅ Simple translation benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 