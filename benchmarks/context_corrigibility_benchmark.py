#!/usr/bin/env python3
"""
Context Corrigibility Benchmark - Tests how in-context examples improve translation
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


class ContextCorrigibilityBenchmark:
    def __init__(self, corpus_dir, source_file, model="gpt-4o", query_method="context"):
        self.model = model
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        self.query_method = query_method
        
        self.target_files = [f for f in self.corpus_dir.glob('*.txt') if f != self.source_file]

    def load_file_pair(self, target_file):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        with open(target_file, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        return source_lines, target_lines

    def get_test_cases(self, num_tests):
        target_file = random.choice(self.target_files)
        source_lines, target_lines = self.load_file_pair(target_file)
        
        valid_indices = [
            i for i in range(min(len(source_lines), len(target_lines)))
            if len(source_lines[i].strip()) > 10 and len(target_lines[i].strip()) > 10
        ]
        
        test_indices = random.sample(valid_indices, min(num_tests, len(valid_indices)))
        return [(source_lines[i].strip(), target_lines[i].strip(), target_file.name) 
                for i in test_indices], target_file

    def get_examples(self, query_obj, query_text, num_examples):
        results = query_obj.search_by_text(query_text, top_k=num_examples * 2)
        examples = []
        
        for result in results:
            line_num, src, tgt, score = result
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

    def evaluate_translation(self, hypothesis, reference):
        return {
            'chrf': chrF_plus(hypothesis, reference),
            'edit': 1.0 - normalized_edit_distance(hypothesis, reference)
        }

    def run_benchmark(self, num_tests=10, example_counts=None, output_file=None):
        example_counts = example_counts or [0, 3, 5]
        test_cases, target_file = self.get_test_cases(num_tests)
        
        print(f"🔄 Context Corrigibility Benchmark")
        print(f"Model: {self.model}")
        print(f"Query method: {self.query_method}")
        print(f"Target file: {target_file.name}")
        print(f"Tests: {len(test_cases)}")
        print(f"Example counts: {example_counts}")
        print()
        
        results = {count: [] for count in example_counts}
        detailed_results = []
        
        query_obj = Query(str(self.source_file), str(target_file), method=self.query_method)
        
        total_tests = len(test_cases) * len(example_counts)
        
        with tqdm(total=total_tests, desc="Testing context corrigibility") as pbar:
            for source_text, reference, _ in test_cases:
                test_result = {
                    "source": source_text,
                    "reference": reference,
                    "results": {}
                }
                
                for count in example_counts:
                    examples = self.get_examples(query_obj, source_text, count) if count > 0 else []
                    translation = self.translate(source_text, examples)
                    scores = self.evaluate_translation(translation, reference)
                    
                    results[count].append(scores)
                    test_result["results"][count] = {
                        "translation": translation,
                        "scores": scores,
                        "num_examples": len(examples)
                    }
                    
                    pbar.update(1)
                
                detailed_results.append(test_result)
        
        self.print_results(results, example_counts)
        
        if output_file:
            self.save_results(results, detailed_results, example_counts, output_file)

    def print_results(self, results, example_counts):
        print(f"\n{'='*60}")
        print("CONTEXT CORRIGIBILITY RESULTS")
        print(f"{'='*60}")
        
        for count in example_counts:
            chrf_scores = [r["chrf"] for r in results[count]]
            edit_scores = [r["edit"] for r in results[count]]
            
            print(f"\n{count} examples:")
            print(f"  chrF+: {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}")
            print(f"  Edit: {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
        
        print(f"\nCORRIGIBILITY ANALYSIS:")
        print("-" * 25)
        baseline_scores = [r["chrf"] for r in results[0]]
        for count in example_counts[1:]:
            context_scores = [r["chrf"] for r in results[count]]
            improvement = mean(context_scores) - mean(baseline_scores)
            print(f"{count} examples: {improvement:+.3f} chrF+ improvement")

    def save_results(self, results, detailed_results, example_counts, output_file):
        output_data = {
            "benchmark": "context_corrigibility",
            "model": self.model,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "summary": {},
            "detailed_results": detailed_results
        }
        
        for count in example_counts:
            chrf_scores = [r["chrf"] for r in results[count]]
            edit_scores = [r["edit"] for r in results[count]]
            
            output_data["summary"][count] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Context Corrigibility Benchmark")
    parser.add_argument("--corpus-dir", default="../Corpus")
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--query-method", default="context", choices=["bm25", "tfidf", "context"])
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--example-counts", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    benchmark = ContextCorrigibilityBenchmark(
        args.corpus_dir, args.source_file, args.model, args.query_method
    )
    benchmark.run_benchmark(args.num_tests, args.example_counts, args.output)
    print("\n✅ Context corrigibility benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 