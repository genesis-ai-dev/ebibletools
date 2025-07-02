#!/usr/bin/env python3
"""Translation benchmark comparing different numbers of examples for ContextQuery in-context learning"""

import argparse
import os
import random
import time
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from query.contextquery import ContextQuery


class TranslationBenchmark:
    def __init__(self, api_key, corpus_dir, source_file):
        self.client = OpenAI(api_key=api_key)
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file

    def get_target_files(self, num_files):
        """Get random target files, excluding the source file"""
        all_files = list(self.corpus_dir.glob('*.txt'))
        target_files = [f for f in all_files if f != self.source_file]
        
        if len(target_files) > num_files:
            target_files = random.sample(target_files, num_files)
        
        return target_files

    def load_file_pair(self, source_file, target_file):
        """Load source-target file pair"""
        with open(source_file, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        with open(target_file, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        return source_lines, target_lines

    def get_examples(self, context_query, query_text, num_examples):
        """Get examples using ContextQuery"""
        results = context_query.search_by_text(query_text, top_k=num_examples * 2)
        examples = []
        
        for _, src, tgt, _ in results:
            if len(src) > 10 and len(tgt) > 10 and src != query_text:
                examples.append((src.strip(), tgt.strip()))
                if len(examples) >= num_examples:
                    break
        
        return examples

    def translate(self, text, examples=None):
        """Translate using OpenAI with optional examples"""
        if examples:
            prompt = "Translate from source to target language. Examples:\n\n"
            for src, tgt in examples:
                prompt += f"Source: {src}\nTarget: {tgt}\n\n"
            prompt += f"Now translate:\nSource: {text}\nTarget:"
        else:
            prompt = f"Translate this text: {text}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def similarity(self, text1, text2):
        """Calculate text similarity (0-1)"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def run_benchmark(self, num_target_files, num_tests_per_file, example_counts):
        """Run the benchmark comparing different numbers of examples"""
        target_files = self.get_target_files(num_target_files)
        
        print(f"Source: {self.source_file.name}")
        print(f"Target files: {len(target_files)}")
        print(f"Tests per file: {num_tests_per_file}")
        print(f"Example counts: {example_counts}")
        print(f"Total tests: {len(target_files) * num_tests_per_file}")
        
        results = {f"examples_{count}": [] for count in example_counts}
        
        for target_file in tqdm(target_files, desc="Processing files"):
            source_lines, target_lines = self.load_file_pair(self.source_file, target_file)
            context_query = ContextQuery(str(self.source_file), str(target_file), verbose=False)
            
            valid_indices = [
                i for i in range(min(len(source_lines), len(target_lines)))
                if len(source_lines[i].strip()) > 10 and len(target_lines[i].strip()) > 10
            ]
            
            if len(valid_indices) < num_tests_per_file:
                print(f"Skipping {target_file.name} - insufficient valid lines")
                continue
            
            test_indices = random.sample(valid_indices, num_tests_per_file)
            
            for idx in test_indices:
                source_text = source_lines[idx].strip()
                ground_truth = target_lines[idx].strip()
                
                for count in example_counts:
                    examples = self.get_examples(context_query, source_text, count)
                    translation = self.translate(source_text, examples)
                    similarity = self.similarity(translation, ground_truth)
                    results[f"examples_{count}"].append(similarity)
                    time.sleep(0.1)  # Rate limiting
        
        self.print_results(results, example_counts)

    def print_results(self, results, example_counts):
        """Print benchmark results"""
        print(f"\n{'='*70}")
        print("CONTEXTQUERY BENCHMARK RESULTS")
        print(f"{'='*70}")
        
        averages = {method: sum(scores) / len(scores) for method, scores in results.items() if scores}
        
        if not averages:
            print("No valid results to display")
            return
        
        sorted_methods = sorted(averages.keys(), key=lambda m: averages[m], reverse=True)
        best_avg = averages[sorted_methods[0]]
        
        for method in sorted_methods:
            avg = averages[method]
            count = method.split('_')[1]
            if method == sorted_methods[0]:
                print(f"{count:>2} examples: {avg:.3f} (best)")
            else:
                diff = avg - best_avg
                pct_diff = (diff / best_avg) * 100
                print(f"{count:>2} examples: {avg:.3f} ({diff:+.3f}, {pct_diff:+.1f}% vs best)")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Translation benchmark using ContextQuery")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), 
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--corpus-dir", default="../Corpus", 
                       help="Directory containing corpus files")
    parser.add_argument("--source-file", default="eng-engULB.txt", 
                       help="Source file name")
    parser.add_argument("--num-target-files", type=int, default=2, 
                       help="Number of target files to test")
    parser.add_argument("--num-tests-per-file", type=int, default=5, 
                       help="Number of tests per file")
    parser.add_argument("--example-counts", nargs="+", type=int, default=[3, 5], 
                       help="Numbers of examples to compare")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        return 1
    
    benchmark = TranslationBenchmark(args.api_key, args.corpus_dir, args.source_file)
    benchmark.run_benchmark(args.num_target_files, args.num_tests_per_file, args.example_counts)
    
    return 0


if __name__ == "__main__":
    exit(main()) 