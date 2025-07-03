#!/usr/bin/env python3
"""
True Source Benchmark - Tests how source text affects translation accuracy
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
from metrics import chrF_plus, normalized_edit_distance
from benchmark_utils import extract_xml_content, format_xml_prompt


class TrueSourceBenchmark:
    def __init__(self, corpus_dir, source_file, model="gpt-4o"):
        self.model = model
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        
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
                for i in test_indices]

    def translate_with_source(self, source_text, target_lang):
        base_prompt = f"Translate from source to {target_lang}: {source_text}"
        prompt = format_xml_prompt(base_prompt, "translation", "your translation here")
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return extract_xml_content(response.choices[0].message.content.strip(), "translation")

    def translate_without_source(self, reference_text, target_lang):
        # Just ask to translate to target language without giving source
        base_prompt = f"Translate this to {target_lang}: {reference_text}"
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

    def run_benchmark(self, num_tests=15, output_file=None):
        test_cases = self.get_test_cases(num_tests)
        target_lang = test_cases[0][2].split('-')[0] if test_cases else "target language"
        
        print(f"🎯 True Source Benchmark")
        print(f"Model: {self.model}")
        print(f"Target Language: {target_lang}")
        print(f"Tests: {len(test_cases)}")
        print()
        
        results = {"with_source": [], "without_source": []}
        detailed_results = []
        
        for source_text, reference, _ in tqdm(test_cases, desc="Testing source effects"):
            # Test 1: With correct source
            trans_with_source = self.translate_with_source(source_text, target_lang)
            scores_with = self.evaluate_translation(trans_with_source, reference)
            
            # Test 2: Without source (asking model to translate the reference back)
            trans_without_source = self.translate_without_source(reference, target_lang)
            scores_without = self.evaluate_translation(trans_without_source, reference)
            
            results["with_source"].append(scores_with)
            results["without_source"].append(scores_without)
            
            detailed_results.append({
                "source": source_text,
                "reference": reference,
                "translations": {
                    "with_source": trans_with_source,
                    "without_source": trans_without_source
                },
                "scores": {
                    "with_source": scores_with,
                    "without_source": scores_without
                }
            })
        
        self.print_results(results)
        
        # Create the data structure (same as what gets saved to JSON)
        summary_stats = {}
        for condition in ["with_source", "without_source"]:
            chrf_scores = [r["chrf"] for r in results[condition]]
            edit_scores = [r["edit"] for r in results[condition]]
            
            summary_stats[condition] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0
            }
        
        output_data = {
            "benchmark": "true_source",
            "model": self.model,
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(results, detailed_results, output_file)
        
        return output_data

    def print_results(self, results):
        print(f"\n{'='*60}")
        print("TRUE SOURCE RESULTS")
        print(f"{'='*60}")
        
        for condition in ["with_source", "without_source"]:
            chrf_scores = [r["chrf"] for r in results[condition]]
            edit_scores = [r["edit"] for r in results[condition]]
            
            print(f"\n{condition.replace('_', ' ').title()}:")
            print(f"  chrF+: {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}")
            print(f"  Edit Similarity: {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
        
        # Show the difference
        with_chrf = [r["chrf"] for r in results["with_source"]]
        without_chrf = [r["chrf"] for r in results["without_source"]]
        improvement = mean(with_chrf) - mean(without_chrf)
        
        print(f"\nSOURCE EFFECT:")
        print(f"  chrF+ improvement with source: {improvement:+.3f}")

    def save_results(self, results, detailed_results, output_file):
        output_data = {
            "benchmark": "true_source",
            "model": self.model,
            "summary": {},
            "detailed_results": detailed_results
        }
        
        for condition in ["with_source", "without_source"]:
            chrf_scores = [r["chrf"] for r in results[condition]]
            edit_scores = [r["edit"] for r in results[condition]]
            
            output_data["summary"][condition] = {
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
    
    parser = argparse.ArgumentParser(description="True Source Benchmark")
    parser.add_argument("--corpus-dir", default="../Corpus")
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--num-tests", type=int, default=15)
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    benchmark = TrueSourceBenchmark(args.corpus_dir, args.source_file, args.model)
    benchmark.run_benchmark(args.num_tests, args.output)
    print("\n✅ True source benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 