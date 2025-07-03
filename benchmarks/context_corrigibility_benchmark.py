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
from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt


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



    def get_test_cases_per_language(self, num_tests_per_language):
        """Get test cases for each target language"""
        all_language_tests = {}
        
        for target_file in self.target_files:
            try:
                source_lines, target_lines = self.load_file_pair(target_file)
                
                valid_indices = [
                    i for i in range(min(len(source_lines), len(target_lines)))
                    if len(source_lines[i].strip()) > 10 and len(target_lines[i].strip()) > 10
                ]
                
                if len(valid_indices) < num_tests_per_language:
                    print(f"⚠️  Skipping {target_file.name} - only {len(valid_indices)} valid lines (need {num_tests_per_language})")
                    continue
                
                test_indices = random.sample(valid_indices, num_tests_per_language)
                test_cases = [(source_lines[i].strip(), target_lines[i].strip()) 
                             for i in test_indices]
                
                all_language_tests[target_file.name] = {
                    'test_cases': test_cases,
                    'target_file': target_file
                }
                
            except Exception as e:
                print(f"⚠️  Error loading {target_file.name}: {e}")
                continue
        
        return all_language_tests

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
        
        # Get test cases for all languages
        all_language_tests = self.get_test_cases_per_language(num_tests)
        
        if not all_language_tests:
            print("❌ No valid target languages found!")
            return None
        
        print(f"🔄 Context Corrigibility Benchmark")
        print(f"Model: {self.model}")
        print(f"Query method: {self.query_method}")
        print(f"Target languages: {len(all_language_tests)} languages")
        print(f"Tests per language: {num_tests}")
        print(f"Example counts: {example_counts}")
        print(f"Languages: {', '.join(all_language_tests.keys())}")
        print()
        
        # Results structure: {language: {count: [scores]}}
        results_by_language = {}
        detailed_results = []
        
        total_tests = len(all_language_tests) * num_tests * len(example_counts)
        
        with tqdm(total=total_tests, desc="Testing context corrigibility across languages") as pbar:
            for lang_name, lang_data in all_language_tests.items():
                target_file = lang_data['target_file']
                test_cases = lang_data['test_cases']
                
                # Initialize results for this language
                results_by_language[lang_name] = {count: [] for count in example_counts}
                
                # Create query object for this language pair
                query_obj = Query(str(self.source_file), str(target_file), method=self.query_method)
                
                for source_text, reference in test_cases:
                    test_result = {
                        "language": lang_name,
                        "source": source_text,
                        "reference": reference,
                        "results": {}
                    }
                    
                    for count in example_counts:
                        examples = self.get_examples(query_obj, source_text, count) if count > 0 else []
                        translation = self.translate(source_text, examples)
                        scores = self.evaluate_translation(translation, reference)
                        
                        results_by_language[lang_name][count].append(scores)
                        test_result["results"][count] = {
                            "translation": translation,
                            "scores": scores,
                            "num_examples": len(examples)
                        }
                        
                        pbar.update(1)
                    
                    detailed_results.append(test_result)
        
        self.print_results(results_by_language, example_counts)
        
        # Create aggregated summary statistics
        summary_stats = self.compute_summary_stats(results_by_language, example_counts)
        
        output_data = {
            "benchmark": "context_corrigibility",
            "model": self.model,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "languages_tested": list(all_language_tests.keys()),
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(results_by_language, detailed_results, example_counts, output_file)
        
        return output_data

    def compute_summary_stats(self, results_by_language, example_counts):
        """Compute both per-language and overall aggregated statistics"""
        summary_stats = {
            "per_language": {},
            "overall": {}
        }
        
        # Per-language stats
        for lang_name, lang_results in results_by_language.items():
            summary_stats["per_language"][lang_name] = {}
            for count in example_counts:
                chrf_scores = [r["chrf"] for r in lang_results[count]]
                edit_scores = [r["edit"] for r in lang_results[count]]
                
                summary_stats["per_language"][lang_name][count] = {
                    "chrf_mean": mean(chrf_scores),
                    "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                    "edit_mean": mean(edit_scores),
                    "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                    "num_tests": len(chrf_scores)
                }
        
        # Overall aggregated stats (combining all languages)
        for count in example_counts:
            all_chrf_scores = []
            all_edit_scores = []
            
            for lang_results in results_by_language.values():
                all_chrf_scores.extend([r["chrf"] for r in lang_results[count]])
                all_edit_scores.extend([r["edit"] for r in lang_results[count]])
            
            summary_stats["overall"][count] = {
                "chrf_mean": mean(all_chrf_scores),
                "chrf_std": stdev(all_chrf_scores) if len(all_chrf_scores) > 1 else 0.0,
                "edit_mean": mean(all_edit_scores),
                "edit_std": stdev(all_edit_scores) if len(all_edit_scores) > 1 else 0.0,
                "num_tests": len(all_chrf_scores),
                "num_languages": len(results_by_language)
            }
        
        return summary_stats

    def print_results(self, results_by_language, example_counts):
        print(f"\n{'='*70}")
        print("CONTEXT CORRIGIBILITY RESULTS - ALL LANGUAGES")
        print(f"{'='*70}")
        
        # Print per-language results
        for lang_name, lang_results in results_by_language.items():
            print(f"\n📍 {lang_name}:")
            for count in example_counts:
                chrf_scores = [r["chrf"] for r in lang_results[count]]
                edit_scores = [r["edit"] for r in lang_results[count]]
                
                print(f"  {count} examples: chrF+ {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}, "
                      f"Edit {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
        
        # Print overall aggregated results
        print(f"\n🌍 OVERALL AGGREGATED RESULTS:")
        print("-" * 40)
        
        overall_results = {}
        for count in example_counts:
            all_chrf = []
            all_edit = []
            for lang_results in results_by_language.values():
                all_chrf.extend([r["chrf"] for r in lang_results[count]])
                all_edit.extend([r["edit"] for r in lang_results[count]])
            
            overall_results[count] = {"chrf": all_chrf, "edit": all_edit}
            print(f"{count} examples: chrF+ {mean(all_chrf):.3f}±{stdev(all_chrf):.3f}, "
                  f"Edit {mean(all_edit):.3f}±{stdev(all_edit):.3f} "
                  f"({len(all_chrf)} tests across {len(results_by_language)} languages)")
        
        # Print corrigibility analysis
        print(f"\n📈 CORRIGIBILITY ANALYSIS (Overall):")
        print("-" * 35)
        baseline_scores = overall_results[0]["chrf"]
        for count in example_counts[1:]:
            context_scores = overall_results[count]["chrf"]
            improvement = mean(context_scores) - mean(baseline_scores)
            print(f"{count} examples: {improvement:+.3f} chrF+ improvement")

    def save_results(self, results_by_language, detailed_results, example_counts, output_file):
        # Compute summary statistics
        summary_stats = self.compute_summary_stats(results_by_language, example_counts)
        
        output_data = {
            "benchmark": "context_corrigibility",
            "model": self.model,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "languages_tested": list(results_by_language.keys()),
            "summary": summary_stats,
            "detailed_results": detailed_results
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