#!/usr/bin/env python3
"""
True Source Benchmark - Tests how having source text affects translation accuracy
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

try:
    from metrics import chrF_plus, normalized_edit_distance
    from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    raise


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
                
                # Need more indices to get both examples and test cases
                total_needed = num_tests_per_language + 6  # 3 for examples, rest for tests
                if len(valid_indices) < total_needed:
                    print(f"⚠️  Skipping {target_file.name} - only {len(valid_indices)} valid lines (need {total_needed})")
                    continue
                
                selected_indices = random.sample(valid_indices, total_needed)
                
                # First 3 pairs are ICL examples, rest are test cases
                example_pairs = [(source_lines[i].strip(), target_lines[i].strip()) 
                                for i in selected_indices[:3]]
                test_cases = [(source_lines[i].strip(), target_lines[i].strip()) 
                             for i in selected_indices[3:3+num_tests_per_language]]
                
                all_language_tests[target_file.name] = {
                    'example_pairs': example_pairs,
                    'test_cases': test_cases,
                    'target_file': target_file
                }
                
            except Exception as e:
                print(f"⚠️  Error loading {target_file.name}: {e}")
                continue
        
        return all_language_tests

    def translate_with_source_examples(self, test_source, example_pairs, target_lang):
        # Build prompt with source-target example pairs
        base_prompt = f"Translate from source to {target_lang}. Examples:\n\n"
        for src, tgt in example_pairs:
            base_prompt += f"Source: {src}\nTarget: {tgt}\n\n"
        base_prompt += f"Now translate:\nSource: {test_source}"
        
        prompt = format_xml_prompt(base_prompt, "translation", "your translation here")
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return extract_xml_content(response.choices[0].message.content.strip(), "translation")

    def translate_with_target_only_examples(self, test_source, example_pairs, target_lang):
        # Build prompt with only target examples (no source)
        base_prompt = f"Translate to {target_lang}. Example target language texts:\n\n"
        for _, tgt in example_pairs:
            base_prompt += f"{tgt}\n"
        base_prompt += f"\nNow translate this to {target_lang}:\n{test_source}"
        
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
        # Get test cases for all languages
        all_language_tests = self.get_test_cases_per_language(num_tests)
        
        if not all_language_tests:
            print("❌ No valid target languages found!")
            return None
        
        print(f"🎯 True Source Benchmark")
        print(f"Model: {self.model}")
        print(f"Target languages: {len(all_language_tests)} languages")
        print(f"Tests per language: {num_tests}")
        print(f"ICL Examples per language: 3")
        print(f"Languages: {', '.join(all_language_tests.keys())}")
        print()
        
        # Results structure: {language: {condition: [scores]}}
        results_by_language = {}
        detailed_results = []
        
        total_tests = len(all_language_tests) * num_tests * 2  # 2 conditions per test
        
        with tqdm(total=total_tests, desc="Testing ICL source effects across languages") as pbar:
            for lang_name, lang_data in all_language_tests.items():
                target_lang = lang_name.split('-')[0] if '-' in lang_name else "target language"
                example_pairs = lang_data['example_pairs']
                test_cases = lang_data['test_cases']
                
                # Initialize results for this language
                results_by_language[lang_name] = {"with_source": [], "without_source": []}
                
                for source_text, reference in test_cases:
                    # Test 1: With source-target example pairs
                    trans_with_source = self.translate_with_source_examples(source_text, example_pairs, target_lang)
                    scores_with = self.evaluate_translation(trans_with_source, reference)
                    
                    # Test 2: With target-only examples  
                    trans_without_source = self.translate_with_target_only_examples(source_text, example_pairs, target_lang)
                    scores_without = self.evaluate_translation(trans_without_source, reference)
                    
                    results_by_language[lang_name]["with_source"].append(scores_with)
                    results_by_language[lang_name]["without_source"].append(scores_without)
                    
                    detailed_results.append({
                        "language": lang_name,
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
                    
                    pbar.update(2)  # 2 conditions tested
        
        self.print_results(results_by_language)
        
        # Create aggregated summary statistics
        summary_stats = self.compute_summary_stats(results_by_language)
        
        output_data = {
            "benchmark": "true_source",
            "model": self.model,
            "languages_tested": list(all_language_tests.keys()),
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(results_by_language, detailed_results, output_file)
        
        return output_data

    def compute_summary_stats(self, results_by_language):
        """Compute both per-language and overall aggregated statistics"""
        summary_stats = {
            "per_language": {},
            "overall": {}
        }
        
        conditions = ["with_source", "without_source"]
        
        # Per-language stats
        for lang_name, lang_results in results_by_language.items():
            summary_stats["per_language"][lang_name] = {}
            for condition in conditions:
                chrf_scores = [r["chrf"] for r in lang_results[condition]]
                edit_scores = [r["edit"] for r in lang_results[condition]]
                
                summary_stats["per_language"][lang_name][condition] = {
                    "chrf_mean": mean(chrf_scores),
                    "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                    "edit_mean": mean(edit_scores),
                    "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                    "num_tests": len(chrf_scores)
                }
        
        # Overall aggregated stats (combining all languages)
        for condition in conditions:
            all_chrf_scores = []
            all_edit_scores = []
            
            for lang_results in results_by_language.values():
                all_chrf_scores.extend([r["chrf"] for r in lang_results[condition]])
                all_edit_scores.extend([r["edit"] for r in lang_results[condition]])
            
            summary_stats["overall"][condition] = {
                "chrf_mean": mean(all_chrf_scores),
                "chrf_std": stdev(all_chrf_scores) if len(all_chrf_scores) > 1 else 0.0,
                "edit_mean": mean(all_edit_scores),
                "edit_std": stdev(all_edit_scores) if len(all_edit_scores) > 1 else 0.0,
                "num_tests": len(all_chrf_scores),
                "num_languages": len(results_by_language)
            }
        
        return summary_stats

    def print_results(self, results_by_language):
        print(f"\n{'='*70}")
        print("TRUE SOURCE RESULTS - ALL LANGUAGES")
        print(f"{'='*70}")
        
        # Print per-language results
        for lang_name, lang_results in results_by_language.items():
            print(f"\n📍 {lang_name}:")
            
            for condition in ["with_source", "without_source"]:
                chrf_scores = [r["chrf"] for r in lang_results[condition]]
                edit_scores = [r["edit"] for r in lang_results[condition]]
                
                condition_display = condition.replace('_', ' ').title()
                print(f"  {condition_display}: chrF+ {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}, "
                      f"Edit {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
            
            # Show improvement for this language
            with_chrf = [r["chrf"] for r in lang_results["with_source"]]
            without_chrf = [r["chrf"] for r in lang_results["without_source"]]
            improvement = mean(with_chrf) - mean(without_chrf)
            print(f"  → Source effect: {improvement:+.3f} chrF+")
        
        # Print overall aggregated results
        print(f"\n🌍 OVERALL AGGREGATED RESULTS:")
        print("-" * 40)
        
        overall_results = {}
        for condition in ["with_source", "without_source"]:
            all_chrf = []
            all_edit = []
            for lang_results in results_by_language.values():
                all_chrf.extend([r["chrf"] for r in lang_results[condition]])
                all_edit.extend([r["edit"] for r in lang_results[condition]])
            
            overall_results[condition] = {"chrf": all_chrf, "edit": all_edit}
            condition_display = condition.replace('_', ' ').title()
            print(f"{condition_display}: chrF+ {mean(all_chrf):.3f}±{stdev(all_chrf):.3f}, "
                  f"Edit {mean(all_edit):.3f}±{stdev(all_edit):.3f} "
                  f"({len(all_chrf)} tests across {len(results_by_language)} languages)")
        
        # Show overall source effect
        print(f"\n📈 OVERALL SOURCE EFFECT:")
        print("-" * 25)
        overall_improvement = mean(overall_results["with_source"]["chrf"]) - mean(overall_results["without_source"]["chrf"])
        print(f"chrF+ improvement with source: {overall_improvement:+.3f}")

    def save_results(self, results_by_language, detailed_results, output_file):
        # Compute summary statistics
        summary_stats = self.compute_summary_stats(results_by_language)
        
        output_data = {
            "benchmark": "true_source",
            "model": self.model,
            "languages_tested": list(results_by_language.keys()),
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="True Source Benchmark")
    # Use project root's Corpus directory
    default_corpus = str(Path(__file__).parent.parent / "Corpus")
    parser.add_argument("--corpus-dir", default=default_corpus)
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