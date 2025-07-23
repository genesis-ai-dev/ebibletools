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

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from query import Query
from metrics import chrF_plus, normalized_edit_distance
from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt


class ContextCorrigibilityBenchmark:
    def __init__(self, corpus_dir, source_file, models=None, query_method="context", verbose=False):
        self.models = models if isinstance(models, list) else [models] if models else ["gpt-4o"]
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        self.query_method = query_method
        self.verbose = verbose
        
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

    def translate(self, text, examples=None, model=None, target_language=None):
        if examples:
            base_prompt = "Translate from source to target language. Examples:\n\n"
            for src, tgt in examples:
                base_prompt += f"Source: {src}\nTarget: {tgt}\n\n"
            base_prompt += f"Now translate:\nSource: {text}"
        else:
            if target_language:
                base_prompt = f"The translation project you are translating into is called \"{target_language}\". Translate this text: {text}"
            else:
                base_prompt = f"Translate this text: {text}"
        
        prompt = format_xml_prompt(base_prompt, "translation", "your translation here")
        
        if self.verbose:
            print(f"\n🔍 VERBOSE - Model: {model}")
            print(f"📥 INPUT ({len(examples) if examples else 0} examples):")
            print(f"   Source: {text}")
            if examples:
                print(f"   Examples:")
                for i, (src, tgt) in enumerate(examples[:3], 1):  # Show first 3 examples
                    print(f"     {i}. {src[:50]}{'...' if len(src) > 50 else ''} → {tgt[:50]}{'...' if len(tgt) > 50 else ''}")
                if len(examples) > 3:
                    print(f"     ... and {len(examples) - 3} more examples")
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.1
        )
        
        translation = extract_xml_content(response.choices[0].message.content.strip(), "translation")
        
        if self.verbose:
            print(f"📤 OUTPUT:")
            print(f"   Translation: {translation}")
            print(f"   Raw response: {response.choices[0].message.content.strip()[:100]}{'...' if len(response.choices[0].message.content.strip()) > 100 else ''}")
        
        return translation

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
        print(f"Models: {', '.join(self.models)}")
        print(f"Query method: {self.query_method}")
        print(f"Target languages: {len(all_language_tests)} languages")
        print(f"Tests per language: {num_tests}")
        print(f"Example counts: {example_counts}")
        print(f"Languages: {', '.join(all_language_tests.keys())}")
        print()
        
        # Store results for each model
        all_results = {}
        detailed_results = {}
        
        for model in self.models:
            print(f"\n🤖 Testing model: {model}")
            
            # Results structure: {language: {count: [scores]}}
            results_by_language = {}
            model_detailed_results = []
        
            total_tests = len(all_language_tests) * num_tests * len(example_counts)
            
            with tqdm(total=total_tests, desc=f"Testing {model}") as pbar:
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
                            target_language_name = target_file.stem  # Remove extension from filename
                            translation = self.translate(source_text, examples, model, target_language_name)
                            scores = self.evaluate_translation(translation, reference)
                            
                            results_by_language[lang_name][count].append(scores)
                            test_result["results"][count] = {
                                "translation": translation,
                                "scores": scores,
                                "num_examples": len(examples)
                            }
                            
                            pbar.update(1)
                        
                        model_detailed_results.append(test_result)
            
            # Store results for this model
            all_results[model] = results_by_language
            detailed_results[model] = model_detailed_results
        
        # Print results for all models
        self.print_results(all_results, example_counts)
        
        # Create aggregated summary statistics for all models
        all_summary_stats = {}
        for model, results_by_language in all_results.items():
            all_summary_stats[model] = self.compute_summary_stats(results_by_language, example_counts)
        
        output_data = {
            "benchmark": "context_corrigibility",
            "models": self.models,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "languages_tested": list(all_language_tests.keys()),
            "summary": all_summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(all_results, detailed_results, example_counts, output_file)
        
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

    def print_results(self, all_results, example_counts):
        print(f"\n{'='*70}")
        print("CONTEXT CORRIGIBILITY RESULTS - ALL MODELS")
        print(f"{'='*70}")
        
        # Print results for each model
        for model, results_by_language in all_results.items():
            print(f"\n🤖 MODEL: {model}")
            print("-" * 50)
            
            # Print per-language results for this model
            for lang_name, lang_results in results_by_language.items():
                print(f"\n📍 {lang_name}:")
                for count in example_counts:
                    chrf_scores = [r["chrf"] for r in lang_results[count]]
                    edit_scores = [r["edit"] for r in lang_results[count]]
                    
                    print(f"  {count} examples: chrF+ {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}, "
                          f"Edit {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
            
            # Print overall aggregated results for this model
            print(f"\n🌍 OVERALL RESULTS FOR {model}:")
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
            
            # Print corrigibility analysis for this model
            print(f"\n📈 CORRIGIBILITY ANALYSIS FOR {model}:")
            print("-" * 35)
            if len(example_counts) > 1:
                baseline_count = example_counts[0]
                baseline_scores = overall_results[baseline_count]["chrf"]
                for count in example_counts[1:]:
                    context_scores = overall_results[count]["chrf"]
                    improvement = mean(context_scores) - mean(baseline_scores)
                    print(f"{count} examples vs {baseline_count}: {improvement:+.3f} chrF+ improvement")
            else:
                print("Need at least 2 example counts to show improvement analysis")

    def save_results(self, all_results, detailed_results, example_counts, output_file):
        # Compute summary statistics for all models
        all_summary_stats = {}
        for model, results_by_language in all_results.items():
            all_summary_stats[model] = self.compute_summary_stats(results_by_language, example_counts)
        
        # Get languages tested from first model
        first_model_results = next(iter(all_results.values()))
        
        output_data = {
            "benchmark": "context_corrigibility",
            "models": self.models,
            "query_method": self.query_method,
            "example_counts": example_counts,
            "languages_tested": list(first_model_results.keys()),
            "summary": all_summary_stats,
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Context Corrigibility & Translation Benchmark - Tests how adding examples affects translation quality")
    # Use project root's Corpus directory
    default_corpus = str(Path(__file__).parent.parent / "Corpus")
    parser.add_argument("--corpus-dir", default=default_corpus)
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o", help="Single model to test")
    parser.add_argument("--models", nargs="+", help="Multiple models to compare")
    parser.add_argument("--query-method", default="context", choices=["bm25", "tfidf", "context"])
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--example-counts", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", action="store_true", help="Show detailed model inputs and outputs")
    
    args = parser.parse_args()
    
    # Handle both --model and --models arguments
    models = args.models if args.models else [args.model]
    
    benchmark = ContextCorrigibilityBenchmark(
        args.corpus_dir, args.source_file, models, args.query_method, args.verbose
    )
    benchmark.run_benchmark(args.num_tests, args.example_counts, args.output)
    print("\n✅ Context corrigibility benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 