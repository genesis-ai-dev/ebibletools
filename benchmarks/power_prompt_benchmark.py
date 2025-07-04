#!/usr/bin/env python3
"""
Power Prompt Benchmark - Tests effectiveness of different prompt styles
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


class PowerPromptBenchmark:
    def __init__(self, corpus_dir, source_file, model="gpt-4o"):
        self.model = model
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        
        self.target_files = [f for f in self.corpus_dir.glob('*.txt') if f != self.source_file]
        
        # Simplified to just 4 key prompt styles
        self.prompt_templates = {
            "basic": "Translate this text: {text}",
            "expert": "You are an expert translator. Translate this text: {text}",
            "biblical": "You are a biblical scholar. Translate this biblical text: {text}",
            "direct": "{text}"
        }

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

    def translate_with_prompt(self, text, prompt_template):
        base_prompt = prompt_template.format(text=text)
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

    def run_benchmark(self, num_tests=12, output_file=None):
        # Get test cases for all languages
        all_language_tests = self.get_test_cases_per_language(num_tests)
        
        if not all_language_tests:
            print("❌ No valid target languages found!")
            return None
        
        print(f"💪 Power Prompt Benchmark")
        print(f"Model: {self.model}")
        print(f"Target languages: {len(all_language_tests)} languages")
        print(f"Tests per language: {num_tests}")
        print(f"Prompt styles: {len(self.prompt_templates)}")
        print(f"Languages: {', '.join(all_language_tests.keys())}")
        print()
        
        # Results structure: {language: {prompt_name: [scores]}}
        results_by_language = {}
        detailed_results = []
        
        total_tests = len(all_language_tests) * num_tests * len(self.prompt_templates)
        
        with tqdm(total=total_tests, desc="Testing prompts across languages") as pbar:
            for lang_name, lang_data in all_language_tests.items():
                test_cases = lang_data['test_cases']
                
                # Initialize results for this language
                results_by_language[lang_name] = {prompt_name: [] for prompt_name in self.prompt_templates.keys()}
                
                for source_text, reference in test_cases:
                    test_result = {
                        "language": lang_name,
                        "source": source_text,
                        "reference": reference,
                        "translations": {},
                        "scores": {}
                    }
                    
                    for prompt_name, prompt_template in self.prompt_templates.items():
                        translation = self.translate_with_prompt(source_text, prompt_template)
                        scores = self.evaluate_translation(translation, reference)
                        
                        results_by_language[lang_name][prompt_name].append(scores)
                        test_result["translations"][prompt_name] = translation
                        test_result["scores"][prompt_name] = scores
                        
                        pbar.update(1)
                    
                    detailed_results.append(test_result)
        
        self.print_results(results_by_language)
        
        # Create aggregated summary statistics
        summary_stats = self.compute_summary_stats(results_by_language)
        
        output_data = {
            "benchmark": "power_prompt",
            "model": self.model,
            "prompt_templates": self.prompt_templates,
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
        
        # Per-language stats
        for lang_name, lang_results in results_by_language.items():
            summary_stats["per_language"][lang_name] = {}
            for prompt_name in self.prompt_templates.keys():
                chrf_scores = [r["chrf"] for r in lang_results[prompt_name]]
                edit_scores = [r["edit"] for r in lang_results[prompt_name]]
                
                summary_stats["per_language"][lang_name][prompt_name] = {
                    "chrf_mean": mean(chrf_scores),
                    "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                    "edit_mean": mean(edit_scores),
                    "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                    "overall": mean([mean(chrf_scores), mean(edit_scores)]),
                    "num_tests": len(chrf_scores)
                }
        
        # Overall aggregated stats (combining all languages)
        for prompt_name in self.prompt_templates.keys():
            all_chrf_scores = []
            all_edit_scores = []
            
            for lang_results in results_by_language.values():
                all_chrf_scores.extend([r["chrf"] for r in lang_results[prompt_name]])
                all_edit_scores.extend([r["edit"] for r in lang_results[prompt_name]])
            
            summary_stats["overall"][prompt_name] = {
                "chrf_mean": mean(all_chrf_scores),
                "chrf_std": stdev(all_chrf_scores) if len(all_chrf_scores) > 1 else 0.0,
                "edit_mean": mean(all_edit_scores),
                "edit_std": stdev(all_edit_scores) if len(all_edit_scores) > 1 else 0.0,
                "overall": mean([mean(all_chrf_scores), mean(all_edit_scores)]),
                "num_tests": len(all_chrf_scores),
                "num_languages": len(results_by_language)
            }
        
        return summary_stats

    def print_results(self, results_by_language):
        print(f"\n{'='*70}")
        print("POWER PROMPT RESULTS - ALL LANGUAGES")
        print(f"{'='*70}")
        
        # Print per-language results
        for lang_name, lang_results in results_by_language.items():
            print(f"\n📍 {lang_name}:")
            
            lang_prompt_scores = {}
            for prompt_name in self.prompt_templates.keys():
                chrf_scores = [r["chrf"] for r in lang_results[prompt_name]]
                edit_scores = [r["edit"] for r in lang_results[prompt_name]]
                
                avg_score = mean([mean(chrf_scores), mean(edit_scores)])
                lang_prompt_scores[prompt_name] = avg_score
                
                print(f"  {prompt_name.upper()}: chrF+ {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}, "
                      f"Edit {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}, "
                      f"Overall {avg_score:.3f}")
            
            # Show best prompt for this language
            best_prompt = max(lang_prompt_scores.items(), key=lambda x: x[1])
            print(f"  → Best: {best_prompt[0]} ({best_prompt[1]:.3f})")
        
        # Print overall aggregated results
        print(f"\n🌍 OVERALL AGGREGATED RESULTS:")
        print("-" * 40)
        
        overall_prompt_scores = {}
        for prompt_name in self.prompt_templates.keys():
            all_chrf = []
            all_edit = []
            for lang_results in results_by_language.values():
                all_chrf.extend([r["chrf"] for r in lang_results[prompt_name]])
                all_edit.extend([r["edit"] for r in lang_results[prompt_name]])
            
            avg_score = mean([mean(all_chrf), mean(all_edit)])
            overall_prompt_scores[prompt_name] = avg_score
            
            print(f"{prompt_name.upper()}: chrF+ {mean(all_chrf):.3f}±{stdev(all_chrf):.3f}, "
                  f"Edit {mean(all_edit):.3f}±{stdev(all_edit):.3f}, "
                  f"Overall {avg_score:.3f} "
                  f"({len(all_chrf)} tests across {len(results_by_language)} languages)")
        
        # Print overall ranking
        print(f"\n📈 OVERALL RANKING:")
        print("-" * 20)
        ranked_prompts = sorted(overall_prompt_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (prompt_name, score) in enumerate(ranked_prompts):
            if i == 0:
                print(f"{prompt_name:<12}: {score:.3f} ⭐")
            else:
                diff = score - ranked_prompts[0][1]
                print(f"{prompt_name:<12}: {score:.3f} ({diff:+.3f})")

    def save_results(self, results_by_language, detailed_results, output_file):
        # Compute summary statistics
        summary_stats = self.compute_summary_stats(results_by_language)
        
        output_data = {
            "benchmark": "power_prompt",
            "model": self.model,
            "prompt_templates": self.prompt_templates,
            "languages_tested": list(results_by_language.keys()),
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Power Prompt Benchmark")
    parser.add_argument("--corpus-dir", default="../Corpus")
    parser.add_argument("--source-file", default="eng-engULB.txt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--num-tests", type=int, default=12)
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    benchmark = PowerPromptBenchmark(args.corpus_dir, args.source_file, args.model)
    benchmark.run_benchmark(args.num_tests, args.output)
    print("\n✅ Power prompt benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 