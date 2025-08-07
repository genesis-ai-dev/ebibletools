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

from metrics import chrF_plus, normalized_edit_distance
from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt


class PowerPromptBenchmark:
    def __init__(self, corpus_dir, source_file, models=None, custom_prompts=None):
        self.models = models if isinstance(models, list) else [models] if models else ["gpt-4o"]
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        
        self.target_files = [f for f in self.corpus_dir.glob('*.txt') if f != self.source_file]
        
        # Prompts must be provided - no hardcoded defaults
        if not custom_prompts:
            raise ValueError("No prompts provided. Use --prompts, --prompts-file, or provide custom_prompts parameter.")
        
        self.prompt_templates = custom_prompts

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

    def translate_with_prompt(self, text, prompt_template, model):
        # Use system prompt for the role/instruction, user prompt for the text
        system_prompt = prompt_template
        user_prompt = format_xml_prompt(f"Translate this text: {text}", "translation", "your translation here")
        
        completion_args = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        if not model.startswith("gpt-5"):
            completion_args["temperature"] = 0.1
        
        response = litellm.completion(**completion_args)
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
        print(f"Models: {', '.join(self.models)}")
        print(f"Target languages: {len(all_language_tests)} languages")
        print(f"Tests per language: {num_tests}")
        print(f"Prompt styles: {len(self.prompt_templates)}")
        print(f"Languages: {', '.join(all_language_tests.keys())}")
        print()
        
        # Store results for each model
        all_results = {}
        detailed_results = {}
        
        for model in self.models:
            print(f"\n🤖 Testing model: {model}")
            
            # Results structure: {language: {prompt_name: [scores]}}
            results_by_language = {}
            model_detailed_results = []
        
            total_tests = len(all_language_tests) * num_tests * len(self.prompt_templates)
            
            with tqdm(total=total_tests, desc=f"Testing {model}") as pbar:
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
                            translation = self.translate_with_prompt(source_text, prompt_template, model)
                            scores = self.evaluate_translation(translation, reference)
                            
                            results_by_language[lang_name][prompt_name].append(scores)
                            test_result["translations"][prompt_name] = translation
                            test_result["scores"][prompt_name] = scores
                            
                            pbar.update(1)
                        
                        model_detailed_results.append(test_result)
            
            # Store results for this model
            all_results[model] = results_by_language
            detailed_results[model] = model_detailed_results
        
        # Print results for all models
        self.print_results(all_results)
        
        # Create aggregated summary statistics for all models
        all_summary_stats = {}
        for model, results_by_language in all_results.items():
            all_summary_stats[model] = self.compute_summary_stats(results_by_language)
        
        output_data = {
            "benchmark": "power_prompt",
            "models": self.models,
            "prompt_templates": self.prompt_templates,
            "languages_tested": list(all_language_tests.keys()),
            "summary": all_summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(all_results, detailed_results, output_file)
        
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

    def print_results(self, all_results):
        print(f"\n{'='*70}")
        print("POWER PROMPT RESULTS - ALL MODELS")
        print(f"{'='*70}")
        
        # Print results for each model
        for model, results_by_language in all_results.items():
            print(f"\n🤖 MODEL: {model}")
            print("-" * 50)
            
            # Print per-language results for this model
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
            
            # Print overall aggregated results for this model
            print(f"\n🌍 OVERALL RESULTS FOR {model}:")
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
            
            # Print overall ranking for this model
            print(f"\n📈 RANKING FOR {model}:")
            print("-" * 20)
            ranked_prompts = sorted(overall_prompt_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (prompt_name, score) in enumerate(ranked_prompts):
                if i == 0:
                    print(f"{prompt_name:<12}: {score:.3f} ⭐")
                else:
                    diff = score - ranked_prompts[0][1]
                    print(f"{prompt_name:<12}: {score:.3f} ({diff:+.3f})")

    def save_results(self, all_results, detailed_results, output_file):
        # Compute summary statistics for all models
        all_summary_stats = {}
        for model, results_by_language in all_results.items():
            all_summary_stats[model] = self.compute_summary_stats(results_by_language)
        
        # Get languages tested from first model
        first_model_results = next(iter(all_results.values()))
        
        output_data = {
            "benchmark": "power_prompt",
            "models": self.models,
            "prompt_templates": self.prompt_templates,
            "languages_tested": list(first_model_results.keys()),
            "summary": all_summary_stats,
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


def load_prompts_from_json(json_file):
    """Load custom system prompts from a JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        # System prompts don't need {text} placeholder - that goes in user message
        return prompts
    except Exception as e:
        print(f"❌ Error loading prompts from {json_file}: {e}")
        return None


def parse_prompt_args(prompt_args):
    """Parse system prompts from command line arguments in format name:template"""
    prompts = {}
    for arg in prompt_args:
        if ":" not in arg:
            print(f"❌ Invalid prompt format: {arg}. Use 'name:system_prompt' format.")
            continue
        
        name, template = arg.split(":", 1)
        prompts[name] = template
    
    return prompts


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Power Prompt Benchmark - Test different prompt styles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use example prompts from file
  python power_prompt_benchmark.py --model gpt-4o --num-tests 10 --keep-defaults
  
  # Use custom prompts from JSON file
  python power_prompt_benchmark.py --prompts-file my_prompts.json
  
  # Use custom system prompts from command line
  python power_prompt_benchmark.py --prompts "simple:You are a simple translator." "formal:You are a formal, academic translator."
  
  # Mix example prompts with additional custom ones
  python power_prompt_benchmark.py --prompts "custom:You are a specialized biblical translator." --keep-defaults
  
JSON file format:
  {
    "prompt_name": "System prompt describing the role/style",
    "another_prompt": "Another system prompt"
  }
        """
    )
    
    # Use project root's Corpus directory
    default_corpus = str(Path(__file__).parent.parent / "Corpus")
    parser.add_argument("--corpus-dir", default=default_corpus, help="Corpus directory path")
    parser.add_argument("--source-file", default="eng-engULB.txt", help="Source file name")
    parser.add_argument("--model", default="gpt-4o", help="Single model to test")
    parser.add_argument("--models", nargs="+", help="Multiple models to compare")
    parser.add_argument("--num-tests", type=int, default=12, help="Number of tests per language")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    # Prompt customization options
    prompt_group = parser.add_argument_group("Prompt Customization")
    prompt_group.add_argument("--prompts-file", type=str, 
                             help="JSON file containing custom prompts")
    prompt_group.add_argument("--prompts", nargs="+", 
                             help="Custom prompts in 'name:template' format")
    prompt_group.add_argument("--keep-defaults", action="store_true",
                             help="Include example prompts from example_prompts.json when using custom ones")
    prompt_group.add_argument("--list-defaults", action="store_true",
                             help="List example prompts from example_prompts.json and exit")
    
    args = parser.parse_args()
    
    # Handle --list-defaults (show example prompts from file)
    if args.list_defaults:
        example_file = Path(__file__).parent / "example_prompts.json"
        if example_file.exists():
            example_prompts = load_prompts_from_json(example_file)
            if example_prompts:
                print("🔖 Example system prompts (from example_prompts.json):")
                for name, template in example_prompts.items():
                    if template:
                        print(f"  {name}: {template}")
                    else:
                        print(f"  {name}: (no system prompt)")
            else:
                print("❌ Could not load example prompts")
        else:
            print("❌ No example_prompts.json file found")
        return 0
    
    # Determine which prompts to use
    custom_prompts = None
    
    if args.prompts_file:
        print(f"📁 Loading prompts from: {args.prompts_file}")
        custom_prompts = load_prompts_from_json(args.prompts_file)
        if not custom_prompts:
            return 1
    
    if args.prompts:
        print(f"⚙️  Loading prompts from command line")
        cmd_prompts = parse_prompt_args(args.prompts)
        if custom_prompts and args.keep_defaults:
            custom_prompts.update(cmd_prompts)
        elif custom_prompts:
            custom_prompts.update(cmd_prompts)
        else:
            custom_prompts = cmd_prompts
    
    # If keeping defaults, merge with example prompts from file
    if args.keep_defaults and custom_prompts:
        example_file = Path(__file__).parent / "example_prompts.json"
        if example_file.exists():
            example_prompts = load_prompts_from_json(example_file)
            if example_prompts:
                example_prompts.update(custom_prompts)
                custom_prompts = example_prompts
            else:
                print("⚠️  Could not load example prompts, using only custom prompts")
        else:
            print("⚠️  No example_prompts.json file found, using only custom prompts")
    
    # Validate that prompts were provided
    if not custom_prompts:
        print("❌ No prompts provided! You must specify prompts using:")
        print("   --prompts 'name:system_prompt' [additional prompts...]")
        print("   --prompts-file path/to/prompts.json")
        print("   --keep-defaults (to use example_prompts.json)")
        print("\nUse --list-defaults to see example prompts")
        return 1
    
    # Print selected prompts
    print(f"✨ Using system prompts:")
    for name, template in custom_prompts.items():
        if template:
            print(f"  {name}: {template}")
        else:
            print(f"  {name}: (no system prompt)")
    print()
    
    # Handle both --model and --models arguments
    models = args.models if args.models else [args.model]
    
    benchmark = PowerPromptBenchmark(args.corpus_dir, args.source_file, models, custom_prompts)
    benchmark.run_benchmark(args.num_tests, args.output)
    print("\n✅ Power prompt benchmark completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 