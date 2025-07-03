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

import sys
sys.path.append(str(Path(__file__).parent.parent))
from metrics import chrF_plus, normalized_edit_distance
from benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt


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
        test_cases = self.get_test_cases(num_tests)
        target_lang = test_cases[0][2].split('-')[0] if test_cases else "target language"
        
        print(f"💪 Power Prompt Benchmark")
        print(f"Model: {self.model}")
        print(f"Target Language: {target_lang}")
        print(f"Tests per prompt: {len(test_cases)}")
        print(f"Prompt styles: {len(self.prompt_templates)}")
        print()
        
        results = {prompt_name: [] for prompt_name in self.prompt_templates.keys()}
        detailed_results = []
        
        total_tests = len(test_cases) * len(self.prompt_templates)
        
        with tqdm(total=total_tests, desc="Testing prompts") as pbar:
            for source_text, reference, _ in test_cases:
                test_result = {
                    "source": source_text,
                    "reference": reference,
                    "translations": {},
                    "scores": {}
                }
                
                for prompt_name, prompt_template in self.prompt_templates.items():
                    translation = self.translate_with_prompt(source_text, prompt_template)
                    scores = self.evaluate_translation(translation, reference)
                    
                    results[prompt_name].append(scores)
                    test_result["translations"][prompt_name] = translation
                    test_result["scores"][prompt_name] = scores
                    
                    pbar.update(1)
                
                detailed_results.append(test_result)
        
        self.print_results(results)
        
        # Create the data structure (same as what gets saved to JSON)
        summary_stats = {}
        for prompt_name in self.prompt_templates.keys():
            chrf_scores = [r["chrf"] for r in results[prompt_name]]
            edit_scores = [r["edit"] for r in results[prompt_name]]
            
            summary_stats[prompt_name] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                "overall": mean([mean(chrf_scores), mean(edit_scores)])
            }
        
        output_data = {
            "benchmark": "power_prompt",
            "model": self.model,
            "prompt_templates": self.prompt_templates,
            "summary": summary_stats,
            "detailed_results": detailed_results
        }
        
        if output_file:
            self.save_results(results, detailed_results, output_file)
        
        return output_data

    def print_results(self, results):
        print(f"\n{'='*60}")
        print("POWER PROMPT RESULTS")
        print(f"{'='*60}")
        
        prompt_scores = {}
        for prompt_name in self.prompt_templates.keys():
            chrf_scores = [r["chrf"] for r in results[prompt_name]]
            edit_scores = [r["edit"] for r in results[prompt_name]]
            
            avg_score = mean([mean(chrf_scores), mean(edit_scores)])
            prompt_scores[prompt_name] = avg_score
            
            print(f"\n{prompt_name.upper()}:")
            print(f"  chrF+: {mean(chrf_scores):.3f}±{stdev(chrf_scores):.3f}")
            print(f"  Edit: {mean(edit_scores):.3f}±{stdev(edit_scores):.3f}")
            print(f"  Overall: {avg_score:.3f}")
        
        print(f"\nRANKING:")
        print("-" * 15)
        ranked_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (prompt_name, score) in enumerate(ranked_prompts):
            if i == 0:
                print(f"{prompt_name:<12}: {score:.3f} ⭐")
            else:
                diff = score - ranked_prompts[0][1]
                print(f"{prompt_name:<12}: {score:.3f} ({diff:+.3f})")

    def save_results(self, results, detailed_results, output_file):
        output_data = {
            "benchmark": "power_prompt",
            "model": self.model,
            "prompt_templates": self.prompt_templates,
            "summary": {},
            "detailed_results": detailed_results
        }
        
        for prompt_name in self.prompt_templates.keys():
            chrf_scores = [r["chrf"] for r in results[prompt_name]]
            edit_scores = [r["edit"] for r in results[prompt_name]]
            
            output_data["summary"][prompt_name] = {
                "chrf_mean": mean(chrf_scores),
                "chrf_std": stdev(chrf_scores) if len(chrf_scores) > 1 else 0.0,
                "edit_mean": mean(edit_scores),
                "edit_std": stdev(edit_scores) if len(edit_scores) > 1 else 0.0,
                "overall": mean([mean(chrf_scores), mean(edit_scores)])
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