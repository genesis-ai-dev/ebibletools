#!/usr/bin/env python3
"""
Enhanced Translation Benchmark with Comprehensive Metrics Evaluation

Compares different numbers of examples for Query-based in-context learning
using professional translation quality metrics. Uses liteLLM for provider flexibility.
"""

import argparse
import os
import random
import time
import json
from pathlib import Path
from tqdm import tqdm
from statistics import mean, stdev

from dotenv import load_dotenv
import litellm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from query import Query
from metrics import chrF_plus, normalized_edit_distance, ter_score


class TranslationBenchmark:
    def __init__(self, api_key, corpus_dir, source_file, query_method="context", models=None):
        # Set up liteLLM with OpenAI (can be easily changed to other providers)
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.models = models if isinstance(models, list) else [models] if models else ["gpt-4o"]
        self.corpus_dir = Path(corpus_dir)
        self.source_file = self.corpus_dir / source_file
        self.query_method = query_method
        
        # Universal metric names (language-agnostic only)
        self.metric_names = ['chrF+', 'Edit Dist', 'TER']

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

    def get_examples(self, query_obj, query_text, num_examples):
        """Get examples using Query with context-aware search when specified"""
        if self.query_method == "context":
            # Use context-aware search for better semantic matching
            results = query_obj.search_by_context(query_text, top_k=num_examples * 2)
        else:
            # Fallback to basic text search for other methods
            results = query_obj.search_by_text(query_text, top_k=num_examples * 2)
        
        examples = []
        for result in results:
            # Query returns (line_number, source_text, target_text, score)
            line_num, src, tgt, score = result
            if len(src) > 10 and len(tgt) > 10 and src != query_text:
                examples.append((src.strip(), tgt.strip()))
                if len(examples) >= num_examples:
                    break
        
        return examples

    def translate(self, text, examples=None, model=None):
        """Translate using liteLLM with optional examples"""
        if examples:
            prompt = "Translate from source to target language. Examples:\n\n"
            for src, tgt in examples:
                prompt += f"Source: {src}\nTarget: {tgt}\n\n"
            prompt += f"Now translate:\nSource: {text}\nTarget:"
        else:
            prompt = f"Translate this text: {text}"
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def evaluate_translation(self, hypothesis, reference):
        """Universal translation evaluation using language-agnostic metrics only"""
        scores = {}
        
        # chrF+ (character n-gram F-score, works with any script)
        scores['chrF+'] = chrF_plus(hypothesis, reference)
        
        # Edit Distance (normalized Levenshtein, inverted to 0-1 where higher is better)
        scores['Edit Dist'] = 1.0 - normalized_edit_distance(hypothesis, reference)
        
        # TER (Translation Error Rate, inverted to 0-1 where higher is better)
        ter = ter_score(hypothesis, reference)
        scores['TER'] = max(0.0, 1.0 - ter)
        
        return scores

    def run_benchmark(self, num_target_files, num_tests_per_file, example_counts, output_file=None):
        """Run the comprehensive benchmark"""
        target_files = self.get_target_files(num_target_files)
        
        print(f"🌍 Universal Translation Benchmark - {self.query_method.upper()} Query")
        print("=" * 65)
        print(f"Models: {', '.join(self.models)}")
        print(f"Source: {self.source_file.name}")
        print(f"Target files: {len(target_files)} files")
        print(f"Tests per file: {num_tests_per_file}")
        print(f"Example counts: {example_counts}")
        print(f"Total tests: {len(target_files) * num_tests_per_file}")
        print(f"Universal metrics (any language/script): {', '.join(self.metric_names)}")
        print()
        
        # Store results for each model
        all_results = {}
        detailed_results = {}
        
        for model in self.models:
            print(f"\n🤖 Testing model: {model}")
            
            # Results structure: {example_count: {metric: [scores]}}
            results = {count: {metric: [] for metric in self.metric_names} for count in example_counts}
            model_detailed_results = []
        
            total_tests = len(target_files) * num_tests_per_file * len(example_counts)
            progress_bar = tqdm(total=total_tests, desc=f"Testing {model}")
            
            for target_file in target_files:
                source_lines, target_lines = self.load_file_pair(self.source_file, target_file)
                query_obj = Query(str(self.source_file), str(target_file), method=self.query_method)
                
                # Find valid test lines
                valid_indices = [
                    i for i in range(min(len(source_lines), len(target_lines)))
                    if len(source_lines[i].strip()) > 10 and len(target_lines[i].strip()) > 10
                ]
                
                if len(valid_indices) < num_tests_per_file:
                    print(f"⚠️  Skipping {target_file.name} - insufficient valid lines")
                    progress_bar.update(num_tests_per_file * len(example_counts))
                    continue
                
                test_indices = random.sample(valid_indices, num_tests_per_file)
                
                for idx in test_indices:
                    source_text = source_lines[idx].strip()
                    ground_truth = target_lines[idx].strip()
                    
                    for count in example_counts:
                        # Get examples and translate
                        examples = self.get_examples(query_obj, source_text, count)
                        translation = self.translate(source_text, examples, model)
                        
                        # Evaluate with all metrics
                        scores = self.evaluate_translation(translation, ground_truth)
                        
                        # Store results
                        for metric, score in scores.items():
                            results[count][metric].append(score)
                        
                        # Store detailed result
                        model_detailed_results.append({
                            'target_file': target_file.name,
                            'line_index': idx,
                            'source': source_text,
                            'reference': ground_truth,
                            'translation': translation,
                            'example_count': count,
                            'scores': scores
                        })
                        
                        progress_bar.update(1)
                        time.sleep(0.1)  # Rate limiting
            
            progress_bar.close()
            
            # Store results for this model
            all_results[model] = results
            detailed_results[model] = model_detailed_results
        
        # Print and save results for all models
        self.print_results(all_results, example_counts)
        
        if output_file:
            self.save_results(all_results, detailed_results, example_counts, output_file)

    def print_results(self, all_results, example_counts):
        """Print comprehensive benchmark results"""
        print(f"\n{'='*80}")
        print(f"UNIVERSAL TRANSLATION EVALUATION RESULTS - ALL MODELS")
        print(f"Query Method: {self.query_method.upper()}")
        print(f"Language-Agnostic Metrics Only")
        print(f"{'='*80}")
        
        # Print results for each model
        for model, results in all_results.items():
            print(f"\n🤖 MODEL: {model}")
            print("-" * 60)
            
            # Calculate averages and standard deviations
            stats = {}
            for count in example_counts:
                stats[count] = {}
                for metric in self.metric_names:
                    scores = results[count][metric]
                    if scores:
                        stats[count][metric] = {
                            'mean': mean(scores),
                            'std': stdev(scores) if len(scores) > 1 else 0.0,
                            'count': len(scores)
                        }
                    else:
                        stats[count][metric] = {'mean': 0.0, 'std': 0.0, 'count': 0}
            
            # Print results table for this model
            print(f"\n{'Metric':<12} ", end="")
            for count in example_counts:
                print(f"{count:>2} examples    ", end="")
            print()
            print("-" * (12 + len(example_counts) * 14))
            
            for metric in self.metric_names:
                print(f"{metric:<12} ", end="")
                for count in example_counts:
                    mean_score = stats[count][metric]['mean']
                    std_score = stats[count][metric]['std']
                    print(f"{mean_score:.3f}±{std_score:.3f}  ", end="")
                print()
            
            # Find best configuration for each metric for this model
            print(f"\n{'Best Configurations:':<25}")
            print("-" * 25)
            for metric in self.metric_names:
                best_count = max(example_counts, key=lambda c: stats[c][metric]['mean'])
                best_score = stats[best_count][metric]['mean']
                print(f"{metric:<12}: {best_count} examples ({best_score:.3f})")
            
            # Overall ranking by average across metrics for this model
            print(f"\n{'Overall Ranking:':<20}")
            print("-" * 20)
            overall_scores = {}
            for count in example_counts:
                # Average across all metrics
                all_means = [stats[count][metric]['mean'] for metric in self.metric_names]
                overall_scores[count] = mean(all_means)
            
            ranked_counts = sorted(overall_scores.keys(), key=lambda c: overall_scores[c], reverse=True)
            for i, count in enumerate(ranked_counts):
                score = overall_scores[count]
                if i == 0:
                    print(f"{count:>2} examples: {score:.3f} ⭐ (best overall)")
                else:
                    diff = score - overall_scores[ranked_counts[0]]
                    print(f"{count:>2} examples: {score:.3f} ({diff:+.3f})")

    def save_results(self, all_results, detailed_results, example_counts, output_file):
        """Save detailed results to JSON file"""
        output_data = {
            'benchmark_config': {
                'query_method': self.query_method,
                'source_file': self.source_file.name,
                'models': self.models,
                'example_counts': example_counts,
                'metrics': self.metric_names
            },
            'summary_stats': {},
            'detailed_results': detailed_results
        }
        
        # Calculate summary statistics for all models
        for model, results in all_results.items():
            output_data['summary_stats'][model] = {}
            for count in example_counts:
                output_data['summary_stats'][model][count] = {}
                for metric in self.metric_names:
                    scores = results[count][metric]
                    if scores:
                        output_data['summary_stats'][model][count][metric] = {
                            'mean': mean(scores),
                            'std': stdev(scores) if len(scores) > 1 else 0.0,
                            'min': min(scores),
                            'max': max(scores),
                            'count': len(scores)
                        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Comprehensive Translation Benchmark")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), 
                       help="API key for the model provider (or set OPENAI_API_KEY env var)")
    parser.add_argument("--corpus-dir", default="../Corpus", 
                       help="Directory containing corpus files")
    parser.add_argument("--source-file", default="eng-engULB.txt", 
                       help="Source file name")
    parser.add_argument("--query-method", default="context", choices=["bm25", "tfidf", "context"],
                       help="Query method to use")
    parser.add_argument("--model", default="gpt-4o", 
                       help="Single model to test (supports any liteLLM model)")
    parser.add_argument("--models", nargs="+", 
                       help="Multiple models to compare")
    parser.add_argument("--num-target-files", type=int, default=2, 
                       help="Number of target files to test")
    parser.add_argument("--num-tests-per-file", type=int, default=5, 
                       help="Number of tests per file")
    parser.add_argument("--example-counts", nargs="+", type=int, default=[0, 3, 5], 
                       help="Numbers of examples to compare (0 = no examples)")
    parser.add_argument("--output", type=str, 
                       help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ Error: API key required. Set OPENAI_API_KEY env var or use --api-key")
        return 1
    
    # Handle both --model and --models arguments
    models = args.models if args.models else [args.model]
    
    try:
        benchmark = TranslationBenchmark(args.api_key, args.corpus_dir, args.source_file, args.query_method, models)
        benchmark.run_benchmark(args.num_target_files, args.num_tests_per_file, args.example_counts, args.output)
        print("\n✅ Benchmark completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 