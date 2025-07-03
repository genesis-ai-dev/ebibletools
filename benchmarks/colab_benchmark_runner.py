# eBible Tools Benchmark Runner for Google Colab

# Cell 1: Install packages
# !pip install litellm matplotlib seaborn plotly pandas numpy scikit-learn tqdm requests python-dotenv faiss-cpu

# Cell 2: Clone repository
# !git clone https://github.com/daniellosey/ebibletools.git
# %cd ebibletools

# Cell 3: Import libraries
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import random

# Setup
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, 'benchmarks'))

from ebibletools.metrics import chrF_plus, normalized_edit_distance
from ebibletools.query import Query
from ebibletools.benchmarks.benchmark_utils import extract_xml_content, format_xml_prompt
from ebibletools.benchmarks.biblical_recall_benchmark import BiblicalRecallBenchmark
from ebibletools.benchmarks.context_corrigibility_benchmark import ContextCorrigibilityBenchmark
from ebibletools.benchmarks.true_source_benchmark import TrueSourceBenchmark
from ebibletools.benchmarks.power_prompt_benchmark import PowerPromptBenchmark

plt.style.use('default')
sns.set_palette("husl")

# Cell 4: API Keys
# from google.colab import userdata
# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Cell 5: Download corpus files
from ebibletools.ebible_downloader import EBibleDownloader

downloader = EBibleDownloader()
downloader.download_file("eng-engULB.txt")

available_files = downloader.list_files()
target_files = [f for f in available_files if not f["name"].startswith("eng-")]
if target_files:
    random_targets = random.sample(target_files, min(5, len(target_files)))
    for file_info in random_targets:
        downloader.download_file(file_info["name"])

# Cell 6: Configuration
models_to_test = [
    "gpt-3.5-turbo", 
    "gpt-4o-mini", 
    "gpt-4o"
]
test_count = 5
corpus_dir = "Corpus"
source_file = "eng-engULB.txt"

# Cell 7: Run benchmarks
print("Running Biblical Recall Benchmark...")
benchmark = BiblicalRecallBenchmark(corpus_dir, source_file, models=models_to_test)
biblical_results = benchmark.run_benchmark(num_tests=test_count)

print("Running Context Corrigibility Benchmark...")
context_results = {}
for model in models_to_test:
    benchmark = ContextCorrigibilityBenchmark(corpus_dir, source_file, model=model)
    context_results[model] = benchmark.run_benchmark(num_tests=test_count, example_counts=[0, 3, 5])

print("Running True Source Benchmark...")
true_source_results = {}
for model in models_to_test:
    benchmark = TrueSourceBenchmark(corpus_dir, source_file, model=model)
    true_source_results[model] = benchmark.run_benchmark(num_tests=test_count)

print("Running Power Prompt Benchmark...")
power_prompt_results = {}
for model in models_to_test:
    benchmark = PowerPromptBenchmark(corpus_dir, source_file, model=model)
    power_prompt_results[model] = benchmark.run_benchmark(num_tests=test_count)

# Cell 8: Visualize results
def extract_performance_data():
    biblical_data = {model: biblical_results["summary"][model]["chrf_mean"] 
                    for model in biblical_results["summary"]}
    
    # Updated to handle new multi-language structure - use overall aggregated results
    context_data = {model: results["summary"]["overall"][5]["chrf_mean"] - results["summary"]["overall"][0]["chrf_mean"]
                   for model, results in context_results.items()}
    
    source_data = {model: results["summary"]["overall"]["with_source"]["chrf_mean"] - results["summary"]["overall"]["without_source"]["chrf_mean"]
                  for model, results in true_source_results.items()}
    
    prompt_data = {model: max(results["summary"]["overall"][prompt]["overall"] for prompt in results["summary"]["overall"]) - 
                           min(results["summary"]["overall"][prompt]["overall"] for prompt in results["summary"]["overall"])
                  for model, results in power_prompt_results.items()}
    
    return biblical_data, context_data, source_data, prompt_data

biblical_data, context_data, source_data, prompt_data = extract_performance_data()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
models = list(biblical_data.keys())
model_names = [m.split('/')[-1] if '/' in m else m for m in models]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(models)]

# Biblical Recall Performance
biblical_scores = [biblical_data[model] for model in models]
bars1 = ax1.bar(model_names, biblical_scores, color=colors)
ax1.set_title('Biblical Recall Performance\n(chrF+ Score)', fontweight='bold')
ax1.set_ylabel('chrF+ Score')
ax1.set_ylim(0, 1)
for bar, score in zip(bars1, biblical_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Context Learning Effect
context_improvements = [context_data[model] for model in models]
bars2 = ax2.bar(model_names, context_improvements, color=colors)
ax2.set_title('Context Learning Effect\n(5 examples vs 0 examples)', fontweight='bold')
ax2.set_ylabel('chrF+ Improvement')
for bar, improvement in zip(bars2, context_improvements):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'+{improvement:.3f}', ha='center', va='bottom', fontweight='bold')

# Source Text Dependency
source_effects = [source_data[model] for model in models]
bar_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in source_effects]
bars3 = ax3.bar(model_names, source_effects, color=bar_colors)
ax3.set_title('Source Text Dependency\n(With source vs Without source)', fontweight='bold')
ax3.set_ylabel('chrF+ Difference')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
for bar, effect in zip(bars3, source_effects):
    y_offset = 0.015 if effect < 0 else -0.025
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
            f'{effect:+.3f}', ha='center', va='bottom' if effect < 0 else 'top', fontweight='bold')

# Prompt Sensitivity
prompt_ranges = [prompt_data[model] for model in models]
bars4 = ax4.bar(model_names, prompt_ranges, color=colors)
ax4.set_title('Prompt Sensitivity\n(Best prompt - Worst prompt)', fontweight='bold')
ax4.set_ylabel('chrF+ Range')
for bar, range_val in zip(bars4, prompt_ranges):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
            f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Cell 9: Summary
print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)

print(f"\nBiblical Recall (Best: {max(biblical_data.values()):.3f}):")
for model, score in sorted(biblical_data.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {score:.3f}")

print(f"\nContext Learning (Best: +{max(context_data.values()):.3f}):")
for model, improvement in sorted(context_data.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: +{improvement:.3f}")

print(f"\nSource Dependency (Best: {max(source_data.values()):+.3f}):")
for model, effect in sorted(source_data.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {effect:+.3f}")

print(f"\nPrompt Sensitivity (Most stable: {min(prompt_data.values()):.3f}):")
for model, range_val in sorted(prompt_data.items(), key=lambda x: x[1]):
    print(f"  {model}: {range_val:.3f}")

best_overall = max(biblical_data.items(), key=lambda x: x[1])[0]
print(f"\nOverall Winner: {best_overall}") 