# eBible Tools Benchmark Runner for Google Colab
# Run this in Google Colab for interactive benchmarking with visualizations

# Cell 1: Setup and Package Installation
!pip install litellm matplotlib seaborn plotly pandas numpy scikit-learn tqdm python-dotenv requests nltk

# Cell 2: Clone Repository and Setup
!git clone https://github.com/genesis-ai-dev/ebibletools
%cd ebibletools

# Cell 3: Setup API Keys
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = userdata.get('ANTHROPIC_API_KEY')

# Cell 4: Download corpus files
from ebible_downloader import EBibleDownloader

downloader = EBibleDownloader()  # Will automatically use project's Corpus directory
downloader.download_file("eng-engULB.txt")  # Source text

# Download specific languages
downloader.download_most_complete_for_language('spa')  # Spanish  
downloader.download_most_complete_for_language('fra')  # French
# Add more languages as needed:
# downloader.download_most_complete_for_language('hin')  # Hindi
# downloader.download_most_complete_for_language("deu")  # German
# downloader.download_most_complete_for_language("por")  # Portuguese

# Cell 5: Check what we downloaded
print("📁 Corpus files:")
!ls -la Corpus/

# Cell 6: Verify benchmark paths work correctly
from pathlib import Path
print("\n🔍 Verifying benchmark paths:")
benchmark_file = Path("benchmarks/biblical_recall_benchmark.py")
corpus_path = benchmark_file.parent.parent / "Corpus"
print(f"   Benchmark location: {benchmark_file}")
print(f"   Corpus path: {corpus_path}")
print(f"   Corpus exists: {corpus_path.exists()}")
print(f"   Files in corpus: {len(list(corpus_path.glob('*.txt')))}")

# Cell 7: Import benchmark classes
import sys
sys.path.append('.')
from benchmarks.biblical_recall_benchmark import BiblicalRecallBenchmark
from benchmarks.context_corrigibility_benchmark import ContextCorrigibilityBenchmark
from benchmarks.true_source_benchmark import TrueSourceBenchmark
from benchmarks.power_prompt_benchmark import PowerPromptBenchmark

# Cell 8: Configuration
models_to_test = [
    "gpt-3.5-turbo",
    "claude-3-haiku-20240307",
    "gpt-4o-mini", 
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "anthropic/claude-sonnet-4-20250514",
]


# Cell 9: Run Biblical Recall Benchmark (Multi-model) - Capture results
print("\n🧠 Running Biblical Recall Benchmark...")
corpus_dir = str(Path("Corpus"))
biblical_benchmark = BiblicalRecallBenchmark(corpus_dir, "eng-engULB.txt", models_to_test)
biblical_results = biblical_benchmark.run_benchmark(num_tests=10)

# Cell 10: Run Context Corrigibility Benchmark - Capture results  
print("\n🎯 Running Context Corrigibility Benchmark...")
context_benchmark = ContextCorrigibilityBenchmark(corpus_dir, "eng-engULB.txt", models_to_test[0])
context_results = context_benchmark.run_benchmark(example_counts=[0, 3, 5], num_tests=5)

# Cell 11: Run True Source Benchmark - Capture results
print("\n📖 Running True Source Benchmark...")
source_benchmark = TrueSourceBenchmark(corpus_dir, "eng-engULB.txt", models_to_test[0])
source_results = source_benchmark.run_benchmark(num_tests=5)

# Cell 12: Run Power Prompt Benchmark - Capture results
print("\n⚡ Running Power Prompt Benchmark...")
prompt_benchmark = PowerPromptBenchmark(corpus_dir, "eng-engULB.txt", models_to_test[0])
prompt_results = prompt_benchmark.run_benchmark(num_tests=5)

print("\n✅ All benchmarks completed!")

# Cell 13: Extract performance data for visualization
def extract_performance_data():
    """Extract performance data from benchmark results for visualization"""
    
    # Biblical Recall data (multi-model)
    biblical_data = {}
    if biblical_results and "summary" in biblical_results:
        for model in biblical_results["summary"]:
            biblical_data[model] = biblical_results["summary"][model]["chrf_mean"]
    
    # Context Corrigibility data (improvement from 0 to 5 examples)
    context_data = {}
    if context_results and "summary" in context_results:
        overall_stats = context_results["summary"]["overall"]
        if 0 in overall_stats and 5 in overall_stats:
            baseline = overall_stats[0]["chrf_mean"]
            with_context = overall_stats[5]["chrf_mean"]
            context_data["improvement"] = with_context - baseline
            context_data["baseline"] = baseline
            context_data["with_context"] = with_context
    
    # True Source data (with vs without source)
    source_data = {}
    if source_results and "summary" in source_results:
        overall_stats = source_results["summary"]["overall"]
        if "with_source" in overall_stats and "without_source" in overall_stats:
            with_source = overall_stats["with_source"]["chrf_mean"]
            without_source = overall_stats["without_source"]["chrf_mean"]
            source_data["improvement"] = with_source - without_source
            source_data["with_source"] = with_source
            source_data["without_source"] = without_source
    
    # Power Prompt data (best prompt style)
    prompt_data = {}
    if prompt_results and "summary" in prompt_results:
        overall_stats = prompt_results["summary"]["overall"]
        # Find best prompt style
        best_prompt = max(overall_stats.items(), key=lambda x: x[1]["chrf_mean"])
        prompt_data["best_prompt"] = best_prompt[0]
        prompt_data["best_score"] = best_prompt[1]["chrf_mean"]
        prompt_data["all_prompts"] = {k: v["chrf_mean"] for k, v in overall_stats.items()}
    
    return biblical_data, context_data, source_data, prompt_data

# Cell 14: Create visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract performance data
biblical_data, context_data, source_data, prompt_data = extract_performance_data()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('eBible Tools Benchmark Results', fontsize=16, fontweight='bold')

# 1. Biblical Recall Comparison
if biblical_data:
    ax1 = axes[0, 0]
    models = list(biblical_data.keys())
    scores = list(biblical_data.values())
    bars = ax1.bar(models, scores, color='skyblue', alpha=0.7)
    ax1.set_title('Biblical Recall Performance (chrF+)', fontweight='bold')
    ax1.set_ylabel('chrF+ Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

# 2. Context Corrigibility Effect
if context_data:
    ax2 = axes[0, 1]
    conditions = ['0 Examples', '5 Examples']
    scores = [context_data['baseline'], context_data['with_context']]
    bars = ax2.bar(conditions, scores, color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax2.set_title('Context Corrigibility Effect', fontweight='bold')
    ax2.set_ylabel('chrF+ Score')
    
    # Add improvement annotation
    improvement = context_data['improvement']
    ax2.annotate(f'Improvement: +{improvement:.3f}', 
                xy=(1, context_data['with_context']), 
                xytext=(0.5, max(scores) + 0.05),
                ha='center', fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

# 3. True Source Effect
if source_data:
    ax3 = axes[1, 0]
    conditions = ['Without Source', 'With Source']
    scores = [source_data['without_source'], source_data['with_source']]
    bars = ax3.bar(conditions, scores, color=['lightcoral', 'lightblue'], alpha=0.7)
    ax3.set_title('True Source Effect', fontweight='bold')
    ax3.set_ylabel('chrF+ Score')
    
    # Add improvement annotation
    improvement = source_data['improvement']
    color = 'green' if improvement > 0 else 'red'
    ax3.annotate(f'Effect: {improvement:+.3f}', 
                xy=(1, source_data['with_source']), 
                xytext=(0.5, max(scores) + 0.05),
                ha='center', fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color))
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

# 4. Power Prompt Comparison
if prompt_data:
    ax4 = axes[1, 1]
    prompts = list(prompt_data['all_prompts'].keys())
    scores = list(prompt_data['all_prompts'].values())
    
    # Color the best prompt differently
    colors = ['gold' if p == prompt_data['best_prompt'] else 'lightsteelblue' for p in prompts]
    bars = ax4.bar(prompts, scores, color=colors, alpha=0.7)
    ax4.set_title('Power Prompt Comparison', fontweight='bold')
    ax4.set_ylabel('chrF+ Score')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n📊 Visualization complete!")
print(f"📋 Results summary:")
if biblical_data:
    best_model = max(biblical_data.items(), key=lambda x: x[1])
    print(f"   🏆 Best model: {best_model[0]} ({best_model[1]:.3f} chrF+)")
if context_data:
    print(f"   📈 Context improvement: +{context_data['improvement']:.3f} chrF+")
if source_data:
    print(f"   📖 Source effect: {source_data['improvement']:+.3f} chrF+")
if prompt_data:
    print(f"   ⚡ Best prompt: {prompt_data['best_prompt']} ({prompt_data['best_score']:.3f} chrF+)") 