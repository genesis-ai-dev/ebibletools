# eBible Tools Benchmark Runner - Google Colab Instructions

This guide shows how to run the eBible Tools benchmarks in Google Colab with proper cell separation.

## Setup Instructions

### Cell 1: Install Required Packages
```python
!pip install litellm matplotlib seaborn plotly pandas numpy scikit-learn tqdm requests python-dotenv faiss-cpu
```

### Cell 2: Clone Repository and Navigate
```python
!git clone https://github.com/daniellosey/ebibletools.git
%cd ebibletools
```

### Cell 3: Import Core Libraries
```python
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
```

### Cell 4: API Key Setup
```python
# Set your API keys using Colab Secrets
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
# Optional: Add other API keys
# os.environ["ANTHROPIC_API_KEY"] = userdata.get('ANTHROPIC_API_KEY')
```

### Cell 5: Test Imports
```python
# Add the project root to Python path
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

print(f"📂 Working directory: {current_dir}")
print("🧪 Testing all imports...")

try:
    # Import metrics module
    from metrics import chrF_plus, normalized_edit_distance
    print("✅ Metrics module imported successfully")
    
    # Import query module  
    from query import Query
    print("✅ Query module imported successfully")
    
    # Import benchmark utilities
    benchmarks_dir = os.path.join(current_dir, 'benchmarks')
    sys.path.append(benchmarks_dir)
    from benchmark_utils import extract_xml_content, format_xml_prompt
    print("✅ Benchmark utilities imported successfully")
    
    # Import benchmark classes
    from biblical_recall_benchmark import BiblicalRecallBenchmark
    from context_corrigibility_benchmark import ContextCorrigibilityBenchmark
    from true_source_benchmark import TrueSourceBenchmark
    from power_prompt_benchmark import PowerPromptBenchmark
    print("✅ All benchmark classes imported successfully")
    
    print("✅ All modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Troubleshooting:")
    print("1. Make sure you're in the ebibletools directory")
    print("2. Check that the git clone completed successfully")
    print("3. Verify package installation completed")
    raise e
```

### Cell 6: Download Corpus Files
```python
print("📥 Downloading corpus files...")

# Import and use the eBible downloader
from ebible_downloader import EBibleDownloader

downloader = EBibleDownloader()

# Download the English source file
print("Downloading English source file...")
downloader.download_file("eng-engULB.txt")

# Download a few target language files
print("Downloading some target language files...")
available_files = downloader.list_files()
target_files = [f for f in available_files if not f["name"].startswith("eng-")]

if target_files:
    import random
    random_targets = random.sample(target_files, min(5, len(target_files)))
    for file_info in random_targets:
        downloader.download_file(file_info["name"])
        
print("✅ Corpus files downloaded!")
```

### Cell 7: Configuration
```python
# Benchmark configuration
models_to_test = [
    "gpt-3.5-turbo", 
    "gpt-4o-mini", 
    "gpt-4o"
    # Add more models as needed:
    # "anthropic/claude-3-5-sonnet",
    # "anthropic/claude-4-sonnet"
]
test_count = 5  # Small number for demo - increase for real testing

# Corpus configuration
corpus_dir = "Corpus"
source_file = "eng-engULB.txt"

print("Configuration:")
print(f"Models: {models_to_test}")
print(f"Test count: {test_count}")
print(f"Corpus: {corpus_dir}/{source_file}")
```

### Cell 8: Run Biblical Recall Benchmark
```python
print("\n🔍 Running Biblical Recall Benchmark...")

benchmark = BiblicalRecallBenchmark(corpus_dir, source_file, models=models_to_test)
benchmark.run_benchmark(num_tests=test_count)
print("✅ Biblical Recall Benchmark completed!")
```

### Cell 9: Run Context Corrigibility Benchmark
```python
print("\n📚 Running Context Corrigibility Benchmark...")

context_results = {}
for model in models_to_test:
    print(f"\nTesting {model}...")
    benchmark = ContextCorrigibilityBenchmark(corpus_dir, source_file, model=model)
    results = benchmark.run_benchmark(num_tests=test_count, example_counts=[0, 3, 5])
    context_results[model] = results

print("✅ Context Corrigibility Benchmark completed!")
```

### Cell 10: Run True Source Benchmark
```python
print("\n🎯 Running True Source Benchmark...")

true_source_results = {}
for model in models_to_test:
    print(f"\nTesting {model}...")
    benchmark = TrueSourceBenchmark(corpus_dir, source_file, model=model)
    results = benchmark.run_benchmark(num_tests=test_count)
    true_source_results[model] = results

print("✅ True Source Benchmark completed!")
```

### Cell 11: Run Power Prompt Benchmark
```python
print("\n⚡ Running Power Prompt Benchmark...")

power_prompt_results = {}
for model in models_to_test:
    print(f"\nTesting {model}...")
    benchmark = PowerPromptBenchmark(corpus_dir, source_file, model=model)
    results = benchmark.run_benchmark(num_tests=test_count)
    power_prompt_results[model] = results

print("✅ Power Prompt Benchmark completed!")
```

### Cell 12: Create Visualizations
```python
print("\n📊 Creating performance visualizations...")

# Extract sample performance data for visualization
def create_sample_visualizations():
    # This is sample data - replace with actual results from benchmarks
    sample_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    biblical_scores = [0.787, 0.688, 0.671]  # Sample chrF+ scores
    context_improvements = [0.414, 0.270, 0.336]  # Improvement from examples
    source_effects = [-0.757, -0.899, -0.022]  # Source dependency effects
    prompt_sensitivity = [0.013, 0.007, 0.009]  # Prompt variation ranges

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Biblical Recall Performance
    bars1 = ax1.bar(sample_models, biblical_scores, color=colors)
    ax1.set_title('Biblical Recall Performance\n(chrF+ Score)', fontweight='bold')
    ax1.set_ylabel('chrF+ Score')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Context Learning Effect
    bars2 = ax2.bar(sample_models, context_improvements, color=colors)
    ax2.set_title('Context Learning Effect\n(5 examples vs 0 examples)', fontweight='bold')
    ax2.set_ylabel('chrF+ Improvement')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Source Text Dependency
    bar_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in source_effects]
    bars3 = ax3.bar(sample_models, source_effects, color=bar_colors)
    ax3.set_title('Source Text Dependency\n(With source vs Without source)', fontweight='bold')
    ax3.set_ylabel('chrF+ Difference')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Prompt Sensitivity
    bars4 = ax4.bar(sample_models, prompt_sensitivity, color=colors)
    ax4.set_title('Prompt Sensitivity\n(Best prompt - Worst prompt)', fontweight='bold')
    ax4.set_ylabel('chrF+ Range')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

create_sample_visualizations()
print("✅ Visualizations completed!")
```

## Notes

1. **API Keys**: Make sure to add your OpenAI API key to Colab Secrets with the name `OPENAI_API_KEY`
2. **Test Count**: Start with a low test count (5) for quick testing, increase for comprehensive results
3. **Models**: You can add or remove models from the `models_to_test` list
4. **Memory**: Large test counts may require Colab Pro for sufficient memory/runtime

## Troubleshooting

If you encounter import errors:
1. Make sure all cells are run in order
2. Check that the git clone was successful
3. Verify that package installation completed without errors
4. Try restarting the runtime and running from the beginning

## Expected Runtime

- Setup (Cells 1-6): ~2-3 minutes
- Each benchmark: ~1-3 minutes per model (depending on test count)
- Total for 3 models with 5 tests each: ~15-20 minutes 