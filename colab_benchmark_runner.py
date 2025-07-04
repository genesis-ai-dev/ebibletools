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

# Cell 7: Configuration
models_to_test = [
    "gpt-3.5-turbo",
    "claude-3-haiku-20240307",
    "gpt-4o-mini", 
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "anthropic/claude-sonnet-4-20250514",
]

# Cell 8: Run Biblical Recall Benchmark (Multi-model)
print("\n🧠 Running Biblical Recall Benchmark...")
!python benchmarks/biblical_recall_benchmark.py --models gpt-3.5-turbo claude-3-haiku-20240307 gpt-4o-mini gpt-4o claude-3-5-sonnet-20240620 anthropic/claude-sonnet-4-20250514 --num-tests 10

# Cell 9: Run Context Corrigibility Benchmark
print("\n🎯 Running Context Corrigibility Benchmark...")
!python benchmarks/context_corrigibility_benchmark.py --model gpt-4o-mini --example-counts 0 3 5 --num-tests 5

# Cell 10: Run True Source Benchmark
print("\n📖 Running True Source Benchmark...")
!python benchmarks/true_source_benchmark.py --model gpt-4o-mini --num-tests 5

# Cell 11: Run Power Prompt Benchmark
print("\n⚡ Running Power Prompt Benchmark...")
!python benchmarks/power_prompt_benchmark.py --model gpt-4o-mini --num-tests 5

print("\n✅ All benchmarks completed!") 