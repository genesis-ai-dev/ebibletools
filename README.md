# eBible Tools

A comprehensive toolkit for biblical text processing, translation benchmarking, and semantic search with advanced query capabilities.

## Features

- **Unified Query Interface**: Switch between BM25, TF-IDF, and context-aware search methods
- **Five Specialized Benchmarks**: Biblical recall, translation corrigibility, source effects, prompt optimization, and simplified translation
- **Multi-Model Comparison**: Compare multiple LLM models side-by-side on the same benchmarks
- **Multi-Provider LLM Support**: Use 100+ LLM providers (OpenAI, Anthropic, Google, Groq, etc.) via liteLLM
- **Professional MT Metrics**: Industry-standard evaluation including chrF+, Edit Distance, TER
- **Semantic Search**: Advanced context-aware search with coverage weighting and branching
- **Corpus Management**: Tools for downloading and processing biblical texts
- **Standardized XML Output**: Consistent response formatting across all models

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd ebibletools
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set up your API keys in .env file
cp benchmarks/env.template .env
# Edit .env with your API keys

# Download some biblical texts
python ebible_downloader.py

# Run benchmarks
cd benchmarks
python biblical_recall_benchmark.py --num-tests 20
python context_corrigibility_benchmark.py --model gpt-4o --example-counts 0 3 5
python true_source_benchmark.py --num-tests 15
python power_prompt_benchmark.py --num-tests 12
python translation_benchmark_simple.py --num-tests 10
```

## Benchmarks

### 📖 **Biblical Recall Benchmark**
Tests how well models can recall biblical text when given verse references.

```bash
# Single model
python biblical_recall_benchmark.py --num-tests 20 --model gpt-4o --output recall_results.json

# Multi-model comparison
python biblical_recall_benchmark.py --models gpt-4o claude-3-opus gemini-pro --num-tests 10
```

**What it tests:**
- Given a biblical reference (e.g., "GEN 1:1", "HEB 4:12"), can the model quote the verse?
- Uses aligned reference file (`vref.txt`) with the actual biblical corpus
- Tests random verses from across the entire Bible (41,900+ verses)
- Tests pure recall ability without completion hints
- **Multi-model support**: Compare recall abilities across different models

**Metrics**: chrF+, Edit Distance similarity

### 🔄 **Context Corrigibility Benchmark** 
Tests how in-context examples improve translation accuracy.

```bash
python context_corrigibility_benchmark.py --model gpt-4o --example-counts 0 3 5 --num-tests 10
```

**What it tests:**
- Translation quality with 0, 3, 5 examples
- How much context examples improve performance
- Corrigibility analysis (improvement measurement)

**Metrics**: chrF+, Edit Distance with improvement tracking

### 🎯 **True Source Benchmark**
Tests how having source text affects translation accuracy and memorization.

```bash
python true_source_benchmark.py --num-tests 15 --model gpt-4o --output source_results.json
```

**What it tests:**
- **With source**: Normal translation with correct source text
- **Without source**: Translation from memory (tests memorization vs. bias)
- Measures the impact of source text availability
- Reveals whether source text biases translation output

**Metrics**: chrF+, Edit Distance with source effect analysis

### 💪 **Power Prompt Benchmark**
Tests effectiveness of different prompt styles for translation.

```bash
python power_prompt_benchmark.py --num-tests 12 --model gpt-4o --output prompt_results.json
```

**What it tests:**
- 4 key prompt styles (basic, expert, biblical scholar, direct)
- Prompt effectiveness ranking
- Statistical comparison of prompt impact

**Metrics**: chrF+, Edit Distance with overall ranking

### 🌍 **Simple Translation Benchmark**
Streamlined translation testing with context examples.

```bash
python translation_benchmark_simple.py --model gpt-4o --num-tests 10 --example-counts 0 3 5
```

**What it tests:**
- Translation quality improvement with context examples
- Simplified, focused testing (202 lines vs 323 in original)
- Single target language per run for cleaner results

**Metrics**: chrF+, Edit Distance with improvement analysis

## Query System

The unified query system supports three different search methods:

### BM25 Query (Default)
```python
from query import Query

# Pure BM25 scoring
query = Query(method='bm25')
results = query.search_by_text("love your neighbor", limit=5)
```

### TF-IDF Query
```python
# TF-IDF based similarity
query = Query(method='tfidf')
results = query.search_by_text("forgiveness", limit=5)
```

### Context Query (Advanced)
```python
# BM25 + branching search with coverage weighting
query = Query(method='context')
results = query.search_by_text("faith and hope", limit=5)
```

All methods support the same interface:
- `search_by_text(text, limit=10)` - Search by text content
- `search_by_line(line_number, limit=10)` - Search by line reference

## Translation Quality Metrics

The `metrics/` module provides comprehensive MT evaluation with professional-grade implementations:

### Quick Example
```python
from metrics import chrF_plus, normalized_edit_distance, ter_score

hypothesis = "The cat sits on the mat."
reference = "The cat sat on the mat."

print(f"chrF+: {chrF_plus(hypothesis, reference):.4f}")
print(f"Edit Distance: {normalized_edit_distance(hypothesis, reference):.4f}")
print(f"TER: {ter_score(hypothesis, reference):.4f}")
```

### Available Metrics
- **chrF/chrF+/chrF++**: Character n-gram F-score variants
- **Edit Distance**: Levenshtein distance (raw and normalized)
- **TER**: Translation Error Rate

All metrics include:
- Professional implementations using established libraries
- Comprehensive linguistic processing
- Batch processing capabilities
- Clear error handling with fast failure

The metrics are **universal** and work with any language or script without requiring language-specific resources.

## Benchmark Command Reference

### Common Options
All benchmarks support these standard options:

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus-dir` | `../Corpus` | Directory containing corpus files |
| `--source-file` | `eng-engULB.txt` | Source language file |
| `--model` | `gpt-4o` | Model to use (any liteLLM model) |
| `--output` | None | Save detailed results to JSON file |

**Note**: API keys are automatically loaded from your `.env` file - no need to specify them manually.

### Biblical Recall Options
```bash
# Single model
python biblical_recall_benchmark.py \
  --num-tests 20 \
  --model gpt-4o \
  --output recall_results.json

# Multi-model comparison  
python biblical_recall_benchmark.py \
  --models gpt-4o claude-3-opus gemini-pro \
  --num-tests 10 \
  --output multi_model_recall.json
```

### Context Corrigibility Options
```bash
python context_corrigibility_benchmark.py \
  --model gpt-4o \
  --query-method context \
  --num-tests 10 \
  --example-counts 0 3 5 \
  --output context_results.json
```

### True Source Options
```bash
python true_source_benchmark.py \
  --num-tests 15 \
  --model gpt-4o \
  --output source_results.json
```

### Power Prompt Options
```bash
python power_prompt_benchmark.py \
  --num-tests 12 \
  --model gpt-4o \
  --output prompt_results.json
```

### Simple Translation Options
```bash
python translation_benchmark_simple.py \
  --model gpt-4o \
  --query-method context \
  --num-tests 10 \
  --example-counts 0 3 5 \
  --output simple_translation_results.json
```

## LLM Provider Support

All benchmarks use liteLLM for universal LLM access. Supported providers include:

- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- **Google**: `gemini/gemini-pro`, `gemini/gemini-pro-vision`
- **Groq**: `groq/llama-3.1-70b-versatile`, `groq/mixtral-8x7b-32768`
- **OpenRouter**: `openrouter/anthropic/claude-3-haiku`
- **And 100+ more providers**

### Environment Setup
Create a `.env` file in your project root with your API keys:

```bash
# Copy the template
cp benchmarks/env.template .env

# Edit .env with your keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
# etc.
```

liteLLM automatically detects and uses the appropriate API keys from your environment.

## Multi-Model Comparison

The biblical recall benchmark supports comparing multiple models:

```bash
# Compare 3 models on biblical recall
python biblical_recall_benchmark.py --models gpt-4o claude-3-opus gemini-pro --num-tests 20
```

**Output includes:**
- Side-by-side performance comparison
- Ranking with performance differences
- Best model highlighted with ⭐
- Statistical significance indicators

**Example output:**
```
BIBLICAL RECALL RESULTS - MODEL COMPARISON
==========================================

gpt-4o ⭐ (best)
  chrF+: 0.734±0.156
  Edit:  0.698±0.134
  High accuracy: 3/20 (15.0%)

claude-3-opus
  chrF+: 0.701±0.142
  Edit:  0.672±0.128
  High accuracy: 2/20 (10.0%)

gemini-pro
  chrF+: 0.623±0.189
  Edit:  0.587±0.156
  High accuracy: 1/20 (5.0%)

RANKING SUMMARY
--------------------------------
1. gpt-4o            chrF+: 0.734
2. claude-3-opus     chrF+: 0.701 (-0.033)
3. gemini-pro        chrF+: 0.623 (-0.111)
```

## Project Structure

```
ebibletools/
├── query/                  # Query system
│   ├── __init__.py        # Unified Query interface
│   ├── base.py            # Base query class
│   ├── bm25_query.py      # BM25 implementation
│   ├── tfidf_query.py     # TF-IDF implementation
│   └── contextquery.py    # Context-aware search
├── metrics/               # Translation quality metrics
│   ├── __init__.py       # Metric imports
│   ├── chrf.py           # chrF variants
│   ├── edit_distance.py  # Levenshtein distance
│   ├── ter.py            # Translation Error Rate
│   └── README.md         # Detailed metric documentation
├── benchmarks/           # Benchmarking framework
│   ├── benchmark_utils.py            # Shared utilities
│   ├── biblical_recall_benchmark.py  # Biblical memory testing (multi-model)
│   ├── context_corrigibility_benchmark.py  # Context learning analysis
│   ├── true_source_benchmark.py      # Source text effect analysis
│   ├── power_prompt_benchmark.py     # Prompt optimization testing
│   ├── translation_benchmark_simple.py     # Simplified translation benchmark
│   ├── translation_benchmark.py     # Original comprehensive benchmark
│   ├── data/
│   │   └── vref.txt      # Biblical verse references (41,900+ verses)
│   └── env.template      # Environment variable template
├── Corpus/               # Biblical text corpus
├── ebible_downloader.py  # Corpus download utility
└── requirements.txt      # Dependencies
```

## Dependencies

All dependencies are **required** for proper functionality:

```bash
pip install -r requirements.txt
```

### Core Requirements
- `requests` - HTTP requests for downloading
- `litellm` - Universal LLM API for multiple providers
- `scikit-learn` - TF-IDF and similarity calculations
- `tqdm` - Progress bars
- `numpy` - Numerical computations
- `python-dotenv` - Environment variable management

### Translation Metrics
- `nltk` - Natural language processing
- `torch` - Required for some metrics
- `transformers` - Transformer models

The system will fail clearly if any required dependencies are missing.

## Usage Examples

### Basic Benchmark Usage
```python
from benchmarks.biblical_recall_benchmark import BiblicalRecallBenchmark

# Test biblical recall (single model)
benchmark = BiblicalRecallBenchmark(
    corpus_dir="Corpus",
    source_file="eng-engULB.txt",
    models=["gpt-4o"]
)
benchmark.run_benchmark(num_tests=20, output_file="recall_results.json")

# Multi-model comparison
benchmark = BiblicalRecallBenchmark(
    corpus_dir="Corpus", 
    source_file="eng-engULB.txt",
    models=["gpt-4o", "claude-3-opus", "gemini-pro"]
)
benchmark.run_benchmark(num_tests=10, output_file="multi_model_recall.json")
```

### Context Learning Analysis
```python
from benchmarks.context_corrigibility_benchmark import ContextCorrigibilityBenchmark

# Test context learning
benchmark = ContextCorrigibilityBenchmark(
    corpus_dir="Corpus",
    source_file="eng-engULB.txt",
    model="gpt-4o"
)
benchmark.run_benchmark(
    num_tests=10,
    example_counts=[0, 3, 5, 10],
    output_file="context_comparison.json"
)
```

### Simplified Translation Testing
```python
from benchmarks.translation_benchmark_simple import SimpleTranslationBenchmark

# Quick translation benchmark
benchmark = SimpleTranslationBenchmark(
    corpus_dir="Corpus",
    source_file="eng-engULB.txt", 
    model="gpt-4o",
    query_method="context"
)
benchmark.run_benchmark(
    num_tests=10,
    example_counts=[0, 3, 5],
    output_file="simple_translation.json"
)
```

### Translation Evaluation
```python
from metrics import chrF_plus, normalized_edit_distance, ter_score

def evaluate_translation(hypothesis, reference):
    """Comprehensive translation evaluation"""
    scores = {
        'chrF+': chrF_plus(hypothesis, reference),
        'Edit Distance': 1.0 - normalized_edit_distance(hypothesis, reference),
        'TER': max(0.0, 1.0 - ter_score(hypothesis, reference))
    }
    return scores

# Example evaluation
hypothesis = "Love your neighbor as yourself."
reference = "You shall love your neighbor as yourself."

scores = evaluate_translation(hypothesis, reference)
for metric, score in scores.items():
    print(f"{metric}: {score:.4f}")
```

### Batch Processing
```python
# Run all benchmarks for comprehensive analysis
models = ["gpt-4o", "claude-3-opus", "gemini-pro"]

# Multi-model biblical recall
recall_benchmark = BiblicalRecallBenchmark("Corpus", "eng-engULB.txt", models)
recall_benchmark.run_benchmark(20, "multi_model_recall.json")

# Individual model testing on other benchmarks
for model in models:
    print(f"Testing {model}...")
    
    # Prompt power
    prompt_benchmark = PowerPromptBenchmark("Corpus", "eng-engULB.txt", model)
    prompt_benchmark.run_benchmark(12, f"prompt_{model.replace('/', '_')}.json")
    
    # Simple translation
    translation_benchmark = SimpleTranslationBenchmark("Corpus", "eng-engULB.txt", model)
    translation_benchmark.run_benchmark(10, [0, 3, 5], f"translation_{model.replace('/', '_')}.json")
```

## XML Output Format

All benchmarks use standardized XML output formatting for consistent parsing across models:

- **Biblical Recall**: `<verse>biblical verse here</verse>`
- **Translation Benchmarks**: `<translation>your translation here</translation>`

This ensures reliable extraction of model responses regardless of the LLM provider.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here] 