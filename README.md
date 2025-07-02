# eBible Tools

A comprehensive toolkit for biblical text processing, translation benchmarking, and semantic search with advanced query capabilities.

## Features

- **Unified Query Interface**: Switch between BM25, TF-IDF, and context-aware search methods
- **Translation Benchmarking**: Comprehensive evaluation framework with multiple quality metrics
- **Multi-Provider LLM Support**: Use 100+ LLM providers (OpenAI, Anthropic, Google, Groq, etc.) via liteLLM
- **Professional MT Metrics**: Industry-standard evaluation including BLEU, chrF, METEOR, ROUGE, BERTScore
- **Semantic Search**: Advanced context-aware search with coverage weighting and branching
- **Corpus Management**: Tools for downloading and processing biblical texts

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd ebibletools
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download some biblical texts
python ebible_downloader.py

# Run a simple benchmark
cd benchmarks
python translation_benchmark.py --query-method context --num-tests-per-file 5 --model gpt-4o
```

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
from metrics import sentence_bleu, chrF_plus, meteor_score, bert_score

hypothesis = "The cat sits on the mat."
reference = "The cat sat on the mat."

print(f"BLEU: {sentence_bleu(hypothesis, reference):.4f}")
print(f"chrF+: {chrF_plus(hypothesis, reference):.4f}")
print(f"METEOR: {meteor_score(hypothesis, reference):.4f}")
print(f"BERTScore: {bert_score(hypothesis, reference):.4f}")
```

### Available Metrics
- **BLEU**: N-gram precision with SacreBLEU implementation
- **chrF/chrF+/chrF++**: Character n-gram F-score variants
- **METEOR**: Full implementation with stemming and synonyms
- **ROUGE**: Multiple variants (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-W, ROUGE-S)
- **BERTScore**: Transformer-based semantic similarity
- **Edit Distance**: Levenshtein distance (raw and normalized)
- **TER**: Translation Error Rate

All metrics include:
- Professional implementations using established libraries
- Comprehensive linguistic processing (stemming, synonyms, etc.)
- Batch processing capabilities
- Clear error handling with fast failure

See [`metrics/README.md`](metrics/README.md) for detailed documentation.

## Benchmarks

The benchmarking system provides comprehensive translation quality evaluation with support for multiple LLM providers via liteLLM:

```bash
cd benchmarks
python translation_benchmark.py --help

# Examples with different providers
python translation_benchmark.py --query-method context --model gpt-4o --num-tests-per-file 10
python translation_benchmark.py --query-method bm25 --model anthropic/claude-3-sonnet-20240229
python translation_benchmark.py --query-method context --model groq/llama2-70b-4096
```

### LLM Provider Support

The benchmark now uses liteLLM for universal LLM access. Supported providers include:

- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-sonnet-20240229`, `anthropic/claude-3-haiku-20240307`
- **Google**: `gemini/gemini-pro`, `gemini/gemini-pro-vision`
- **Groq**: `groq/llama2-70b-4096`, `groq/mixtral-8x7b-32768`
- **OpenRouter**: `openrouter/anthropic/claude-3-haiku`
- **And 100+ more providers**

Set the appropriate API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
# etc.
```

### Command Line Options
- `--query-method`: Choose between `bm25`, `tfidf`, or `context`
- `--model`: LLM model to use (supports any liteLLM model)
- `--num-tests-per-file`: Number of test cases to evaluate per file
- `--num-target-files`: Number of target language files to test
- `--example-counts`: List of example counts to compare (e.g., 0 3 5)
- `--corpus-dir`: Path to corpus directory
- `--output`: Save results to JSON file

### Quick Provider Switching Example

```python
from benchmarks.translation_benchmark import TranslationBenchmark

# Switch between providers easily
models_to_test = [
    "gpt-4o",                                    # OpenAI
    "anthropic/claude-3-sonnet-20240229",       # Anthropic  
    "groq/llama2-70b-4096",                     # Groq
]

for model in models_to_test:
    print(f"Testing {model}...")
    benchmark = TranslationBenchmark(
        api_key="your-api-key", 
        corpus_dir="Corpus",
        source_file="eng-engULB.txt",
        model=model
    )
    # Run your benchmark...
```

See [`benchmarks/README.md`](benchmarks/README.md) for detailed usage.

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
│   ├── bleu.py           # BLEU with SacreBLEU
│   ├── chrf.py           # chrF variants
│   ├── meteor.py         # METEOR with linguistics
│   ├── rouge.py          # ROUGE variants
│   ├── bert_score.py     # BERTScore implementation
│   ├── edit_distance.py  # Levenshtein distance
│   ├── ter.py            # Translation Error Rate
│   └── README.md         # Detailed metric documentation
├── benchmarks/           # Benchmarking framework
│   ├── translation_benchmark.py  # Main benchmark script (liteLLM-powered)
│   ├── litellm_example.py        # Example showing multi-provider usage
│   └── README.md         # Benchmark documentation
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
- `openai` - OpenAI API integration (legacy)
- `litellm` - Universal LLM API for multiple providers
- `scikit-learn` - TF-IDF and similarity calculations
- `tqdm` - Progress bars
- `numpy` - Numerical computations
- `faiss-cpu` - Efficient similarity search
- `python-dotenv` - Environment variable management

### Translation Metrics
- `nltk` - Natural language processing for METEOR and ROUGE
- `sacrebleu` - Professional BLEU implementation
- `bert-score` - Transformer-based semantic evaluation
- `torch` - Required for BERTScore
- `transformers` - Transformer models for BERTScore

The system will fail clearly if any required dependencies are missing.

## Usage Examples

### Basic Query Usage
```python
from query import Query

# Initialize with your preferred method
query = Query(method='context')

# Search for verses about love
results = query.search_by_text("love your enemies", limit=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Reference: {result['reference']}")
    print()
```

### Translation Evaluation
```python
from metrics import (
    sentence_bleu, chrF_plus, meteor_score, 
    rouge_l, bert_score, ter_score
)

def evaluate_translation(hypothesis, reference):
    """Comprehensive translation evaluation"""
    scores = {
        'BLEU': sentence_bleu(hypothesis, reference),
        'chrF+': chrF_plus(hypothesis, reference),
        'METEOR': meteor_score(hypothesis, reference),
        'ROUGE-L': rouge_l(hypothesis, reference)[2],  # F1
        'BERTScore': bert_score(hypothesis, reference),
        'TER': ter_score(hypothesis, reference)
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
from query import Query
from metrics import corpus_bleu, bert_score_batch

# Process multiple queries
query = Query(method='bm25')
texts = ["faith", "hope", "love"]
all_results = []

for text in texts:
    results = query.search_by_text(text, limit=3)
    all_results.extend(results)

# Evaluate multiple translations
hypotheses = ["Translation 1", "Translation 2"]
references = ["Reference 1", "Reference 2"]

bleu_score = corpus_bleu(hypotheses, references)
_, _, bert_f1_scores = bert_score_batch(hypotheses, references)

print(f"Corpus BLEU: {bleu_score:.4f}")
print(f"BERTScore F1: {bert_f1_scores}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here] 