# eBible Tools

Tools for downloading and analyzing Bible translations for machine learning applications, particularly for generating in-context learning examples for translation training. Made this because I'm tired of remaking basically this from scratch every time I need it (:

## Components

### EBible Downloader (`ebible_downloader.py`)
Downloads parallel Bible translations from the [eBible corpus](https://github.com/BibleNLP/ebible/tree/main/corpus).

```python
from ebible_downloader import EBibleDownloader

downloader = EBibleDownloader()

# Download all files
downloader.download_all()

# Download only English files
downloader.download_all(filter_term="eng-")

# Download first 10 files only
downloader.download_all(max_files=10)

# Force re-download existing files
downloader.download_all(skip_existing=False)
```

### Context Query (`contextquery.py`)
Finds similar Bible verses for generating translation training examples using BM25 scoring and branching search.

```python
from query.contextquery import ContextQuery

cq = ContextQuery("eng-engULB.txt", "npi-npiulb.txt")
results = cq.search_by_text("blessed are the peacemakers", top_k=3)
```

**Return Format:**
Both `search_by_text()` and `search_by_line()` return `List[Tuple[int, str, str, float]]`:
```python
[
    (line_number, source_text, target_text, coverage),
    (1234, "Blessed are the peacemakers", "धन्या शान्ति निर्माता", 0.75),
    (5678, "Blessed are the merciful", "धन्या दयावान", 0.50),
    # ...
]
```
- `line_number`: 1-indexed line in original files
- `source_text`: matching verse in source language  
- `target_text`: corresponding verse in target language
- `coverage`: ratio of query words found (0.0-1.0)

**Algorithm:**
1. Uses BM25 to score verse relevance
2. Finds covered substrings in selected verses
3. Splits remaining query into branches
4. Searches branches in parallel until all examples found

### Example Usage (`example.py`)
Demonstrates finding translation context examples using English ULB and Nepali ULB:

```python
python example.py
```

## Benchmarks

The `benchmarks/` directory contains tools for evaluating translation quality using different ContextQuery configurations.

### Translation Benchmark (`benchmarks/translation_benchmark.py`)
Compares translation quality across different numbers of in-context examples using OpenAI's API.

**Setup:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
cp benchmarks/env.template .env
# Edit .env to add your OpenAI API key
```

**Basic Usage:**
```bash
cd benchmarks
python translation_benchmark.py
```

**Advanced Configuration:**
```bash
# Compare different example counts
python translation_benchmark.py --example-counts 1 3 5 10

# Test with more files and examples
python translation_benchmark.py --num-target-files 5 --num-tests-per-file 10

# Use different source file
python translation_benchmark.py --source-file your-source.txt
```

**Output:**
- Configuration summary
- Progress tracking with tqdm
- Similarity scores comparing translations to ground truth
- Performance rankings across different example counts

See `benchmarks/README.md` for detailed documentation.

## Installation

```bash
pip install -r requirements.txt
```

## Use Case

Generate few-shot learning examples for LLM translation training by finding verses similar to a target verse across different translations. The branching search ensures comprehensive coverage of complex queries while maintaining semantic coherence. 