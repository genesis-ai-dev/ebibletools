# eBible Tools

Tools for downloading and analyzing Bible translations for machine learning applications, particularly for generating in-context learning examples for translation training. Made this because I'm tired of remaking basically this from scratch every time I need it (:

## Components

### EBible Downloader (`ebible_downloader.py`)
Downloads parallel Bible translations from the [eBible corpus](https://github.com/BibleNLP/ebible/tree/main/corpus).

```python
from ebible_downloader import EBibleDownloader

downloader = EBibleDownloader()
downloader.list_files()  # Browse available translations
downloader.download_file("eng-engULB.txt")  # Download English ULB
```

### Context Query (`contextquery.py`)
Finds similar Bible verses for generating translation training examples using BM25 scoring and branching search.

```python
from contextquery import ContextQuery

cq = ContextQuery("eng-engULB.txt", "npi-npiulb.txt")
results = cq.search_by_text("blessed are the peacemakers", top_k=3)
```

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

## Installation

```bash
pip install -r requirements.txt
```

## Use Case

Generate few-shot learning examples for LLM translation training by finding verses similar to a target verse across different translations. The branching search ensures comprehensive coverage of complex queries while maintaining semantic coherence. 