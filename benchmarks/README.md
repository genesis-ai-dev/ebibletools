# Translation Benchmarks

Comprehensive translation quality evaluation framework using professional machine translation metrics.

## Features

- **Multiple Query Methods**: Compare BM25, TF-IDF, and context-aware search
- **Professional Metrics**: BLEU, chrF+, METEOR, ROUGE-L, BERTScore, Edit Distance, TER
- **Statistical Analysis**: Mean, standard deviation, and confidence intervals
- **Detailed Output**: JSON export with individual translation scores
- **Progress Tracking**: Real-time progress bars and ETA

## Translation Benchmark (`translation_benchmark.py`)

Evaluates translation quality using different numbers of in-context examples with comprehensive metrics.

### Quick Start

```bash
# Basic benchmark with default settings
python translation_benchmark.py

# Compare different query methods
python translation_benchmark.py --query-method bm25
python translation_benchmark.py --query-method tfidf  
python translation_benchmark.py --query-method context

# More comprehensive evaluation
python translation_benchmark.py --num-target-files 5 --num-tests-per-file 10 --example-counts 0 1 3 5 10
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--corpus-dir` | `../Corpus` | Directory containing corpus files |
| `--source-file` | `eng-engULB.txt` | Source language file |
| `--query-method` | `context` | Query method: `bm25`, `tfidf`, or `context` |
| `--num-target-files` | `2` | Number of target files to test |
| `--num-tests-per-file` | `5` | Number of test cases per file |
| `--example-counts` | `[0, 3, 5]` | Numbers of examples to compare |
| `--output` | None | Save detailed results to JSON file |

### Evaluation Metrics

The benchmark evaluates translations using seven professional metrics:

#### Primary Metrics
- **BLEU**: N-gram precision with brevity penalty (SacreBLEU implementation)
- **chrF+**: Character + word n-gram F-score with Unicode normalization
- **METEOR**: Exact matching, stemming, and WordNet synonyms
- **ROUGE-L**: Longest common subsequence F-score

#### Semantic Metrics  
- **BERTScore**: Transformer-based semantic similarity (DeBERTa-XLarge)

#### Distance Metrics
- **Edit Distance**: Normalized Levenshtein distance (inverted, higher = better)
- **TER**: Translation Error Rate (inverted, higher = better)

All metrics are normalized to 0-1 range where **higher scores indicate better translations**.

### Example Usage

#### Basic Evaluation
```bash
python translation_benchmark.py --query-method context --num-tests-per-file 10
```

#### Comprehensive Comparison
```bash
# Compare all query methods
python translation_benchmark.py --query-method bm25 --output results_bm25.json
python translation_benchmark.py --query-method tfidf --output results_tfidf.json  
python translation_benchmark.py --query-method context --output results_context.json

# Large-scale evaluation
python translation_benchmark.py \
  --num-target-files 10 \
  --num-tests-per-file 20 \
  --example-counts 0 1 3 5 10 15 \
  --output comprehensive_results.json
```

#### Custom Corpus
```bash
python translation_benchmark.py \
  --corpus-dir /path/to/corpus \
  --source-file custom-source.txt \
  --query-method context
```

### Output Format

#### Console Output
```
🔧 Translation Benchmark - CONTEXT Query
============================================================
Source: eng-engULB.txt
Target files: 2 files
Tests per file: 5
Example counts: [0, 3, 5]
Total tests: 30
Metrics: BLEU, chrF+, METEOR, ROUGE-L, BERTScore, Edit Dist, TER

Running evaluations: 100%|████████████| 30/30 [02:15<00:00, 4.5s/it]

================================================================================
COMPREHENSIVE TRANSLATION EVALUATION RESULTS
Query Method: CONTEXT
================================================================================

Metric        0 examples     3 examples     5 examples    
------------------------------------------------------------
BLEU         0.245±0.123   0.389±0.156   0.421±0.134  
chrF+        0.456±0.089   0.623±0.112   0.651±0.098  
METEOR       0.398±0.145   0.567±0.123   0.598±0.109  
ROUGE-L      0.423±0.134   0.589±0.145   0.612±0.132  
BERTScore    0.789±0.067   0.856±0.054   0.873±0.048  
Edit Dist    0.534±0.156   0.698±0.123   0.724±0.109  
TER          0.467±0.134   0.634±0.145   0.661±0.132  

Best Configurations:
BLEU        : 5 examples (0.421)
chrF+       : 5 examples (0.651)
METEOR      : 5 examples (0.598)
ROUGE-L     : 5 examples (0.612)
BERTScore   : 5 examples (0.873)
Edit Dist   : 5 examples (0.724)
TER         : 5 examples (0.661)

Overall Ranking:
 5 examples: 0.634 ⭐ (best overall)
 3 examples: 0.608 (-0.026)
 0 examples: 0.473 (-0.161)

✅ Benchmark completed successfully!
```

#### JSON Output Structure
```json
{
  "benchmark_config": {
    "query_method": "context",
    "source_file": "eng-engULB.txt", 
    "example_counts": [0, 3, 5],
    "metrics": ["BLEU", "chrF+", "METEOR", "ROUGE-L", "BERTScore", "Edit Dist", "TER"]
  },
  "summary_stats": {
    "5": {
      "BLEU": {
        "mean": 0.421,
        "std": 0.134,
        "min": 0.156,
        "max": 0.678,
        "count": 10
      }
    }
  },
  "detailed_results": [
    {
      "target_file": "spa-spalb.txt",
      "line_index": 1234,
      "source": "In the beginning was the Word",
      "reference": "En el principio era el Verbo", 
      "translation": "Al principio era la Palabra",
      "example_count": 5,
      "scores": {
        "BLEU": 0.456,
        "chrF+": 0.723,
        "METEOR": 0.634,
        "ROUGE-L": 0.667,
        "BERTScore": 0.892,
        "Edit Dist": 0.789,
        "TER": 0.712
      }
    }
  ]
}
```

### Performance Considerations

#### Speed vs Quality Trade-offs
- **BERTScore**: Slowest (~3-5s per evaluation) but most semantically aware
- **METEOR**: Medium speed (~0.5s) with linguistic processing
- **BLEU/chrF+/ROUGE**: Fast (~0.1s) and reliable
- **Edit/TER**: Fastest (~0.05s) for basic similarity

#### Optimization Tips
```bash
# For quick testing (fast metrics only)
python translation_benchmark.py --num-tests-per-file 3

# For semantic evaluation (includes BERTScore)
python translation_benchmark.py --num-tests-per-file 10

# For large-scale evaluation
python translation_benchmark.py \
  --num-target-files 20 \
  --num-tests-per-file 50 \
  --example-counts 0 3 5 10
```

### Interpreting Results

#### Metric Interpretation
- **BLEU**: 0.3+ = reasonable, 0.5+ = good, 0.7+ = excellent
- **chrF+**: Generally higher than BLEU, 0.5+ = good, 0.7+ = excellent  
- **METEOR**: Balanced metric, 0.4+ = reasonable, 0.6+ = good
- **BERTScore**: 0.8+ = good semantic similarity, 0.9+ = excellent
- **ROUGE-L**: Similar to BLEU but sequence-aware
- **Edit Distance**: 0.7+ = good character similarity
- **TER**: 0.6+ = reasonable, 0.8+ = good (inverted from standard TER)

#### Statistical Significance
- Standard deviations indicate consistency across test cases
- Lower std = more consistent performance
- Compare confidence intervals when scores are close

#### Best Practices
1. **Multiple Metrics**: Don't rely on single metric - use ensemble evaluation
2. **Statistical Power**: Use ≥20 test cases per configuration for reliability  
3. **Domain Specificity**: Some metrics work better for certain text types
4. **Baseline Comparison**: Always include 0-example baseline

### Integration Examples

#### Custom Evaluation Script
```python
from benchmarks.translation_benchmark import TranslationBenchmark

# Initialize benchmark
benchmark = TranslationBenchmark(
    api_key="your-key",
    corpus_dir="path/to/corpus", 
    source_file="source.txt",
    query_method="context"
)

# Run evaluation
benchmark.run_benchmark(
    num_target_files=5,
    num_tests_per_file=10, 
    example_counts=[0, 3, 5, 10],
    output_file="results.json"
)
```

#### Batch Comparison
```bash
#!/bin/bash
# Compare all query methods
for method in bm25 tfidf context; do
    echo "Testing $method..."
    python translation_benchmark.py \
        --query-method $method \
        --num-target-files 5 \
        --num-tests-per-file 20 \
        --output "results_${method}.json"
done
```

This comprehensive evaluation framework provides research-grade translation quality assessment suitable for academic publication and production systems.