
======================================================================
MULTI-TARGET TRANSLATION BENCHMARK RESULTS
======================================================================
Source file: eng-engULB.txt
Target files tested: 50
Total translation tests: 240
======================================================================
BASELINE    : 0.182
CONTEXTQUERY: 0.308 (+0.126, +69.0%, 165/240 wins)
BM25        : 0.293 (+0.111, +60.7%, 165/240 wins)
TFIDF       : 0.292 (+0.110, +60.5%, 168/240 wins)

Method wins across all tests:
BASELINE    : 32/240 (13.3%)
CONTEXTQUERY: 83/240 (34.6%)
BM25        : 62/240 (25.8%)
TFIDF       : 63/240 (26.2%)


======================================================================
SEARCH METHOD COMPARISON (5 Examples Each)
======================================================================
Source file: eng-engULB.txt
Target files tested: 50
Total translation tests: 245
Examples per translation: 5
======================================================================
TFIDF       : 0.296 (best)
CONTEXTQUERY: 0.293 (-0.003, -1.0% vs best)
BM25        : 0.289 (-0.007, -2.4% vs best)







## With 10 examples

Method wins across all tests:
BASELINE    : 32/240 (13.3%)
CONTEXTQUERY: 83/240 (34.6%)
BM25        : 62/240 (25.8%)
TFIDF       : 63/240 (26.2%)

## With 5 examples:

Head-to-head wins:
TFIDF       : 67/245 (27.3%)
CONTEXTQUERY: 108/245 (44.1%)
BM25        : 70/245 (28.6%)