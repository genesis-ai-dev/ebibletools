# Universal Translation Quality Metrics

This module provides **language-agnostic** translation quality evaluation metrics that work with any language, script, or writing system - including low-resource languages without pre-trained models or linguistic resources.

## Philosophy: True Universality

All metrics in this module are:
- **Script-agnostic**: Work with Latin, Cyrillic, CJK, Arabic, Devanagari, etc.
- **Language-independent**: No tokenization, stemming, or dictionary dependencies
- **Resource-free**: No pre-trained models or linguistic databases required
- **Low-resource friendly**: Evaluate any language pair, even undocumented ones

## Universal Metrics

### 1. chrF+ (Character n-gram F-score Enhanced)
Character-level evaluation with enhanced processing.

```python
from metrics import chrF_plus

hypothesis = "The fast brown fox leaps over the sleepy dog."
reference = "The quick brown fox jumps over the lazy dog."

score = chrF_plus(hypothesis, reference)
print(f"chrF+: {score:.4f}")  # 0-1, higher is better
```

**Why Universal:**
- Character-level analysis works with any script
- Unicode normalization handles all writing systems  
- No tokenization or linguistic knowledge required
- Identical algorithm for all languages

**Technical Details:**
- Character n-grams (default n=6) with word n-grams (default n=2)
- Beta parameter balances precision vs recall (default=2.0)
- Unicode NFKC normalization for consistent character handling

### 2. Edit Distance (Normalized Levenshtein)
Pure character-level string comparison.

```python
from metrics import normalized_edit_distance

hypothesis = "The fast brown fox"
reference = "The quick brown fox"

# Raw distance (number of character edits)
raw = edit_distance(hypothesis, reference)
print(f"Raw edit distance: {raw}")

# Normalized (0-1, lower is better)
normalized = normalized_edit_distance(hypothesis, reference)
print(f"Normalized: {normalized:.4f}")
```

**Why Universal:**
- Pure character-level string comparison
- Works with any Unicode text
- No linguistic knowledge or models required
- Mathematical algorithm independent of language

**Technical Details:**
- Minimum edit operations (insert, delete, substitute)
- Normalized by maximum string length
- Dynamic programming implementation

### 3. TER (Translation Error Rate)
Minimum edit operations normalized by reference length.

```python
from metrics import ter_score

hypothesis = "The cat sits on mat."
reference = "The cat sat on the mat."

score = ter_score(hypothesis, reference)
print(f"TER: {score:.4f}")  # 0+, lower is better
```

**Why Universal:**
- Character/word-level edit operations
- Language-independent algorithm
- Works with any script or writing system
- No linguistic preprocessing required

**Technical Details:**
- Edit operations: insertions, deletions, substitutions
- Normalized by reference length
- Can operate at character or word level

## Installation

Minimal dependencies for universal metrics:

```bash
pip install numpy>=1.20.0
```

No language-specific libraries, models, or linguistic resources required.

## Removed Language-Dependent Metrics

The following metrics were **intentionally removed** due to language dependencies:

- **BLEU**: Requires language-specific tokenization rules
- **METEOR**: Uses English-only Porter stemmer and WordNet
- **ROUGE**: Depends on tokenization and optional stemming
- **BERTScore**: Requires pre-trained transformer models (unavailable for most languages)

These metrics work well for high-resource languages but fail for:
- Low-resource languages without linguistic tools
- Historical languages without modern NLP support  
- Code-mixed or multilingual text
- Languages with non-standard orthographies

## Usage Examples

### Basic Evaluation
```python
from metrics import chrF_plus, normalized_edit_distance, ter_score

hypothesis = "The cat is sleeping on the sofa."
reference = "The cat sleeps on the couch."

print(f"chrF+: {chrF_plus(hypothesis, reference):.4f}")
print(f"Edit Distance: {normalized_edit_distance(hypothesis, reference):.4f}")
print(f"TER: {ter_score(hypothesis, reference):.4f}")
```

### Cross-Script Evaluation
```python
# Works identically across any scripts
chinese_hyp = "猫在沙发上睡觉"
chinese_ref = "猫睡在沙发上"

arabic_hyp = "القطة تنام على الأريكة"
arabic_ref = "القطة نائمة على الأريكة"

# Same algorithm, same reliability
for hyp, ref in [(chinese_hyp, chinese_ref), (arabic_hyp, arabic_ref)]:
    print(f"chrF+: {chrF_plus(hyp, ref):.4f}")
    print(f"Edit: {normalized_edit_distance(hyp, ref):.4f}")
    print(f"TER: {ter_score(hyp, ref):.4f}")
```

### Batch Evaluation
```python
hypotheses = ["Translation 1", "Translation 2", "Translation 3"]
references = ["Reference 1", "Reference 2", "Reference 3"]

for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
    chrf = chrF_plus(hyp, ref)
    edit = normalized_edit_distance(hyp, ref)
    ter = ter_score(hyp, ref)
    print(f"Pair {i+1}: chrF+={chrf:.4f}, Edit={edit:.4f}, TER={ter:.4f}")
```

## Metric Characteristics

| Metric | Range | Best Value | Type | Granularity |
|--------|-------|------------|------|-------------|
| chrF+ | 0-1 | Higher | Character n-gram | Fine-grained |
| Edit Distance | 0-1 | Lower | Character-level | Fine-grained |
| TER | 0+ | Lower | Edit operations | Word/character |

## Research Applications

These universal metrics are ideal for:
- **Cross-linguistic studies**: Compare translation quality across language families
- **Low-resource MT**: Evaluate systems for under-resourced languages
- **Historical texts**: Evaluate translations of ancient/historical languages
- **Code-switching**: Handle multilingual or mixed-language content
- **Universal benchmarks**: Create language-independent evaluation standards

## Limitations

While universal, these metrics have trade-offs:
- **No semantic understanding**: Cannot detect meaning-preserving paraphrases
- **Surface-level**: Focus on form rather than meaning
- **Context-blind**: No discourse or pragmatic analysis

For high-resource languages with good linguistic tools, combining universal metrics with language-specific ones provides the most comprehensive evaluation. 