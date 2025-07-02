import re
import unicodedata
from collections import Counter
from typing import List, Tuple

def _normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """Normalize text for chrF evaluation"""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    if remove_punctuation:
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def _extract_char_ngrams(text: str, n: int, whitespace: bool = True) -> List[str]:
    """Extract character n-grams from text"""
    if not whitespace:
        # Remove spaces for character-only n-grams
        text = re.sub(r'\s+', '', text)
    
    text = text.strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def _extract_word_ngrams(text: str, n: int) -> List[str]:
    """Extract word n-grams from text"""
    words = text.strip().split()
    if len(words) < n:
        return [' '.join(words)] if words else []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def _calculate_f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-score with given beta"""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def chrF(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0, 
         remove_whitespace: bool = False, remove_punctuation: bool = True) -> float:
    """
    Calculate chrF score (character n-gram F-score)
    
    Args:
        hypothesis: Generated translation
        reference: Ground truth translation
        n: Maximum character n-gram length
        beta: F-score beta parameter (default 2.0 gives more weight to recall)
        remove_whitespace: Whether to remove whitespace from character n-grams
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        chrF score between 0 and 1
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    
    # Normalize texts
    hyp_norm = _normalize_text(hypothesis, remove_punctuation)
    ref_norm = _normalize_text(reference, remove_punctuation)
    
    total_precision = 0.0
    total_recall = 0.0
    valid_n = 0
    
    for i in range(1, n + 1):
        hyp_ngrams = Counter(_extract_char_ngrams(hyp_norm, i, not remove_whitespace))
        ref_ngrams = Counter(_extract_char_ngrams(ref_norm, i, not remove_whitespace))
        
        if not hyp_ngrams or not ref_ngrams:
            continue
            
        matches = sum((hyp_ngrams & ref_ngrams).values())
        precision = matches / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
        recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        
        total_precision += precision
        total_recall += recall
        valid_n += 1
    
    if valid_n == 0:
        return 0.0
    
    avg_precision = total_precision / valid_n
    avg_recall = total_recall / valid_n
    
    return _calculate_f_score(avg_precision, avg_recall, beta)

def chrF_plus(hypothesis: str, reference: str, n_char: int = 6, n_word: int = 2, 
              beta: float = 2.0, remove_punctuation: bool = True) -> float:
    """
    Calculate chrF+ score (character + word n-grams)
    
    Args:
        hypothesis: Generated translation
        reference: Ground truth translation
        n_char: Maximum character n-gram length
        n_word: Maximum word n-gram length
        beta: F-score beta parameter
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        chrF+ score between 0 and 1
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    
    # Normalize texts
    hyp_norm = _normalize_text(hypothesis, remove_punctuation)
    ref_norm = _normalize_text(reference, remove_punctuation)
    
    total_precision = 0.0
    total_recall = 0.0
    total_n = 0
    
    # Character n-grams
    for i in range(1, n_char + 1):
        hyp_ngrams = Counter(_extract_char_ngrams(hyp_norm, i, True))
        ref_ngrams = Counter(_extract_char_ngrams(ref_norm, i, True))
        
        if hyp_ngrams and ref_ngrams:
            matches = sum((hyp_ngrams & ref_ngrams).values())
            precision = matches / sum(hyp_ngrams.values())
            recall = matches / sum(ref_ngrams.values())
            
            total_precision += precision
            total_recall += recall
            total_n += 1
    
    # Word n-grams
    for i in range(1, n_word + 1):
        hyp_ngrams = Counter(_extract_word_ngrams(hyp_norm, i))
        ref_ngrams = Counter(_extract_word_ngrams(ref_norm, i))
        
        if hyp_ngrams and ref_ngrams:
            matches = sum((hyp_ngrams & ref_ngrams).values())
            precision = matches / sum(hyp_ngrams.values())
            recall = matches / sum(ref_ngrams.values())
            
            total_precision += precision
            total_recall += recall
            total_n += 1
    
    if total_n == 0:
        return 0.0
    
    avg_precision = total_precision / total_n
    avg_recall = total_recall / total_n
    
    return _calculate_f_score(avg_precision, avg_recall, beta)

def chrF_plus_plus(hypothesis: str, reference: str, n_char: int = 6, n_word: int = 2, beta: float = 2.0) -> float:
    """
    Calculate chrF++ score (character + word n-grams with enhanced whitespace handling)
    
    Args:
        hypothesis: Generated translation
        reference: Ground truth translation
        n_char: Maximum character n-gram length
        n_word: Maximum word n-gram length
        beta: F-score beta parameter
    
    Returns:
        chrF++ score between 0 and 1
    """
    # Enhanced normalization for chrF++
    hyp_normalized = re.sub(r'\s+', ' ', hypothesis.strip())
    ref_normalized = re.sub(r'\s+', ' ', reference.strip())
    
    return chrF_plus(hyp_normalized, ref_normalized, n_char, n_word, beta, remove_punctuation=False) 