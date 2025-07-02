import math
from typing import List, Dict
from collections import Counter
from .base import QueryBase

class BM25Query(QueryBase):
    def __init__(self, source_file: str, target_file: str, k1: float = 1.5, b: float = 0.75, verbose: bool = True):
        self.k1 = k1
        self.b = b
        super().__init__(source_file, target_file, verbose)
    
    def _preprocess(self):
        self.doc_lengths = {}
        self.term_freqs = {}
        self.doc_freqs = Counter()
        
        for idx in self.valid_indices:
            doc = self._normalize_text(self.source_verses[idx])
            words = doc.split()
            self.doc_lengths[idx] = len(words)
            self.term_freqs[idx] = Counter(words)
            
            for word in set(words):
                self.doc_freqs[word] += 1
        
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.total_docs = len(self.valid_indices)
    
    def _score_document(self, query_words: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        term_freq = self.term_freqs[doc_idx]
        
        for word in query_words:
            if word in term_freq:
                tf = term_freq[word]
                df = self.doc_freqs[word]
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score 