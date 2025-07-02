import math
from typing import List, Dict
from collections import Counter
from .base import QueryBase

class TFIDFQuery(QueryBase):
    def __init__(self, source_file: str, target_file: str, verbose: bool = True):
        super().__init__(source_file, target_file, verbose)
    
    def _preprocess(self):
        self.term_freqs = {}
        self.doc_freqs = Counter()
        
        for idx in self.valid_indices:
            doc = self._normalize_text(self.source_verses[idx])
            words = doc.split()
            self.term_freqs[idx] = Counter(words)
            
            for word in set(words):
                self.doc_freqs[word] += 1
        
        self.total_docs = len(self.valid_indices)
    
    def _score_document(self, query_words: List[str], doc_idx: int) -> float:
        score = 0.0
        term_freq = self.term_freqs[doc_idx]
        doc_word_count = sum(term_freq.values())
        
        for word in query_words:
            if word in term_freq and word in self.doc_freqs:
                tf = term_freq[word] / doc_word_count
                idf = math.log(self.total_docs / self.doc_freqs[word])
                score += tf * idf
        
        return score 