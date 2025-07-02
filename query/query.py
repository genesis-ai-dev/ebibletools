import math
from typing import List, Tuple, Set
from collections import Counter
from .base import QueryBase

class Query(QueryBase):
    def __init__(self, source_file: str, target_file: str, method: str = "bm25", 
                 coverage_weight: float = 0.5, k1: float = 1.5, b: float = 0.75, verbose: bool = True):
        self.method = method.lower()
        self.coverage_weight = coverage_weight
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
        if self.method == "bm25":
            return self._bm25_score(query_words, doc_idx)
        elif self.method == "tfidf":
            return self._tfidf_score(query_words, doc_idx)
        elif self.method == "context":
            return self._bm25_score(query_words, doc_idx)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _bm25_score(self, query_words: List[str], doc_idx: int) -> float:
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
    
    def _tfidf_score(self, query_words: List[str], doc_idx: int) -> float:
        score = 0.0
        term_freq = self.term_freqs[doc_idx]
        doc_word_count = sum(term_freq.values())
        
        for word in query_words:
            if word in term_freq and word in self.doc_freqs:
                tf = term_freq[word] / doc_word_count
                idf = math.log(self.total_docs / self.doc_freqs[word])
                score += tf * idf
        
        return score
    
    def _find_covered_substring(self, query_text: str, verse_text: str) -> str:
        query_words = self._normalize_text(query_text).split()
        verse_words = set(self._normalize_text(verse_text).split())
        
        longest_substring = ""
        for i in range(len(query_words)):
            for j in range(i + 1, len(query_words) + 1):
                substring_words = query_words[i:j]
                if all(word in verse_words for word in substring_words):
                    substring = ' '.join(substring_words)
                    if len(substring) > len(longest_substring):
                        longest_substring = substring
        
        return longest_substring
    
    def _remove_substring_and_split(self, query_text: str, covered_substring: str) -> List[str]:
        if not covered_substring:
            return []
        
        query_norm = self._normalize_text(query_text)
        covered_norm = self._normalize_text(covered_substring)
        
        remaining = query_norm.replace(covered_norm, ' | ')
        parts = [part.strip() for part in remaining.split('|') if part.strip()]
        
        return [part for part in parts if len(part.split()) >= 1]
    
    def _compute_coverage(self, original_query: str, verse_text: str) -> float:
        query_words = set(self._normalize_text(original_query).split())
        verse_words = set(self._normalize_text(verse_text).split())
        covered = query_words.intersection(verse_words)
        return len(covered) / len(query_words) if query_words else 0.0
    
    def _branching_search(self, original_query: str, top_k: int, exclude_idx: int = -1) -> List[Tuple[int, str, str, float]]:
        results: List[Tuple[int, str, str, float]] = []
        query_branches = [self._normalize_text(original_query)]
        used_indices = {exclude_idx} if exclude_idx >= 0 else set()
        restart_count = 0
        
        while len(results) < top_k and restart_count < 3:
            if not query_branches:
                restart_count += 1
                query_branches = [self._normalize_text(original_query)]
                continue
            
            best_score = -1.0
            best_idx = -1
            best_branch_idx = -1
            
            for branch_idx, branch_query in enumerate(query_branches):
                if not branch_query.strip():
                    continue
                    
                query_words = branch_query.split()
                
                for idx in self.valid_indices:
                    if idx in used_indices:
                        continue
                    
                    bm25_val = self._bm25_score(query_words, idx)
                    coverage = self._compute_coverage(branch_query, self.source_verses[idx])
                    
                    score = bm25_val * (1 + self.coverage_weight * coverage)
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_branch_idx = branch_idx
            
            if best_idx == -1:
                break
                
            source_text = self.source_verses[best_idx].strip()
            target_text = self.target_verses[best_idx].strip()
            coverage = self._compute_coverage(original_query, source_text)
            
            results.append((best_idx + 1, source_text, target_text, coverage))
            used_indices.add(best_idx)
            
            current_branch = query_branches[best_branch_idx]
            covered_substring = self._find_covered_substring(current_branch, source_text)
            
            query_branches.pop(best_branch_idx)
            
            if covered_substring:
                new_branches = self._remove_substring_and_split(current_branch, covered_substring)
                query_branches.extend(new_branches)
        
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        if self.method == "context":
            return self._branching_search(query_text, top_k)
        else:
            return self._simple_search(query_text, top_k)
    
    def search_by_line(self, line_number: int, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        query_idx = line_number - 1
        query_text = self.source_verses[query_idx].strip()
        
        if self.method == "context":
            return self._branching_search(query_text, top_k, exclude_idx=query_idx)
        else:
            return self._simple_search(query_text, top_k, exclude_idx=query_idx) 