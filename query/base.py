import os
import re
from typing import List, Tuple
from abc import ABC, abstractmethod

class QueryBase(ABC):
    def __init__(self, source_file: str, target_file: str, verbose: bool = True):
        self.source_file = source_file
        self.target_file = target_file
        self.verbose = verbose
        self.source_verses: List[str] = []
        self.target_verses: List[str] = []
        self.valid_indices: List[int] = []
        self._load_files()
        self._preprocess()
    
    def _load_files(self):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            self.source_verses = f.readlines()
        
        with open(self.target_file, 'r', encoding='utf-8') as f:
            self.target_verses = f.readlines()
        
        if len(self.source_verses) != len(self.target_verses):
            raise ValueError(f"Files have different lengths: {len(self.source_verses)} vs {len(self.target_verses)}")
        
        self.valid_indices = [
            i for i in range(len(self.source_verses))
            if self.source_verses[i].strip() and self.target_verses[i].strip()
        ]
        
        if self.verbose:
            print(f"Loaded {len(self.source_verses)} verses, {len(self.valid_indices)} valid pairs")
    
    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @abstractmethod
    def _preprocess(self):
        pass
    
    @abstractmethod
    def _score_document(self, query_words: List[str], doc_idx: int) -> float:
        pass
    
    def _simple_search(self, query_text: str, top_k: int, exclude_idx: int = -1) -> List[Tuple[int, str, str, float]]:
        query_words = self._normalize_text(query_text).split()
        
        scores = []
        for idx in self.valid_indices:
            if idx == exclude_idx:
                continue
            score = self._score_document(query_words, idx)
            scores.append((score, idx))
        
        scores.sort(reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            source_text = self.source_verses[idx].strip()
            target_text = self.target_verses[idx].strip()
            results.append((idx + 1, source_text, target_text, score))
        
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        return self._simple_search(query_text, top_k)
    
    def search_by_line(self, line_number: int, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        query_idx = line_number - 1
        query_text = self.source_verses[query_idx].strip()
        return self._simple_search(query_text, top_k, exclude_idx=query_idx) 