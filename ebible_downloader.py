import os
import requests
from typing import List, Dict

class EBibleDownloader:
    def __init__(self):
        self.api_base = "https://api.github.com/repos/BibleNLP/ebible"
        self.raw_base = "https://raw.githubusercontent.com/BibleNLP/ebible/main"
        self.corpus_path = "corpus"
        self.output_dir = "Corpus"
        
    def list_files(self, filter_term: str = "") -> List[Dict[str, str]]:
        url = f"{self.api_base}/contents/{self.corpus_path}"
        response = requests.get(url)
        response.raise_for_status()
        
        files = response.json()
        txt_files = [
            {
                "name": file["name"],
                "download_url": file["download_url"],
                "size": file["size"]
            }
            for file in files
            if file["name"].endswith(".txt") and file["type"] == "file"
        ]
        
        if filter_term:
            txt_files = [f for f in txt_files if filter_term.lower() in f["name"].lower()]
        
        return sorted(txt_files, key=lambda x: x["name"])
    
    def download_file(self, filename: str) -> bool:
        url = f"{self.raw_base}/{self.corpus_path}/{filename}"
        response = requests.get(url)
        response.raise_for_status()
        
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded {filename} to {filepath}")
        return True
    
    def format_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB" 