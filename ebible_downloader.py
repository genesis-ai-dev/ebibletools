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
    
    def download_all(self, filter_term: str = "", max_files: int = None, 
                    skip_existing: bool = True) -> Dict[str, bool]:
        """Download all available files from the eBible corpus
        
        Args:
            filter_term: Only download files containing this term
            max_files: Limit number of files to download (None for all)
            skip_existing: Skip files that already exist locally
            
        Returns:
            Dict mapping filename to success status
        """
        print(f"Getting list of available files...")
        available_files = self.list_files(filter_term)
        
        if not available_files:
            print("No files found matching criteria.")
            return {}
        
        if max_files:
            available_files = available_files[:max_files]
            print(f"Limiting to first {max_files} files")
        
        print(f"Found {len(available_files)} files to download")
        
        # Calculate total size
        total_size = sum(f["size"] for f in available_files)
        print(f"Total download size: {self.format_size(total_size)}")
        
        # Confirm with user for large downloads
        if total_size > 100 * 1024 * 1024:  # 100MB
            response = input(f"This will download {self.format_size(total_size)}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return {}
        
        results = {}
        skipped = 0
        
        for i, file_info in enumerate(available_files, 1):
            filename = file_info["name"]
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if file already exists
            if skip_existing and os.path.exists(filepath):
                print(f"[{i}/{len(available_files)}] Skipping {filename} (already exists)")
                results[filename] = True
                skipped += 1
                continue
            
            print(f"[{i}/{len(available_files)}] Downloading {filename} ({self.format_size(file_info['size'])})")
            
            try:
                success = self.download_file(filename)
                results[filename] = success
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                results[filename] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        print(f"\n{'='*50}")
        print(f"DOWNLOAD SUMMARY")
        print(f"{'='*50}")
        print(f"Total files: {len(available_files)}")
        print(f"Successfully downloaded: {successful - skipped}")
        print(f"Skipped (already exist): {skipped}")
        print(f"Failed: {failed}")
        print(f"Final success rate: {successful}/{len(available_files)} ({successful/len(available_files)*100:.1f}%)")
        
        return results
    
    def format_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB" 