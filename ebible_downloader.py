import os
import requests
import random
import subprocess
from typing import List, Dict, Optional

class EBibleDownloader:
    def __init__(self, output_dir=None):
        self.api_base = "https://api.github.com/repos/BibleNLP/ebible"
        self.raw_base = "https://raw.githubusercontent.com/BibleNLP/ebible/main"
        self.corpus_path = "corpus"
        
        # If no output_dir specified, use Corpus in the same directory as this script
        if output_dir is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, "Corpus")
        else:
            self.output_dir = output_dir
        
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
    
    def download_random_ebible(self) -> Optional[str]:
        """Download a random eBible file"""
        files = self.list_files()
        if not files:
            return None
        
        filename = random.choice(files)["name"]
        self.download_file(filename)
        return filename
    
    def download_all(self, filter_term: str = "", max_files: Optional[int] = None, 
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
        total_size = sum(int(f["size"]) for f in available_files)
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
            
            print(f"[{i}/{len(available_files)}] Downloading {filename} ({self.format_size(int(file_info['size']))})")
            
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
    
    def download_most_complete_per_language(self, skip_existing: bool = True) -> Dict[str, bool]:
        """Download the most complete file for each language
        
        Args:
            skip_existing: Skip files that already exist locally
            
        Returns:
            Dict mapping filename to success status
        """
        import json
        from pathlib import Path
        
        # Try to load the analysis file
        analysis_file = Path("most_complete_per_language.json")
        if not analysis_file.exists():
            print("Analysis file not found. Running corpus analysis...")
            subprocess.run(["python", "analyze_corpus.py"], check=True)
        
        # Load the most complete files list
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        filenames = [data['filename'] for data in analysis.values()]
        
        print(f"Found {len(filenames)} most complete files (1 per language)")
        print(f"Starting download...")
        
        results = {}
        skipped = 0
        
        for i, filename in enumerate(filenames, 1):
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if file already exists
            if skip_existing and os.path.exists(filepath):
                print(f"[{i}/{len(filenames)}] Skipping {filename} (already exists)")
                results[filename] = True
                skipped += 1
                continue
            
            print(f"[{i}/{len(filenames)}] Downloading {filename}")
            
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
        print(f"MOST COMPLETE FILES DOWNLOAD SUMMARY")
        print(f"{'='*50}")
        print(f"Total languages: {len(filenames)}")
        print(f"Successfully downloaded: {successful - skipped}")
        print(f"Skipped (already exist): {skipped}")
        print(f"Failed: {failed}")
        print(f"Final success rate: {successful}/{len(filenames)} ({successful/len(filenames)*100:.1f}%)")
        
        return results

    def download_most_complete_for_language(self, language_code: str, skip_existing: bool = True) -> bool:
        """Download the most complete file for a specific language
        
        Args:
            language_code: 3-letter language code (e.g., 'eng', 'spa', 'fra')
            skip_existing: Skip if file already exists locally
            
        Returns:
            True if successful, False otherwise
        """
        import json
        from pathlib import Path
        
        # Normalize language code to lowercase
        language_code = language_code.lower()
        
        # Try to load the analysis file
        analysis_file = Path("most_complete_per_language.json")
        if not analysis_file.exists():
            print("Analysis file not found. Running corpus analysis...")
            subprocess.run(["python", "analyze_corpus.py"], check=True)
        
        # Load the most complete files list
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Check if language exists
        if language_code not in analysis:
            available_languages = sorted(analysis.keys())
            print(f"Language '{language_code}' not found.")
            print(f"Available languages: {', '.join(available_languages[:10])}...")
            print(f"Total available: {len(available_languages)} languages")
            return False
        
        # Get the filename for this language
        language_data = analysis[language_code]
        filename = language_data['filename']
        size_mb = language_data['size_mb']
        
        print(f"Language: {language_code}")
        print(f"Most complete file: {filename} ({size_mb} MB)")
        
        # Show alternatives if any
        if language_data.get('alternatives'):
            print(f"Alternatives available: {len(language_data['alternatives'])} files")
            for alt in language_data['alternatives'][:3]:  # Show first 3
                print(f"  - {alt['filename']} ({alt['size_mb']} MB)")
            if len(language_data['alternatives']) > 3:
                print(f"  ... and {len(language_data['alternatives']) - 3} more")
        
        # Check if file already exists
        filepath = os.path.join(self.output_dir, filename)
        if skip_existing and os.path.exists(filepath):
            print(f"✓ File already exists: {filepath}")
            return True
        
        # Download the file
        print(f"Downloading {filename}...")
        try:
            success = self.download_file(filename)
            if success:
                print(f"✓ Successfully downloaded {filename}")
            else:
                print(f"✗ Failed to download {filename}")
            return success
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False

    def list_available_languages(self, filter_term: str = "") -> None:
        """List all available languages with their most complete files
        
        Args:
            filter_term: Optional filter to search for specific languages
        """
        import json
        from pathlib import Path
        
        # Try to load the analysis file
        analysis_file = Path("most_complete_per_language.json")
        if not analysis_file.exists():
            print("Analysis file not found. Running corpus analysis...")
            subprocess.run(["python", "analyze_corpus.py"], check=True)
        
        # Load the analysis
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Filter if requested
        if filter_term:
            filtered = {lang: data for lang, data in analysis.items() 
                       if filter_term.lower() in lang.lower() or 
                          filter_term.lower() in data['filename'].lower()}
            print(f"Languages matching '{filter_term}':")
        else:
            filtered = analysis
            print("All available languages:")
        
        print("=" * 70)
        print(f"{'Lang':<6} {'File':<35} {'Size':<8} {'Alternatives'}")
        print("-" * 70)
        
        for lang_code in sorted(filtered.keys()):
            data = filtered[lang_code]
            alt_count = len(data.get('alternatives', []))
            alt_text = f"{alt_count} files" if alt_count > 0 else "none"
            
            print(f"{lang_code:<6} {data['filename']:<35} {data['size_mb']:>6.1f} MB  {alt_text}")
        
        print("=" * 70)
        print(f"Total: {len(filtered)} languages")
        
        if filter_term and not filtered:
            print(f"No languages found matching '{filter_term}'")

    def format_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB" 