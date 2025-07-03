#!/usr/bin/env python3
"""
Analyze eBible corpus to find the most complete file per language
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def analyze_corpus(corpus_dir="Corpus"):
    """Analyze corpus files and find the most complete file per language"""
    
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory {corpus_dir} not found")
    
    # Group files by language code (first 3 characters before hyphen)
    language_files = defaultdict(list)
    
    for filepath in corpus_path.glob("*.txt"):
        filename = filepath.name
        
        # Extract language code (e.g., "eng" from "eng-engULB.txt")
        match = re.match(r'^([a-z]{3})-', filename)
        if match:
            lang_code = match.group(1)
            file_size = filepath.stat().st_size
            
            language_files[lang_code].append({
                'filename': filename,
                'filepath': str(filepath),
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024)
            })
    
    # Find the largest file for each language
    most_complete_files = {}
    
    for lang_code, files in language_files.items():
        # Sort by size (largest first)
        files_sorted = sorted(files, key=lambda x: x['size_bytes'], reverse=True)
        largest_file = files_sorted[0]
        
        most_complete_files[lang_code] = {
            'filename': largest_file['filename'],
            'size_mb': round(largest_file['size_mb'], 2),
            'alternatives': [
                {
                    'filename': f['filename'], 
                    'size_mb': round(f['size_mb'], 2)
                } 
                for f in files_sorted[1:]
            ]
        }
    
    return most_complete_files

def save_analysis(analysis, output_file="most_complete_per_language.json"):
    """Save analysis results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis saved to {output_file}")

def print_summary(analysis):
    """Print summary of analysis results"""
    print(f"\nFound {len(analysis)} languages in corpus:")
    print("=" * 60)
    
    # Sort by language code
    for lang_code in sorted(analysis.keys()):
        data = analysis[lang_code]
        print(f"{lang_code}: {data['filename']} ({data['size_mb']} MB)")
        
        if data['alternatives']:
            print(f"     Alternatives: {len(data['alternatives'])} files")
    
    print("=" * 60)
    print(f"Total most complete files: {len(analysis)}")
    
    # Statistics
    sizes = [data['size_mb'] for data in analysis.values()]
    print(f"Average size: {sum(sizes)/len(sizes):.2f} MB")
    print(f"Largest file: {max(sizes):.2f} MB")
    print(f"Smallest file: {min(sizes):.2f} MB")

def main():
    print("Analyzing eBible corpus...")
    
    try:
        analysis = analyze_corpus()
        print_summary(analysis)
        save_analysis(analysis)
        
        # Create list of filenames for downloader
        most_complete_filenames = [data['filename'] for data in analysis.values()]
        
        with open("most_complete_filenames.txt", 'w', encoding='utf-8') as f:
            for filename in sorted(most_complete_filenames):
                f.write(f"{filename}\n")
        
        print(f"\nFilename list saved to most_complete_filenames.txt")
        print(f"Ready to add download function to ebible_downloader.py")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 