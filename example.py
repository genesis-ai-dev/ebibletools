#!/usr/bin/env python3

import os
from ebible_downloader import EBibleDownloader
from contextquery import ContextQuery

def format_examples(results, show_coverage=True):
    """Format search results nicely"""
    if not results:
        return "No similar verses found."
    
    output = []
    for line_num, source, target, coverage in results:
        coverage_pct = coverage * 100
        output.append(f"Line {line_num} (Coverage: {coverage_pct:.1f}%):" if show_coverage else f"Line {line_num}:")
        output.append(f"  Source: {source}")
        output.append(f"  Target: {target}")
        output.append("")
    
    return "\n".join(output)

def ensure_file_exists(downloader, filename):
    """Download file if it doesn't exist"""
    filepath = os.path.join("Corpus", filename)
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists")
        return True
    
    print(f"Downloading {filename}...")
    return downloader.download_file(filename)

def main():
    print("eBible Translation Context Examples")
    print("=" * 50)
    
    # Initialize downloader
    downloader = EBibleDownloader()
    
    # Download English and Spanish translations
    print("\n1. Downloading translations...")
    source_file = "eng-engULB.txt"
    target_file = "npi-npiulb.txt"
    
    if not ensure_file_exists(downloader, source_file):
        print(f"Failed to download {source_file}")
        return
    
    if not ensure_file_exists(downloader, target_file):
        print(f"Failed to download {target_file}")
        return
    
    # Initialize context query tool
    print(f"\n2. Loading translations...")
    source_path = os.path.join("Corpus", source_file)
    target_path = os.path.join("Corpus", target_file)
    
    query_tool = ContextQuery(source_path, target_path)
    
    # Example queries
    queries = [
        "love thy neighbor as thyself",
        "blessed are the peacemakers",
        "faith hope and love",
        "the lord is my shepherd"
    ]
    
    print(f"\n3. Finding translation examples...")
    
    for query in queries:
        print(f"\n{'='*60}\nQuery: '{query}'\n{'='*60}")
        results = query_tool.search_by_text(query, top_k=3)
        print(f"\nIn-Context Learning Examples:\n{'-' * 40}")
        print(format_examples(results))
    
    # Example using line number
    print(f"\n{'='*60}\nExample: Using specific verse as query\n{'='*60}")
    print(f"\nUsing line 25000 as query...")
    results = query_tool.search_by_line(25000, top_k=3)
    print(f"\nSimilar verses:\n{'-' * 40}")
    print(format_examples(results))

if __name__ == "__main__":
    main() 