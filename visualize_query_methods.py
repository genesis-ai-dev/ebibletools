#!/usr/bin/env python3
"""
Query Method Comparison Visualization
Compares performance of bm25, tfidf, and context query methods
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_overall_scores(data):
    """Extract overall scores for visualization"""
    query_method = data['query_method']
    
    # Get overall scores for the example count (should be 12)
    summary = data['summary']
    
    scores = {}
    for model, model_data in summary.items():
        overall_data = model_data['overall']
        # Get the scores for 12 examples (the only example count in your case)
        example_count = 12
        if example_count in overall_data:
            scores[model] = overall_data[example_count]
        else:
            # Fallback: get first available example count
            first_count = list(overall_data.keys())[0]
            scores[model] = overall_data[first_count]
    
    return query_method, scores

def create_comparison_plots(results_files, output_dir="plots"):
    """Create comparison plots for query methods"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load all results
    all_data = {}
    for file_path in results_files:
        if Path(file_path).exists():
            data = load_results(file_path)
            method, scores = extract_overall_scores(data)
            all_data[method] = scores
        else:
            print(f"⚠️  File not found: {file_path}")
    
    if not all_data:
        print("❌ No valid result files found!")
        return
    
    print(f"📊 Creating visualizations for {len(all_data)} query methods")
    
    # Get all models
    all_models = set()
    for method_data in all_data.values():
        all_models.update(method_data.keys())
    all_models = sorted(list(all_models))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Query Method Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. chrF+ scores comparison
    ax1 = axes[0, 0]
    methods = list(all_data.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    for i, model in enumerate(all_models):
        chrf_means = []
        chrf_stds = []
        
        for method in methods:
            if model in all_data[method]:
                chrf_means.append(all_data[method][model]['chrf_mean'])
                chrf_stds.append(all_data[method][model]['chrf_std'])
            else:
                chrf_means.append(0)
                chrf_stds.append(0)
        
        offset = (i - len(all_models)/2 + 0.5) * width / len(all_models)
        ax1.bar(x + offset, chrf_means, width/len(all_models), 
                label=model, yerr=chrf_stds, capsize=3, alpha=0.8)
    
    ax1.set_xlabel('Query Method')
    ax1.set_ylabel('chrF+ Score')
    ax1.set_title('chrF+ Scores by Query Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Edit distance scores comparison
    ax2 = axes[0, 1]
    for i, model in enumerate(all_models):
        edit_means = []
        edit_stds = []
        
        for method in methods:
            if model in all_data[method]:
                edit_means.append(all_data[method][model]['edit_mean'])
                edit_stds.append(all_data[method][model]['edit_std'])
            else:
                edit_means.append(0)
                edit_stds.append(0)
        
        offset = (i - len(all_models)/2 + 0.5) * width / len(all_models)
        ax2.bar(x + offset, edit_means, width/len(all_models), 
                label=model, yerr=edit_stds, capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Query Method')
    ax2.set_ylabel('Edit Distance Score')
    ax2.set_title('Edit Distance Scores by Query Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of chrF+ scores
    ax3 = axes[1, 0]
    heatmap_data = []
    model_labels = []
    
    for model in all_models:
        row = []
        for method in methods:
            if model in all_data[method]:
                row.append(all_data[method][model]['chrf_mean'])
            else:
                row.append(0)
        heatmap_data.append(row)
        model_labels.append(model)
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods)
    ax3.set_yticks(range(len(model_labels)))
    ax3.set_yticklabels(model_labels)
    ax3.set_title('chrF+ Score Heatmap')
    
    # Add text annotations
    for i in range(len(model_labels)):
        for j in range(len(methods)):
            text = ax3.text(j, i, f'{heatmap_data[i][j]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax3)
    
    # 4. Performance ranking
    ax4 = axes[1, 1]
    
    # Calculate average performance across models for each method
    method_averages = {}
    for method in methods:
        chrf_scores = []
        for model in all_models:
            if model in all_data[method]:
                chrf_scores.append(all_data[method][model]['chrf_mean'])
        method_averages[method] = np.mean(chrf_scores) if chrf_scores else 0
    
    # Sort methods by performance
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)
    
    method_names = [item[0] for item in sorted_methods]
    avg_scores = [item[1] for item in sorted_methods]
    
    bars = ax4.bar(method_names, avg_scores, alpha=0.8)
    ax4.set_xlabel('Query Method')
    ax4.set_ylabel('Average chrF+ Score')
    ax4.set_title('Average Performance Ranking')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plots
    plot_file = Path(output_dir) / "query_method_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {plot_file}")
    
    # Create a detailed summary table
    create_summary_table(all_data, output_dir)
    
    plt.show()

def create_summary_table(all_data, output_dir):
    """Create a detailed summary table"""
    methods = list(all_data.keys())
    all_models = set()
    for method_data in all_data.values():
        all_models.update(method_data.keys())
    all_models = sorted(list(all_models))
    
    # Create summary text
    summary_lines = []
    summary_lines.append("QUERY METHOD PERFORMANCE COMPARISON")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    for model in all_models:
        summary_lines.append(f"🤖 {model}:")
        summary_lines.append("-" * 30)
        
        for method in methods:
            if model in all_data[method]:
                data = all_data[method][model]
                chrf = data['chrf_mean']
                edit = data['edit_mean']
                num_tests = data['num_tests']
                summary_lines.append(f"  {method:8}: chrF+ {chrf:.3f}, Edit {edit:.3f} ({num_tests} tests)")
            else:
                summary_lines.append(f"  {method:8}: No data")
        summary_lines.append("")
    
    # Overall ranking
    summary_lines.append("📈 OVERALL RANKING (by average chrF+):")
    summary_lines.append("-" * 35)
    
    method_averages = {}
    for method in methods:
        chrf_scores = []
        for model in all_models:
            if model in all_data[method]:
                chrf_scores.append(all_data[method][model]['chrf_mean'])
        method_averages[method] = np.mean(chrf_scores) if chrf_scores else 0
    
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, avg_score) in enumerate(sorted_methods, 1):
        summary_lines.append(f"{i}. {method}: {avg_score:.3f}")
    
    # Save summary
    summary_file = Path(output_dir) / "query_method_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"📋 Summary saved to: {summary_file}")
    
    # Print to console as well
    print("\n" + '\n'.join(summary_lines))

def main():
    # Default file names based on your description
    result_files = [
        "results.json",
        "results2.json", 
        "results3.json"
    ]
    
    print("🎨 Query Method Visualization Tool")
    print("=" * 40)
    
    # Check which files exist
    existing_files = [f for f in result_files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No result files found!")
        print("Expected files: results.json, results2.json, results3.json")
        print("Make sure these files are in the current directory.")
        return 1
    
    print(f"📁 Found {len(existing_files)} result files:")
    for f in existing_files:
        print(f"  - {f}")
    
    create_comparison_plots(existing_files)
    
    print("\n✅ Visualization completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 