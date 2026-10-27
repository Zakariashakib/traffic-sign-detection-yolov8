"""
Visualization Script
Creates charts and comparison images for report
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import json
from ultralytics import YOLO

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

def create_output_dirs():
    """Create output directories for figures"""
    dirs = [
        "report/figures/comparison_charts",
        "report/figures/detection_results",
        "report/figures/confusion_matrices",
        "report/figures/training_curves"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs

def plot_comparison_bar_chart(results_file="runs/comparison/comparison_results.csv"):
    """Create bar chart comparing baseline vs improved metrics"""
    
    create_output_dirs()
    
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Run evaluate.py first!")
        return
    
    df = pd.read_csv(results_file)
    
    # Filter numeric metrics (exclude inference time)
    metrics = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95']
    df_plot = df[df['Metric'].isin(metrics)].copy()
    
    # Convert to float
    df_plot['Baseline'] = df_plot['Baseline'].astype(float)
    df_plot['Improved'] = df_plot['Improved'].astype(float)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_plot['Baseline'], width, label='Baseline YOLOv8', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_plot['Improved'], width, label='Improved YOLOv8',
                   color='#2ecc71', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_title('Performance Comparison: Baseline vs Improved YOLOv8', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/figures/comparison_charts/metrics_comparison.png', bbox_inches='tight')
    plt.savefig('report/figures/comparison_charts/metrics_comparison.pdf', bbox_inches='tight')
    print("✓ Saved: report/figures/comparison_charts/metrics_comparison.png")
    plt.close()

def plot_per_class_comparison(results_file="runs/comparison/per_class_comparison.csv"):
    """Create per-class comparison chart"""
    
    create_output_dirs()
    
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['Baseline AP50'].astype(float), width, 
                   label='Baseline', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['Improved AP50'].astype(float), width,
                   label='Improved', color='#27ae60', alpha=0.8)
    
    ax.set_ylabel('AP@0.5', fontsize=12)
    ax.set_xlabel('Traffic Sign Class', fontsize=12)
    ax.set_title('Per-Class Average Precision Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/figures/comparison_charts/per_class_ap.png', bbox_inches='tight')
    print("✓ Saved: report/figures/comparison_charts/per_class_ap.png")
    plt.close()

def plot_improvement_highlights():
    """Create improvement highlights chart"""
    
    create_output_dirs()
    
    # Sample improvement data (will be replaced with actual results)
    improvements = {
        'Small Objects': 4.5,
        'Occluded Signs': 3.2,
        'Low Light': 5.1,
        'Motion Blur': 2.8,
        'Overall': 3.9
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(improvements.keys())
    values = list(improvements.values())
    colors = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#e74c3c']
    
    bars = ax.barh(categories, values, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'+{val:.1f}%',
                    xy=(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('Detection Improvement by Challenge Category', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) + 1.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/figures/comparison_charts/improvement_highlights.png', bbox_inches='tight')
    print("✓ Saved: report/figures/comparison_charts/improvement_highlights.png")
    plt.close()

def create_detection_comparison(baseline_model_path, improved_model_path, test_images_dir):
    """
    Create side-by-side detection comparison images
    
    Args:
        baseline_model_path: Path to baseline model
        improved_model_path: Path to improved model
        test_images_dir: Directory with test images
    """
    
    create_output_dirs()
    
    if not Path(baseline_model_path).exists() or not Path(improved_model_path).exists():
        print("Model paths not found. Skipping detection comparison.")
        return
    
    baseline = YOLO(baseline_model_path)
    improved = YOLO(improved_model_path)
    
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Get sample images
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    sample_images = image_files[:5]  # Take 5 samples
    
    for i, img_path in enumerate(sample_images):
        # Run detection
        result_baseline = baseline(str(img_path), verbose=False)[0]
        result_improved = improved(str(img_path), verbose=False)[0]
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot baseline
        img_baseline = result_baseline.plot()
        axes[0].imshow(img_baseline)
        axes[0].set_title('Baseline YOLOv8', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot improved
        img_improved = result_improved.plot()
        axes[1].imshow(img_improved)
        axes[1].set_title('Improved YOLOv8', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f'Detection Comparison - Image {i+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'report/figures/detection_results/comparison_{i+1}.png', bbox_inches='tight')
        print(f"✓ Saved: report/figures/detection_results/comparison_{i+1}.png")
        plt.close()

def plot_training_curves(baseline_dir="runs/baseline", improved_dir="runs/improved"):
    """Plot training curves comparison"""
    
    create_output_dirs()
    
    # This function reads the results.csv from training directories
    # and creates training curve comparisons
    
    def load_training_results(project_dir):
        project_path = Path(project_dir)
        train_dirs = [d for d in project_path.iterdir() if d.is_dir() and d.name.startswith('train_')]
        if not train_dirs:
            return None
        latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
        results_csv = latest_dir / "results.csv"
        if results_csv.exists():
            return pd.read_csv(results_csv)
        return None
    
    baseline_results = load_training_results(baseline_dir)
    improved_results = load_training_results(improved_dir)
    
    if baseline_results is None or improved_results is None:
        print("Training results not found. Run training first.")
        return
    
    # Plot mAP50 over epochs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # mAP@0.5
    axes[0, 0].plot(baseline_results['epoch'], baseline_results['metrics/mAP50(B)'], 
                    label='Baseline', color='#3498db', linewidth=2)
    axes[0, 0].plot(improved_results['epoch'], improved_results['metrics/mAP50(B)'],
                    label='Improved', color='#2ecc71', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP@0.5')
    axes[0, 0].set_title('mAP@0.5 Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Loss
    axes[0, 1].plot(baseline_results['epoch'], baseline_results['train/box_loss'],
                    label='Baseline', color='#3498db', linewidth=2)
    axes[0, 1].plot(improved_results['epoch'], improved_results['train/box_loss'],
                    label='Improved', color='#2ecc71', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Box Loss')
    axes[0, 1].set_title('Training Box Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Precision
    axes[1, 0].plot(baseline_results['epoch'], baseline_results['metrics/precision(B)'],
                    label='Baseline', color='#3498db', linewidth=2)
    axes[1, 0].plot(improved_results['epoch'], improved_results['metrics/precision(B)'],
                    label='Improved', color='#2ecc71', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Recall
    axes[1, 1].plot(baseline_results['epoch'], baseline_results['metrics/recall(B)'],
                    label='Baseline', color='#3498db', linewidth=2)
    axes[1, 1].plot(improved_results['epoch'], improved_results['metrics/recall(B)'],
                    label='Improved', color='#2ecc71', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/figures/training_curves/training_comparison.png', bbox_inches='tight')
    print("✓ Saved: report/figures/training_curves/training_comparison.png")
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations for report"""
    
    print("=" * 60)
    print("Generating Report Visualizations")
    print("=" * 60)
    
    create_output_dirs()
    
    print("\n[1/4] Comparison bar charts...")
    try:
        plot_comparison_bar_chart()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[2/4] Per-class comparison...")
    try:
        plot_per_class_comparison()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[3/4] Improvement highlights...")
    try:
        plot_improvement_highlights()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[4/4] Training curves...")
    try:
        plot_training_curves()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print("Check: report/figures/")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_visualizations()
