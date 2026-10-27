"""
Model Evaluation Script
Compares baseline vs improved YOLOv8 models
Generates metrics for report
"""

import os
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import pandas as pd
import json
from datetime import datetime

def find_best_model(project_dir, direct_name=None):
    """Find the best.pt model in weights/ or in the most recent training run"""
    if direct_name:
        direct_path = Path("weights") / direct_name
        if direct_path.exists():
            return direct_path
            
    project_path = Path(project_dir)
    if not project_path.exists():
        return None
    
    # Get all training directories
    train_dirs = [d for d in project_path.iterdir() if d.is_dir() and d.name.startswith('train_')]
    
    if not train_dirs:
        return None
    
    # Sort by modification time and get latest
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    best_model = latest_dir / "weights" / "best.pt"
    
    return best_model if best_model.exists() else None

def evaluate_model(model_path, data_config, split='test'):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to model weights
        data_config: Path to data config yaml
        split: Dataset split to evaluate on
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating: {model_path}")
    print("-" * 50)
    
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_config,
        split=split,
        verbose=True,
        plots=True
    )
    
    # Extract metrics
    metrics = {
        'model': str(model_path),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6),
    }
    
    # Per-class metrics
    class_names = model.names
    per_class = {}
    for i, (p, r, ap50, ap) in enumerate(zip(
        results.box.p, results.box.r, results.box.ap50, results.box.ap
    )):
        per_class[class_names[i]] = {
            'precision': float(p),
            'recall': float(r),
            'AP50': float(ap50),
            'AP50-95': float(ap)
        }
    metrics['per_class'] = per_class
    
    # Speed metrics
    speed = results.speed
    metrics['speed'] = {
        'preprocess_ms': speed['preprocess'],
        'inference_ms': speed['inference'],
        'postprocess_ms': speed['postprocess'],
        'total_ms': speed['preprocess'] + speed['inference'] + speed['postprocess']
    }
    
    return metrics

def compare_models():
    """Compare baseline and improved models"""
    
    print("=" * 70)
    print("MODEL COMPARISON: Baseline vs Improved")
    print("=" * 70)
    
    config_path = Path(__file__).parent / "config.yaml"
    
    # Find models
    baseline_model = find_best_model("runs/baseline", "baseline_best.pt")
    improved_model = find_best_model("runs/improved", "improved_best.pt")
    
    results = {}
    
    if baseline_model:
        print("\n[1/2] Evaluating Baseline Model...")
        results['baseline'] = evaluate_model(baseline_model, str(config_path))
    else:
        print("\n⚠ Baseline model not found! Run train_baseline.py first.")
        results['baseline'] = None
    
    if improved_model:
        print("\n[2/2] Evaluating Improved Model...")
        results['improved'] = evaluate_model(improved_model, str(config_path))
    else:
        print("\n⚠ Improved model not found! Run train_improved.py first.")
        results['improved'] = None
    
    # Generate comparison report
    if results['baseline'] and results['improved']:
        generate_comparison_report(results)
    
    return results

def generate_comparison_report(results):
    """Generate comparison report for documentation"""
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    baseline = results['baseline']
    improved = results['improved']
    
    # Create comparison table
    comparison = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95', 'Inference (ms)'],
        'Baseline': [
            f"{baseline['precision']:.4f}",
            f"{baseline['recall']:.4f}",
            f"{baseline['f1_score']:.4f}",
            f"{baseline['mAP50']:.4f}",
            f"{baseline['mAP50-95']:.4f}",
            f"{baseline['speed']['inference_ms']:.2f}"
        ],
        'Improved': [
            f"{improved['precision']:.4f}",
            f"{improved['recall']:.4f}",
            f"{improved['f1_score']:.4f}",
            f"{improved['mAP50']:.4f}",
            f"{improved['mAP50-95']:.4f}",
            f"{improved['speed']['inference_ms']:.2f}"
        ],
        'Improvement': [
            f"+{(improved['precision'] - baseline['precision'])*100:.2f}%",
            f"+{(improved['recall'] - baseline['recall'])*100:.2f}%",
            f"+{(improved['f1_score'] - baseline['f1_score'])*100:.2f}%",
            f"+{(improved['mAP50'] - baseline['mAP50'])*100:.2f}%",
            f"+{(improved['mAP50-95'] - baseline['mAP50-95'])*100:.2f}%",
            f"{improved['speed']['inference_ms'] - baseline['speed']['inference_ms']:.2f}ms"
        ]
    }
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    # Save results
    output_dir = Path("runs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV for report
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    
    # Save detailed JSON
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Per-class comparison
    print("\n" + "-" * 70)
    print("PER-CLASS PERFORMANCE")
    print("-" * 70)
    
    class_comparison = []
    for class_name in baseline['per_class'].keys():
        base_ap = baseline['per_class'][class_name]['AP50']
        imp_ap = improved['per_class'][class_name]['AP50']
        class_comparison.append({
            'Class': class_name,
            'Baseline AP50': f"{base_ap:.4f}",
            'Improved AP50': f"{imp_ap:.4f}",
            'Improvement': f"+{(imp_ap - base_ap)*100:.2f}%"
        })
    
    df_class = pd.DataFrame(class_comparison)
    print("\n" + df_class.to_string(index=False))
    df_class.to_csv(output_dir / "per_class_comparison.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - comparison_results.csv (for report tables)")
    print("  - detailed_results.json (full metrics)")
    print("  - per_class_comparison.csv (per-class analysis)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and Compare Models")
    parser.add_argument("--baseline", type=str, help="Path to baseline model")
    parser.add_argument("--improved", type=str, help="Path to improved model")
    
    args = parser.parse_args()
    
    compare_models()
