"""
Improved YOLOv8 Training Script
Implements 3 key improvements:
1. Enhanced Data Augmentation
2. Higher Resolution Training (640 -> 800)
3. Class Balancing

These improvements address real-world challenges:
- Small object detection
- Weather/lighting variations
- Occlusion
- Class imbalance
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import yaml
import shutil

def check_gpu():
    """Check if GPU is available"""
    print("=" * 60)
    print("GPU Status Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ No GPU detected! Training will be very slow.")
        return False

def train_improved(epochs=100, batch_size=8, imgsz=800, resume=False, test_mode=False):
    """
    Train improved YOLOv8n model with enhancements
    
    Key Improvements:
    1. Data Augmentation: rotation, brightness, blur, noise
    2. Higher Resolution: 800x800 instead of 640x640
    3. Class Balancing: through augmentation weights
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size (reduced for higher resolution)
        imgsz: Image size (800 for better small object detection)
        resume: Resume from last checkpoint
        test_mode: Run quick test
    """
    
    # Check GPU
    has_gpu = check_gpu()
    device = 'cuda:0' if has_gpu else 'cpu'
    
    if test_mode:
        epochs = 5
        print("\n[TEST MODE] Training for 5 epochs only\n")
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Initializing IMPROVED YOLOv8n Model")
    print("=" * 60)
    print("\n📌 IMPROVEMENTS APPLIED:")
    print("  1. Enhanced Data Augmentation (rotation, blur, brightness)")
    print("  2. Higher Resolution Training (640 → 800)")
    print("  3. Advanced Mosaic & Mixup for class balancing")
    print()
    
    model = YOLO('yolov8n.pt')
    
    # IMPROVED Training parameters
    train_params = {
        'data': str(config_path),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,  # IMPROVEMENT 2: Higher resolution
        'device': device,
        'project': 'runs/improved',
        'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'patience': 25,
        'save': True,
        'plots': True,
        'verbose': True,
        
        # IMPROVEMENT 1: Enhanced Data Augmentation
        'hsv_h': 0.02,      # Increased HSV-Hue
        'hsv_s': 0.8,       # Increased saturation variation
        'hsv_v': 0.5,       # Increased value/brightness variation
        'degrees': 15.0,    # NEW: Rotation augmentation (±15°)
        'translate': 0.15,  # Increased translation
        'scale': 0.6,       # More scale variation
        'shear': 5.0,       # NEW: Shear augmentation
        'perspective': 0.001,  # NEW: Perspective distortion
        'flipud': 0.1,      # Slight vertical flip chance
        'fliplr': 0.5,      # Horizontal flip
        
        # IMPROVEMENT 3: Advanced augmentation for class balancing
        'mosaic': 1.0,      # Keep mosaic for multi-scale
        'mixup': 0.15,      # NEW: Mixup augmentation
        'copy_paste': 0.1,  # NEW: Copy-paste augmentation
        
        # Additional training improvements
        'cos_lr': True,     # Cosine learning rate scheduler
        'close_mosaic': 15, # Disable mosaic in last 15 epochs
        'warmup_epochs': 5, # Warmup epochs
    }
    
    print("\nTraining Parameters (IMPROVED):")
    print("-" * 40)
    for k, v in train_params.items():
        print(f"  {k}: {v}")
    print()
    
    # Start training
    results = model.train(**train_params)
    
    # Save training info
    save_training_info(results, train_params, "improved")
    
    print("\n" + "=" * 60)
    print("Improved Model Training Complete!")
    print("=" * 60)
    print(f"Best model saved at: runs/improved/{train_params['name']}/weights/best.pt")
    
    return results

def save_training_info(results, params, model_type):
    """Save training information for report"""
    
    info_dir = Path("runs") / model_type / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    
    info = {
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'parameters': params,
        'improvements': [
            "Enhanced Data Augmentation (rotation: 15°, shear: 5°, perspective)",
            "Higher Resolution Training (800x800 vs 640x640)",
            "Mixup & Copy-Paste Augmentation for class balancing",
            "Cosine Learning Rate Scheduler",
        ],
        'results': {
            'box_loss': float(results.results_dict.get('train/box_loss', 0)),
            'cls_loss': float(results.results_dict.get('train/cls_loss', 0)),
            'dfl_loss': float(results.results_dict.get('train/dfl_loss', 0)),
            'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        }
    }
    
    with open(info_dir / "training_info.yaml", 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Improved YOLOv8 Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (smaller for 800px)")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size (default 800)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--test", action="store_true", help="Test mode (5 epochs)")
    
    args = parser.parse_args()
    
    train_improved(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        resume=args.resume,
        test_mode=args.test
    )
