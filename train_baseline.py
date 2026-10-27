"""
Baseline YOLOv8 Training Script
Trains standard YOLOv8n on GTSDB dataset
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import yaml

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
        print("  Please install CUDA and cuDNN for GPU support.")
        return False

def train_baseline(epochs=100, batch_size=16, imgsz=640, resume=False, test_mode=False):
    """
    Train baseline YOLOv8n model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
        resume: Resume from last checkpoint
        test_mode: Run quick test with 5 epochs
    """
    
    # Check GPU
    has_gpu = check_gpu()
    device = 'cuda:0' if has_gpu else 'cpu'
    
    # Adjust for test mode
    if test_mode:
        epochs = 5
        print("\n[TEST MODE] Training for 5 epochs only\n")
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    print("=" * 60)
    print("Initializing YOLOv8n Baseline Model")
    print("=" * 60)
    
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n
    
    # Training parameters
    train_params = {
        'data': str(config_path),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': 'runs/baseline',
        'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'patience': 20,
        'save': True,
        'plots': True,
        'verbose': True,
        
        # Default augmentation (baseline)
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 0.0,  # Rotation
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
    }
    
    print("\nTraining Parameters:")
    print("-" * 40)
    for k, v in train_params.items():
        print(f"  {k}: {v}")
    print()
    
    # Start training
    results = model.train(**train_params)
    
    # Save training info
    save_training_info(results, train_params, "baseline")
    
    print("\n" + "=" * 60)
    print("Baseline Training Complete!")
    print("=" * 60)
    print(f"Best model saved at: runs/baseline/{train_params['name']}/weights/best.pt")
    
    return results

def save_training_info(results, params, model_type):
    """Save training information for report"""
    
    info_dir = Path("runs") / model_type / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    
    info = {
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'parameters': params,
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
    
    parser = argparse.ArgumentParser(description="Train Baseline YOLOv8 Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--test", action="store_true", help="Test mode (5 epochs)")
    
    args = parser.parse_args()
    
    train_baseline(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        resume=args.resume,
        test_mode=args.test
    )
