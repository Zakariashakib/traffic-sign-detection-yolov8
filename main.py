"""
Main Entry Point for Traffic Sign Detection Project
Provides unified interface for all operations
"""

import argparse
import subprocess
import sys
from pathlib import Path

def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     TRAFFIC SIGN DETECTION USING YOLOv8 WITH IMPROVEMENTS            ║
║                   Final Year Project                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def check_environment():
    """Check if environment is properly set up"""
    print("\n[Environment Check]")
    print("-" * 50)
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}", end=" ")
    if py_version.major >= 3 and py_version.minor >= 10:
        print("✓")
    else:
        print("⚠ (3.10+ recommended)")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} ✓")
            print(f"CUDA: {torch.version.cuda} ✓")
        else:
            print("GPU: Not available ⚠")
    except ImportError:
        print("PyTorch: Not installed ⚠")
    
    # Check Ultralytics
    try:
        from ultralytics import YOLO
        print("Ultralytics: Installed ✓")
    except ImportError:
        print("Ultralytics: Not installed ⚠")
    
    # Check dataset
    dataset_path = Path("datasets/GTSDB")
    if dataset_path.exists():
        train_images = len(list((dataset_path / "images" / "train").glob("*")))
        if train_images > 0:
            print(f"Dataset: Found ({train_images} training images) ✓")
        else:
            print("Dataset: Directory exists but empty ⚠")
    else:
        print("Dataset: Not found ⚠")
    
    print("-" * 50)

def main():
    print_header()
    
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --check           Check environment setup
  python main.py --download        Download dataset
  python main.py --train-baseline  Train baseline model
  python main.py --train-improved  Train improved model
  python main.py --evaluate        Compare models
  python main.py --visualize       Generate report figures
  python main.py --all             Run complete pipeline
        """
    )
    
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--train-baseline", action="store_true", help="Train baseline model")
    parser.add_argument("--train-improved", action="store_true", help="Train improved model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate and compare models")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--test", action="store_true", help="Test mode (5 epochs)")
    
    args = parser.parse_args()
    
    if args.check or len(sys.argv) == 1:
        check_environment()
        if len(sys.argv) == 1:
            parser.print_help()
        return
    
    if args.download or args.all:
        print("\n[Step 1] Downloading Dataset...")
        subprocess.run([sys.executable, "scripts/download_dataset.py"])
    
    if args.train_baseline or args.all:
        print("\n[Step 2] Training Baseline Model...")
        cmd = [sys.executable, "train_baseline.py", f"--epochs={args.epochs}"]
        if args.test:
            cmd.append("--test")
        subprocess.run(cmd)
    
    if args.train_improved or args.all:
        print("\n[Step 3] Training Improved Model...")
        cmd = [sys.executable, "train_improved.py", f"--epochs={args.epochs}"]
        if args.test:
            cmd.append("--test")
        subprocess.run(cmd)
    
    if args.evaluate or args.all:
        print("\n[Step 4] Evaluating Models...")
        subprocess.run([sys.executable, "evaluate.py"])
    
    if args.visualize or args.all:
        print("\n[Step 5] Generating Visualizations...")
        subprocess.run([sys.executable, "visualize_results.py"])
    
    if args.all:
        print("\n" + "=" * 70)
        print("COMPLETE PIPELINE FINISHED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check results in: runs/baseline/ and runs/improved/")
        print("2. Check figures in: report/figures/")
        print("3. Write your report using the generated data")

if __name__ == "__main__":
    main()
