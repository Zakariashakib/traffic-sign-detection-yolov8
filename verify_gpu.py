"""
GPU Environment Verification Script
Creates evidence for report - shows real GPU training setup
"""

import subprocess
import platform
import os
from datetime import datetime
from pathlib import Path

def create_evidence_folder():
    """Create folder for saving evidence screenshots and logs"""
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "screenshots").mkdir(exist_ok=True)
    (evidence_dir / "logs").mkdir(exist_ok=True)
    return evidence_dir

def check_nvidia_gpu():
    """Run nvidia-smi and save output"""
    print("=" * 70)
    print("🖥️  GPU CONFIGURATION CHECK")
    print("=" * 70)
    print()
    
    evidence_dir = create_evidence_folder()
    
    try:
        # Run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            gpu_info = result.stdout
            print(gpu_info)
            
            # Save to file for evidence
            with open(evidence_dir / "logs" / "nvidia_smi_output.txt", 'w') as f:
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 70 + "\n")
                f.write(gpu_info)
            
            print(f"\n✓ Saved to: evidence/logs/nvidia_smi_output.txt")
            print("\n📸 TAKE A SCREENSHOT NOW for your report!")
            
            return True
        else:
            print("❌ nvidia-smi failed. Check NVIDIA drivers.")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA GPU may not be available.")
        return False

def check_cuda():
    """Check CUDA installation"""
    print("\n" + "=" * 70)
    print("🔧 CUDA CONFIGURATION")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n✓ PyTorch Version: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"✓ GPU Count: {torch.cuda.device_count()}")
            
            return True
        return False
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_ultralytics():
    """Verify YOLOv8 installation"""
    print("\n" + "=" * 70)
    print("📦 YOLOV8 INSTALLATION CHECK")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"\n✓ Ultralytics Version: {ultralytics.__version__}")
        
        # Test model loading
        print("✓ Loading YOLOv8n... ", end="")
        model = YOLO('yolov8n.pt')
        print("Success!")
        
        return True
        
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_system_info():
    """Get system information for report"""
    print("\n" + "=" * 70)
    print("💻 SYSTEM INFORMATION")
    print("=" * 70)
    
    import psutil
    
    info = {
        'OS': f"{platform.system()} {platform.release()}",
        'Python': platform.python_version(),
        'CPU': platform.processor(),
        'RAM': f"{psutil.virtual_memory().total / 1e9:.1f} GB",
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Save system info
    evidence_dir = create_evidence_folder()
    with open(evidence_dir / "logs" / "system_info.txt", 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    return info

def generate_hardware_table():
    """Generate hardware table for report"""
    print("\n" + "=" * 70)
    print("📋 HARDWARE TABLE FOR REPORT")
    print("=" * 70)
    
    table = """
| Component | Specification |
|-----------|---------------|
| Operating System | [YOUR OS] |
| CPU | [YOUR CPU] |
| GPU | NVIDIA [YOUR GPU] |
| GPU Memory | [X] GB VRAM |
| RAM | [X] GB |
| CUDA Version | [X.X] |
| Python Version | [X.X] |
| PyTorch Version | [X.X] |

📌 Copy this to your report and fill in your values!
"""
    print(table)
    
    # Save template
    evidence_dir = create_evidence_folder()
    with open(evidence_dir / "hardware_table_template.md", 'w') as f:
        f.write(table)

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║            GPU & ENVIRONMENT VERIFICATION SCRIPT                      ║
║         For Teacher Evidence - Traffic Sign Detection                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    create_evidence_folder()
    
    # Run all checks
    gpu_ok = check_nvidia_gpu()
    cuda_ok = check_cuda()
    yolo_ok = check_ultralytics()
    get_system_info()
    generate_hardware_table()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  GPU (nvidia-smi): {'✓ Pass' if gpu_ok else '❌ Fail'}")
    print(f"  CUDA (PyTorch):   {'✓ Pass' if cuda_ok else '❌ Fail'}")
    print(f"  YOLOv8:           {'✓ Pass' if yolo_ok else '❌ Fail'}")
    
    if gpu_ok and cuda_ok and yolo_ok:
        print("\n🎉 All checks passed! You're ready to train.")
        print("\n📸 REMEMBER TO TAKE SCREENSHOTS OF THIS OUTPUT!")
    else:
        print("\n⚠️  Some checks failed. Please fix before training.")
    
    print("\n📁 Evidence saved to: evidence/")
    print("   - logs/nvidia_smi_output.txt")
    print("   - logs/system_info.txt")
    print("   - hardware_table_template.md")

if __name__ == "__main__":
    main()
