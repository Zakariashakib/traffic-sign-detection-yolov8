"""
GTSDB Dataset Downloader
Downloads German Traffic Sign Detection Benchmark dataset
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_gtsdb():
    """Download GTSDB dataset from official source"""
    
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "datasets" / "GTSDB"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # GTSDB URLs (Using Roboflow pre-formatted YOLO dataset)
    # This is easier to use and already in YOLO format
    print("=" * 60)
    print("GTSDB Dataset Download Instructions")
    print("=" * 60)
    print()
    print("Option 1: Download from Roboflow (Recommended - Pre-formatted)")
    print("-" * 60)
    print("1. Visit: https://universe.roboflow.com/gtsdb-9stsk/traffic-signs-detection/dataset/1")
    print("2. Export as 'YOLOv8' format")
    print("3. Extract to: datasets/GTSDB/")
    print()
    print("Option 2: Download Original GTSDB")
    print("-" * 60)
    print("1. Visit: https://benchmark.ini.rub.de/gtsdb_news.html")
    print("2. Download 'FullIJCNN2013.zip'")
    print("3. Run prepare_dataset.py to convert")
    print()
    print("Option 3: Use Kaggle Dataset")
    print("-" * 60)
    print("1. Install: pip install kaggle")
    print("2. Download: kaggle datasets download -d safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb")
    print()
    
    # Create directory structure
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure at: {dataset_dir}")
    print()
    print("After downloading, your structure should look like:")
    print("""
datasets/GTSDB/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
    """)

if __name__ == "__main__":
    download_gtsdb()
