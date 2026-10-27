"""
GTSDB Dataset Preparation Script
Converts original GTSDB format to YOLO format
"""

import os
import csv
import shutil
import random
from pathlib import Path
from PIL import Image

# GTSDB class mapping (43 classes -> 4 super-categories)
CLASS_MAPPING = {
    # Prohibitory signs (0-9, 15-17)
    **{i: 0 for i in range(10)},
    **{i: 0 for i in range(15, 18)},
    
    # Mandatory signs (33-41)
    **{i: 1 for i in range(33, 42)},
    
    # Danger/Warning signs (11-31)
    **{i: 2 for i in range(11, 32)},
    
    # Other (10, 32, 42)
    10: 3, 32: 3, 42: 3
}

def convert_bbox_to_yolo(img_width, img_height, x1, y1, x2, y2):
    """Convert bounding box to YOLO format (normalized center x, y, width, height)"""
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def prepare_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """
    Prepare GTSDB dataset for YOLO training
    
    Args:
        source_dir: Path to original GTSDB dataset
        output_dir: Path to output YOLO format dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test is 1 - train - val)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Read annotations
    gt_file = source_dir / "gt.txt"
    if not gt_file.exists():
        print(f"Error: {gt_file} not found!")
        print("Please download the GTSDB dataset first.")
        return
    
    # Parse annotations
    annotations = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                filename = parts[0]
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                class_id = int(parts[5])
                
                if filename not in annotations:
                    annotations[filename] = []
                annotations[filename].append((x1, y1, x2, y2, class_id))
    
    # Get all images
    image_files = list(source_dir.glob("*.ppm")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    random.shuffle(image_files)
    
    # Split dataset
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    # Process each split
    stats = {'train': 0, 'val': 0, 'test': 0}
    for split, files in splits.items():
        for img_path in files:
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Convert to jpg and save
            new_name = img_path.stem + ".jpg"
            img.convert('RGB').save(output_dir / "images" / split / new_name)
            
            # Create label file
            label_file = output_dir / "labels" / split / (img_path.stem + ".txt")
            
            if img_path.name in annotations:
                with open(label_file, 'w') as f:
                    for x1, y1, x2, y2, orig_class in annotations[img_path.name]:
                        # Map to super-category
                        yolo_class = CLASS_MAPPING.get(orig_class, 3)
                        x_c, y_c, w, h = convert_bbox_to_yolo(img_width, img_height, x1, y1, x2, y2)
                        f.write(f"{yolo_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                stats[split] += 1
            else:
                # Create empty label file for images without signs
                label_file.touch()
    
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Train: {len(splits['train'])} images ({stats['train']} with annotations)")
    print(f"Val:   {len(splits['val'])} images ({stats['val']} with annotations)")
    print(f"Test:  {len(splits['test'])} images ({stats['test']} with annotations)")
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GTSDB dataset for YOLO")
    parser.add_argument("--source", type=str, default="./datasets/GTSDB_raw", 
                        help="Source directory with original GTSDB")
    parser.add_argument("--output", type=str, default="./datasets/GTSDB",
                        help="Output directory for YOLO format")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    if args.test:
        print("Test mode - checking paths...")
        print(f"Source: {args.source}")
        print(f"Output: {args.output}")
    else:
        prepare_dataset(args.source, args.output)
