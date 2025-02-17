import cv2
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

def load_template(template_path, size):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template = scale_img(template, size)
    
    return template


def scale_img(image, target_size):
    """ Scale image while maintaining aspect ratio. """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def train_test_split_yolo(
    source_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """Split YOLO dataset into train/val/test maintaining image-label pairs"""
    
    # Setup paths
    source_path = Path(source_dir)
    original_images = source_path / "train" / "images"
    original_labels = source_path / "train" / "labels"
    
    # Create temporary directory for original files
    temp_dir = source_path / "original_data"
    temp_images = temp_dir / "images"
    temp_labels = temp_dir / "labels"
    
    # Move original files to temp
    temp_images.mkdir(parents=True, exist_ok=True)
    temp_labels.mkdir(parents=True, exist_ok=True)
    
    # Move all files to temp directory
    for img in original_images.iterdir():
        shutil.move(str(img), str(temp_images / img.name))
    for lbl in original_labels.iterdir():
        shutil.move(str(lbl), str(temp_labels / lbl.name))
    
    # Get all image files
    image_files = [f.name for f in temp_images.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']]
    
    # Create train/val/test splits
    train_images, temp_images_list = train_test_split(image_files, train_size=train_ratio, random_state=42)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_images, test_images = train_test_split(temp_images_list, train_size=val_ratio_adjusted, random_state=42)
    
    # Create directories
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split in splits:
        (source_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (source_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Move files to their respective splits
    for split, images in splits.items():
        for img in images:
            # Move image
            shutil.move(
                str(temp_dir / 'images' / img),
                str(source_path / split / 'images' / img)
            )
            
            # Move corresponding label
            label = img.rsplit('.', 1)[0] + '.txt'
            if (temp_dir / 'labels' / label).exists():
                shutil.move(
                    str(temp_dir / 'labels' / label),
                    str(source_path / split / 'labels' / label)
                )
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Update data.yaml
    yaml_path = source_path / 'data.yaml'
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    yaml_data.update({
        'train': './train/images',
        'val': './val/images',
        'test': './test/images'
    })
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    # Print summary
    print(f"Dataset split complete:")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")