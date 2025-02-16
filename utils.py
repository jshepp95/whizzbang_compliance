import cv2
import os
import shutil
import random
from sklearn.model_selection import train_test_split

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

def train_val_test_split(images_dir, labels_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, **kwargs):
    assert os.path.exists(images_dir) and os.path.exists(labels_dir), "Dataset path incorrect!"

    train_ratio = train_ratio
    val_ratio = val_ratio
    test_ratio = test_ratio

    all_images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]

    train_images, temp_images = train_test_split(all_images, train_size=train_ratio, random_state=42)

    val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    def move_files(file_list, dest_folder):
        """Moves images and corresponding labels to the correct split folder."""
        os.makedirs(os.path.join(images_dir, dest_folder), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, dest_folder), exist_ok=True)

        for file_name in file_list:
            shutil.move(
                os.path.join(images_dir, file_name), os.path.join(images_dir, dest_folder, file_name)
            )
            label_name = file_name.replace(".jpg", ".txt").replace(".png", ".txt")
            if os.path.exists(os.path.join(labels_dir, label_name)):
                shutil.move(
                    os.path.join(labels_dir, label_name), os.path.join(labels_dir, dest_folder, label_name)
                )

    move_files(train_images, "train")
    move_files(val_images, "val")
    move_files(test_images, "test")

    print(f"Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")