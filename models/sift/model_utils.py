import numpy as np
import cv2
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

def setup_logger(name):
    """Setup and return a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    intersection = max(0, xB - xA) * max(0, yB - yA)
    union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - intersection
    
    return intersection / union if union > 0 else 0

def save_run_results(metrics, failures, run_name, base_dir='runs'):
    """Save evaluation results to runs directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save failures
    with open(run_dir / 'failures.json', 'w') as f:
        json.dump(failures, f, indent=4)
        
    return run_dir

def visualize_case(img_data, failure_data, class_id):
    """
    Visualize a failure case
    
    Args:
        img_data: Original image data dictionary containing path/file_name
        failure_data: Dictionary containing false_positives, false_negatives, low_iou for this image
        class_id: The class ID being analyzed
    """
    # Read image
    img = cv2.imread(img_data['file_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw false negatives (missed detections) in green
    for box in failure_data['false_negatives']:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'Missed {class_id}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Draw false positives (wrong detections) in red
    for box in failure_data['false_positives']:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f'False {class_id}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.title(f"Failures for Image {img_data['id']} - Class {class_id}\n" +
              f"False Positives: {len(failure_data['false_positives'])} " +
              f"False Negatives: {len(failure_data['false_negatives'])}")
    plt.axis('off')
    plt.show()

# def analyze_run(run_dir):
#     """Analyze failures from a run"""
#     run_dir = Path(run_dir)
    
#     # Load failures and metrics
#     with open(run_dir / 'failures.json', 'r') as f:
#         failures = json.load(f)
#     with open(run_dir / 'metrics.json', 'r') as f:
#         metrics = json.load(f)
        
#     return failures, metrics