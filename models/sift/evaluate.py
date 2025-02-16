from collections import defaultdict
import cv2
from models.sift.model_utils import setup_logger, compute_iou, save_run_results

logger = setup_logger("evaluator")

class DetectorEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
    
    def evaluate(self, imgs, annotations, detector_fn, run_name="experiment"):
        """Evaluate detector performance and save results"""
        logger.info(f"Starting evaluation run: {run_name}")
        
        # Initialize statistics
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_scores": []})
        failures = defaultdict(lambda: {"false_positives": [], "false_negatives": [], "low_iou": []})

        # Evaluation loop
        for img_data in imgs:
            img_id = img_data["id"]
            img_path = img_data["file_name"]
            
            # Load and process image
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get ground truth and detections
            gt_bboxes = defaultdict(list)
            for ann in annotations.get(img_id, []):
                gt_bboxes[ann["category_id"]].append(ann["bbox"])
                
            detected_bboxes = detector_fn(img_gray)
            
            # Evaluate each class
            for class_id, gt_boxes in gt_bboxes.items():
                detected_boxes = detected_bboxes.get(class_id, [])
                matched = set()
                
                for det_box in detected_boxes:
                    best_iou, best_gt = 0, None
                    
                    for gt_box in gt_boxes:
                        iou = compute_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou, best_gt = iou, gt_box
                    
                    if best_iou >= self.iou_threshold:
                        class_stats[class_id]["tp"] += 1
                        class_stats[class_id]["iou_scores"].append(best_iou)
                        matched.add(tuple(best_gt))
                    else:
                        class_stats[class_id]["fp"] += 1
                        failures[img_id]["false_positives"].append(det_box)
                
                # Count false negatives
                for gt_box in gt_boxes:
                    if tuple(gt_box) not in matched:
                        class_stats[class_id]["fn"] += 1
                        failures[img_id]["false_negatives"].append(gt_box)
        
        # Calculate metrics
        metrics = {}
        for class_id, stats in class_stats.items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-6)
            recall = stats["tp"] / (stats["tp"] + stats["fn"] + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            metrics[class_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total_predictions": stats["tp"] + stats["fp"],
                "total_ground_truth": stats["tp"] + stats["fn"]
            }
            
            logger.info(f"Class {class_id}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Save results
        run_dir = save_run_results(metrics, failures, run_name)
        logger.info(f"Results saved to: {run_dir}")
        
        return metrics, failures