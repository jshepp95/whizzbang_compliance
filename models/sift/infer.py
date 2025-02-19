import cv2
import numpy as np
from models.sift.model_utils import setup_logger

logger = setup_logger("sift_detector")

class SIFTDetector:
    def __init__(self, templates):
        self.templates = templates
        self.sift = cv2.SIFT_create()
        self.template_features = {
            class_id: self.sift.detectAndCompute(template, None) 
            for class_id, template in templates.items()
        }
        
        FLANN_IDX_TREE = 1
        self.index_params = dict(algorithm=FLANN_IDX_TREE, trees=5)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.MATCH_THRESHOLD = 0.75

    def detect_objects(self, img_gray):
        detected_bboxes = {}
        
        kp_tgt, desc_tgt = self.sift.detectAndCompute(img_gray, None)
        if desc_tgt is None:
            return detected_bboxes

        for class_id, (kp, desc) in self.template_features.items():
            matches = self.flann.knnMatch(desc, desc_tgt, k=2)
            good_matches = [m for m, n in matches if m.distance < self.MATCH_THRESHOLD * n.distance]

            if len(good_matches) > 4:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    h, w = self.templates[class_id].shape
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    x_min, y_min = np.min(transformed_corners, axis=0)[0]
                    x_max, y_max = np.max(transformed_corners, axis=0)[0]
                    
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    
                    detected_bboxes[class_id] = [bbox]
                    logger.info(f"Detected {class_id} at {bbox}")

        return detected_bboxes