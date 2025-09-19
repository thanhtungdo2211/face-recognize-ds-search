import io
import base64
import cv2
import os
import sys
import numpy as np
import torch
import time
from typing import List

# Import the required face processing models
from backend.ai_service2.base import SCRFD, ARCFACE, align_face
from backend.ai_service.face_detection.yolov5_face.detector import Yolov5Face
from uuid import uuid4

from pydantic import BaseModel
from search_module.search import upsert, search

MODEL_YOLO = 1
MODEL_SCRFD = 2
MODEL_DETECT_FACE = MODEL_SCRFD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_IMGS = "/home/mq/disk2T/tungdt/spa/data/img"
PATH_IMGS_FACE = "/home/mq/disk2T/tungdt/spa/data/face"
PATH_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PATH_MODEL_DETECTOR = ""
if MODEL_DETECT_FACE == MODEL_SCRFD:
    PATH_MODEL_DETECTOR = os.path.join(PATH_MODEL, "backend/ai_service2/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx")
elif MODEL_DETECT_FACE == MODEL_YOLO:
    PATH_MODEL_DETECTOR = os.path.join(PATH_MODEL, "backend/ai_service2/yoloface/yolov5_face/weights/yolov5m-face.pt")
PATH_MODEL_RECOGNIZER = os.path.join(PATH_MODEL, "backend/ai_service2/arcface/arcface_r100.onnx")

# Load models
detector = None
if MODEL_DETECT_FACE == MODEL_SCRFD:
    detector = SCRFD(PATH_MODEL_DETECTOR)
elif MODEL_DETECT_FACE == MODEL_YOLO:
    detector = Yolov5Face(model_file=PATH_MODEL_DETECTOR)
recognizer = ARCFACE(PATH_MODEL_RECOGNIZER)

def get_face_feature(img_path):
    """Extract face feature from image path"""
    try:
        image = cv2.imread(img_path)
        if image is None:
            return None

        bboxes = []
        landmarks = []
        if MODEL_DETECT_FACE == MODEL_SCRFD:
            bboxes, landmarks = detector(image, 0.5, input_size=(640, 640))
        elif MODEL_DETECT_FACE == MODEL_YOLO:
            bboxes, landmarks = detector.detect(image=image)
        
        if len(bboxes) == 0:
            return None

        x1, y1, x2, y2 = bboxes[0][:4]
        lm = landmarks[0]
        face_crop = align_face(image.copy(), [x1, y1, x2, y2], lm)
        
        # Extract feature
        feature = recognizer(face_crop.copy())
        return feature
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def add_noise_to_feature(feature, noise_level=0.001):
    """Add small random noise to feature vector to create variations"""
    noise = np.random.normal(0, noise_level, feature.shape)
    noisy_feature = feature + noise
    # Re-normalize to maintain unit vector (important for ArcFace)
    norm = np.linalg.norm(noisy_feature)
    return noisy_feature / norm if norm > 0 else noisy_feature

def generate_feature_variations(base_feature, num_variations=500, noise_levels=None):
    """Generate multiple variations of a base feature"""
    if noise_levels is None:
        noise_levels = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    variations = []
    variations.append(base_feature)  # Include original
    
    for i in range(num_variations - 1):
        # Use different noise levels
        noise_level = noise_levels[i % len(noise_levels)]
        variation = add_noise_to_feature(base_feature, noise_level)
        variations.append(variation)
    
    return variations

def bulk_upsert_test_images(num_vectors_per_image=500, batch_size=50):
    """
    Upsert 500 vectors for test1.jpg and 500 vectors for test2.jpg
    """
    test_images = {
        "test1.jpg": {
            "path": "/home/mq/disk2T/tungdt/spa/data/test1.jpg",
            "user_id": "user_test1",
            "camera_id": "camera_001"
        },
        "test2.jpg": {
            "path": "/home/mq/disk2T/tungdt/spa/data/test2.jpg", 
            "user_id": "user_test2",
            "camera_id": "camera_002"
        }
    }
    
    total_start_time = time.time()
    overall_results = []
    
    print(f"Starting bulk upsert: {num_vectors_per_image} vectors per image")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    for img_name, img_info in test_images.items():
        print(f"\nProcessing {img_name}...")
        
        # Extract base feature
        base_feature = get_face_feature(img_info["path"])
        if base_feature is None:
            print(f"Failed to extract feature from {img_name}")
            continue
        
        print(f"Base feature shape: {base_feature.shape}")
        print(f"Base feature norm: {np.linalg.norm(base_feature):.6f}")
        
        # Generate variations
        print(f"Generating {num_vectors_per_image} feature variations...")
        variations = generate_feature_variations(base_feature, num_vectors_per_image)
        
        # Upsert in batches
        total_upserted = 0
        batch_times = []
        
        for batch_start in range(0, len(variations), batch_size):
            batch_end = min(batch_start + batch_size, len(variations))
            batch_features = variations[batch_start:batch_end]
            
            batch_start_time = time.time()
            
            # Upsert batch
            result = upsert(
                user_id=img_info["user_id"],
                features=batch_features,
                camera_id=img_info["camera_id"]
            )
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if result["status"] == "success":
                total_upserted += result["upserted_count"]
                print(f"  Batch {batch_start//batch_size + 1}: "
                      f"Upserted {result['upserted_count']} vectors in {batch_time:.3f}s")
            else:
                print(f"  Batch {batch_start//batch_size + 1}: ERROR - {result.get('error', 'Unknown error')}")
        
        # Summary for this image
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        total_time = sum(batch_times)
        
        result_summary = {
            "image": img_name,
            "user_id": img_info["user_id"],
            "camera_id": img_info["camera_id"],
            "requested_vectors": num_vectors_per_image,
            "upserted_vectors": total_upserted,
            "total_batches": len(batch_times),
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "vectors_per_second": total_upserted / total_time if total_time > 0 else 0
        }
        
        overall_results.append(result_summary)
        
        print(f"\n{img_name} Summary:")
        print(f"  Total upserted: {total_upserted}/{num_vectors_per_image}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average batch time: {avg_batch_time:.3f}s")
        print(f"  Vectors per second: {result_summary['vectors_per_second']:.1f}")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    total_vectors = sum(r["upserted_vectors"] for r in overall_results)
    
    for result in overall_results:
        print(f"{result['image']:12} | {result['user_id']:12} | "
              f"{result['upserted_vectors']:4d} vectors | "
              f"{result['total_time']:6.2f}s | "
              f"{result['vectors_per_second']:6.1f} v/s")
    
    print(f"\nTotal vectors upserted: {total_vectors}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Overall rate: {total_vectors/total_time:.1f} vectors/second")
    
    return overall_results



if __name__ == "__main__":
    # Configuration
    NUM_VECTORS_PER_IMAGE = 500
    BATCH_SIZE = 50  # Adjust based on your system performance
    
    print("🚀 Starting bulk upsert test...")
    print(f"Target: {NUM_VECTORS_PER_IMAGE} vectors per image")
    print(f"Images: test1.jpg, test2.jpg")
    print(f"Total vectors to upsert: {NUM_VECTORS_PER_IMAGE * 2}")
    
    # Run bulk upsert
    results = bulk_upsert_test_images(
        num_vectors_per_image=NUM_VECTORS_PER_IMAGE,
        batch_size=BATCH_SIZE
    )