import io
import base64
import cv2
import os
import sys
import numpy as np
import torch

# Import the required face processing models
from backend.ai_service2.base import SCRFD, ARCFACE, align_face
from backend.ai_service.face_detection.yolov5_face.detector import Yolov5Face
from typing import List
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

class ImagePaths(BaseModel):
    id_user: str 
    paths: List[str]

# Provide a proper ImagePaths instance (replace id_user and paths as needed)
image_paths = ImagePaths(
    id_user="user_3",
    paths=["data/2025-09-17T074325_None.jpg"]
)
result = {}
avatar_path = ""
result_imgs = []
features = []

id_user = image_paths.id_user

os.makedirs(PATH_IMGS_FACE, exist_ok=True)
PATH_IMGS_FACE_ID = os.path.join(PATH_IMGS_FACE, id_user)
os.makedirs(PATH_IMGS_FACE_ID, exist_ok=True)

for img_path in image_paths.paths:
    if not os.path.isfile(img_path):
        result_imgs.append({
            "path": img_path,
            "error": "File not found"
        })
        continue

    image = cv2.imread(img_path)
    if image is None:
        result_imgs.append({
            "path": img_path,
            "error": "Unable to read image"
        })
        continue

    bboxes = []
    landmarks = []
    if MODEL_DETECT_FACE == MODEL_SCRFD:
        bboxes, landmarks = detector(image, 0.5, input_size=(640, 640))
    elif MODEL_DETECT_FACE == MODEL_YOLO:
        bboxes, landmarks = detector.detect(image=image)
    if len(bboxes) == 0:
        result_imgs.append({
            "path": img_path,
            "error": "No face detected"
        })
        continue

    x1, y1, x2, y2 = bboxes[0][:4]
    lm = landmarks[0]
    face_crop = align_face(image.copy(), [x1, y1, x2, y2], lm)
    
    # Extract feature
    feature = recognizer(face_crop.copy())

    search_res = search(
        query_vector=feature
    )
    print(search_res)
    
    if not search_res["result"]:
        features.append(feature)

print(f"Upsert {len(features)} features")
res = upsert(user_id=id_user,
        features=features,
        camera_id=1)

print(res)