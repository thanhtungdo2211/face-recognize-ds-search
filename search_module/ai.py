import os

import cv2
import torch

# Import the required face processing models
from backend.ai_service2.base import SCRFD, ARCFACE, align_face
from backend.ai_service.face_detection.yolov5_face.detector import Yolov5Face
from typing import List
from uuid import uuid4

from pydantic import BaseModel
# from backend.core.g_config import GlobalConfig

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
    id_user="user_1",
    paths=["data/test1.jpg"]
)
result = {}
avatar_path = ""
result_imgs = []

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

    # Save cropped face
    face_filename = f"face_{uuid4().hex[:8]}.jpg"
    face_path = os.path.join(PATH_IMGS_FACE_ID, face_filename)
    cv2.imwrite(face_path, face_crop)

    if len(avatar_path) <= 0:
        avatar_path = face_path
    
    # Extract feature
    feature = recognizer(face_crop.copy())
    print(type(feature))
    result_imgs.append({
        "input_path": img_path,
        "face_crop_path": face_path,
        "feature": feature.tolist()
    })

if len(avatar_path) > 0:
    result["avatar_path"] = avatar_path
result["data"] = result_imgs
result["num_success"] = len(bboxes)