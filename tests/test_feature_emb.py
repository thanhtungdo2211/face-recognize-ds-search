import io
import base64
import os

import cv2
import numpy as np
import torch

# Import the required face processing models
from backend.ai_service2.base import SCRFD, ARCFACE, align_face
from backend.ai_service.face_detection.yolov5_face.detector import Yolov5Face


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
    image = cv2.imread(img_path)

    bboxes = []
    landmarks = []
    if MODEL_DETECT_FACE == MODEL_SCRFD:
        bboxes, landmarks = detector(image, 0.5, input_size=(640, 640))
    elif MODEL_DETECT_FACE == MODEL_YOLO:
        bboxes, landmarks = detector.detect(image=image)

    x1, y1, x2, y2 = bboxes[0][:4]
    lm = landmarks[0]
    face_crop = align_face(image.copy(), [x1, y1, x2, y2], lm)
    
    # Extract feature
    feature = recognizer(face_crop.copy())
    return feature

feat1 = get_face_feature("data/test1.jpg")
feat2 = get_face_feature("data/test2.jpg")

euclid_distance = np.linalg.norm(feat1 - feat2)
cosine_distance = 1 - (np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))

print(euclid_distance, cosine_distance)