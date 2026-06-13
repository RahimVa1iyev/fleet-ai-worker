import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

# YOLO class adları (COCO dataset)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

_session = None

def get_session():
    global _session
    if _session is None:
        model_path = os.environ.get("YOLO_MODEL_PATH", "/app/models/yolov8n.onnx")
        _session = ort.InferenceSession(model_path)
        print(f"[YoloService] Model yükləndi: {model_path}")
    return _session

def preprocess(image_bytes: bytes) -> np.ndarray:
    """Frame-i YOLO üçün hazırla: resize + normalize"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((640, 640))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    arr = np.expand_dims(arr, axis=0)     # batch dimension əlavə et
    return arr

def run_inference(image_bytes: bytes) -> list[dict]:
    """
    Frame-i analiz et, aşkarlamaları qaytar.
    Hər element: {object, confidence}
    """
    session = get_session()
    input_tensor = preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    detections = []
    predictions = outputs[0][0]  # shape: (num_detections, 85)

    for pred in predictions.T:   # YOLOv8 output formatı
        confidence = float(pred[4])
        if confidence < 0.25:    # minimum confidence threshold
            continue

        class_scores = pred[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id]) * confidence

        if class_conf < 0.25:
            continue

        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
        detections.append({
            "object": class_name,
            "confidence": round(class_conf, 2)
        })

    # Eyni obyektləri birləşdir (unikal saxla)
    seen = {}
    for d in detections:
        key = d["object"]
        if key not in seen or d["confidence"] > seen[key]["confidence"]:
            seen[key] = d

    return list(seen.values())