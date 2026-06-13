import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from bullmq import Worker
from services.database import (
    update_event_processing,
    update_event_completed,
    update_event_failed,
    get_driver_fcm_token,
)
from services.storage import download_frame
from services.yolo import run_inference

QUEUE_NAME = "ai-analysis"

def calculate_risk(detections: list, accel_data: dict, event_type: str) -> tuple[int, str]:
    """
    Evristik risk xalı hesabla.
    Qaytarır: (score, severity)
    """
    score = 0

    # 1. Hadisə növünə görə baza xal
    base_scores = {
        "COLLISION":     60,
        "HARSH_BRAKING": 30,
        "SHARP_TURN":    20,
    }
    score += base_scores.get(event_type, 20)

    # 2. Aşkarlanan obyektlərə görə xal
    high_risk_objects = {"person", "bicycle", "motorcycle", "child"}
    medium_risk_objects = {"car", "truck", "bus", "traffic light", "stop sign"}

    for d in detections:
        obj = d["object"]
        conf = d["confidence"]
        if obj in high_risk_objects:
            score += int(30 * conf)
        elif obj in medium_risk_objects:
            score += int(15 * conf)

    # 3. G-qüvvəsinə görə əlavə xal
    g_force = accel_data.get("gForce", 0) if accel_data else 0
    if g_force >= 1.0:
        score += 20
    elif g_force >= 0.7:
        score += 10

    score = min(score, 100)  # maksimum 100

    # Severity təsnifatı (texniki tapşırıq Bölmə M7)
    if score >= 70:
        severity = "HIGH"
    elif score >= 30:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return score, severity

def build_summary(detections: list, event_type: str, severity: str, score: int) -> str:
    """İnsan dilində xülasə yarat"""
    event_names = {
        "HARSH_BRAKING": "Sert əyləc",
        "COLLISION":     "Toqquşma",
        "SHARP_TURN":    "Kəskin dönüş",
    }
    event_name = event_names.get(event_type, event_type)

    if detections:
        obj_list = ", ".join([d["object"] for d in detections[:3]])
        return f"{event_name} aşkarlandı. Çərçivədə: {obj_list}. Risk xalı: {score}/100. Şiddət: {severity}."
    else:
        return f"{event_name} aşkarlandı. Çərçivədə obyekt tapılmadı. Risk xalı: {score}/100. Şiddət: {severity}."

async def process_job(job, job_token):
    """BullMQ-dan gələn hər işi emal et"""
    data = job.data
    event_id   = data.get("eventId")
    frame_key  = data.get("frameR2Key") or data.get("frameUrl", "").replace("r2://fleet-events/", "")
    accel_data = data.get("accelData", {})
    event_type = data.get("eventType", "HARSH_BRAKING")

    print(f"[Worker] İş başladı — eventId: {event_id}, type: {event_type}")

    try:
        # 1. Status PROCESSING
        update_event_processing(event_id)

        # 2. R2-dən frame endir
        print(f"[Worker] Frame endiriliyr: {frame_key}")
        image_bytes = download_frame(frame_key)

        # 3. YOLO inference
        print(f"[Worker] YOLO analiz başlayır...")
        detections = run_inference(image_bytes)
        print(f"[Worker] Aşkarlamalar: {detections}")

        # 4. Risk hesabla
        score, severity = calculate_risk(detections, accel_data, event_type)
        print(f"[Worker] Risk xalı: {score}, Severity: {severity}")

        # 5. Xülasə yarat
        summary = build_summary(detections, event_type, severity, score)

        ai_result = {
            "detections": detections,
            "riskScore":  score,
            "summary":    summary,
        }

        # 6. DB-yə yaz
        update_event_completed(event_id, ai_result, severity, score)
        print(f"[Worker] DB yeniləndi — COMPLETED")

    except Exception as e:
        error_msg = str(e)
        print(f"[Worker] XƏTA: {error_msg}")
        update_event_failed(event_id, error_msg)
        raise  # BullMQ retry üçün xətanı yenidən at

async def main():
    print(f"[Worker] Python AI Worker başladı — queue: {QUEUE_NAME}")

    worker = Worker(
        QUEUE_NAME,
        process_job,
        {
            "connection": {
                "host": os.environ.get("REDIS_HOST", "redis"),
                "port": int(os.environ.get("REDIS_PORT", 6379)),
            },
            "concurrency": int(os.environ.get("WORKER_CONCURRENCY", 2)),
        }
    )

    # Worker-i açıq saxla
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())