import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

import firebase_admin
from firebase_admin import credentials, messaging

# Firebase başlat
_firebase_app = None

def get_firebase_app():
    global _firebase_app
    if _firebase_app is None:
        cred_path = os.environ.get("FCM_SERVICE_ACCOUNT_PATH", "/secrets/firebase-adminsdk.json")
        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app

def send_fcm_notification(fcm_token: str, event_id: str, driver_id: str, severity: str, event_type: str):
    try:
        get_firebase_app()
        if severity == "HIGH":
            message = messaging.Message(
                notification=messaging.Notification(
                    title="Təhlükəli hadisə",
                    body=f"{event_type} — HIGH risk aşkarlandı",
                ),
                data={"eventId": event_id, "severity": severity, "type": "analysis_done"},
                token=fcm_token,
            )
        else:
            message = messaging.Message(
                data={"eventId": event_id, "severity": severity, "type": "analysis_done"},
                token=fcm_token,
            )
        response = messaging.send(message)
        print(f"[Worker] FCM göndərildi — eventId: {event_id}, driverId: {driver_id}, severity: {severity}, messageId: {response}")
    except Exception as e:
        error_msg = str(e)
        print(f"[Worker] FCM xətası — eventId: {event_id}, driverId: {driver_id}, xəta: {error_msg}")
        if "not a valid FCM registration token" in error_msg or "not registered" in error_msg.lower() or "invalid" in error_msg.lower():
            print(f"[Worker] Diqqət — token etibarsızdır, DB-də təmizlənməlidir: driverId: {driver_id}")
            try:
                clear_driver_fcm_token(driver_id)
                print(f"[Worker] Token DB-də təmizləndi: driverId: {driver_id}")
            except Exception as clear_err:
                print(f"[Worker] Token təmizlənərkən xəta: {str(clear_err)}")

from bullmq import Worker
from services.database import (
    update_event_processing,
    update_event_completed,
    update_event_failed,
    get_driver_fcm_token,
    clear_driver_fcm_token,
    get_event_status,
)
from services.storage import download_frame
from services.yolo import run_inference
from services.enrichment import get_weather, get_road_info
from services.narrative import generate_narrative

QUEUE_NAME = "ai-analysis"

def calculate_risk(
    detections: list, 
    accel_data: dict, 
    event_type: str,
    current_speed_kmh: float = None,
    speed_limit_kmh: float = None,
) -> tuple[int, str]:
    """
    Evristik risk xalı hesabla.
    Qaytarır: (score, severity)
    """
    score = 0

    # 1. Hadisə növünə görə baza xal
    base_scores = {
        "COLLISION":            70,
        "HARSH_BRAKING":        30,
        "SHARP_TURN":           20,
        "HARSH_ACCELERATION":   15,
    }

    if event_type == "SPEEDING":
        # SPEEDING üçün dinamik hesablama — sürət limitinin neçə 
        # faiz aşıldığına əsaslanır, sabit xal əvəzinə
        if current_speed_kmh and speed_limit_kmh and speed_limit_kmh > 0:
            overspeed_pct = ((current_speed_kmh - speed_limit_kmh) / speed_limit_kmh) * 100
            overspeed_pct = max(0, overspeed_pct)
            speeding_score = 10 + (overspeed_pct * 0.6)
            score += min(speeding_score, 70)
        else:
            # Sürət/limit məlumatı yoxdursa, fallback
            score += 20
    else:
        score += base_scores.get(event_type, 20)

    # 2. Aşkarlanan obyektlərə görə xal (tavanla məhdudlaşdırılıb)
    high_risk_objects = {"person", "bicycle", "motorcycle"}
    medium_risk_objects = {"car", "truck", "bus", "traffic light", "stop sign"}

    object_score = 0
    for d in detections:
        obj = d["object"]
        conf = d["confidence"]
        if obj in high_risk_objects:
            object_score += int(30 * conf)
        elif obj in medium_risk_objects:
            object_score += int(15 * conf)
    score += min(object_score, 40)

    # 3. G-qüvvəsinə görə əlavə xal (genişləndirilmiş şkala)
    g_force = (accel_data.get("gForce") if accel_data else None) or 0
    if g_force >= 2.5:
        score += 35
    elif g_force >= 1.5:
        score += 25
    elif g_force >= 1.0:
        score += 20
    elif g_force >= 0.7:
        score += 10

    score = max(0, min(score, 100))  # 0-100 aralığında clamp

    # Severity təsnifatı (texniki tapşırıq Bölmə M7, dəyişməz)
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
    driver_id  = data.get("driverId", "unknown")
    frame_key  = data.get("frameR2Key") or data.get("frameUrl", "").replace("r2://fleet-events/", "")
    accel_data = data.get("accelData", {})
    event_type = data.get("eventType", "HARSH_BRAKING")
    gps        = data.get("gps") or {}
    lat        = gps.get("lat")
    lng        = gps.get("lng")

    print(f"[Worker] İş başladı — eventId: {event_id}, type: {event_type}")

    try:
        # İdempotentlik: event artıq COMPLETED-dirsə, təkrar emal etmə
        current_status = get_event_status(event_id)
        if current_status == "COMPLETED":
            print(f"[Worker] Event artıq COMPLETED — təkrar emal atlanır, eventId: {event_id}")
            return

        # 1. Status PROCESSING
        update_event_processing(event_id)

        if not frame_key:
            raise ValueError(f"frameR2Key tapılmadı — eventId: {event_id}, job data: {data}")

        # 2. R2-dən frame endir
        print(f"[Worker] Frame endiriliyr: {frame_key}")
        image_bytes = download_frame(frame_key)

        # 3. YOLO inference
        print(f"[Worker] YOLO analiz başlayır...")
        detections = run_inference(image_bytes)
        print(f"[Worker] Aşkarlamalar: {detections}")

        # 4. Enrichment (Weather & Road)
        weather_info = None
        road_info = None
        if lat is not None and lng is not None:
            try:
                weather_info = get_weather(float(lat), float(lng))
                road_info = get_road_info(float(lat), float(lng))
            except Exception as e:
                print(f"[Worker] Enrichment xətası (uduldu): {e}")
        else:
            print(f"[Worker] GPS tapılmadı — eventId: {event_id}, enrichment atlanır")

        # 5. Risk hesabla
        score, severity = calculate_risk(
            detections, accel_data, event_type,
            current_speed_kmh=gps.get("speed"),
            speed_limit_kmh=road_info.get("speedLimitKmh") if road_info else None,
        )
        print(f"[Worker] Risk xalı: {score}, Severity: {severity}")

        # 6. Xülasə yarat
        summary = build_summary(detections, event_type, severity, score)

        # 5.2. Narrative Generation
        narrative_result = None
        try:
            facts = {
                "eventType": event_type,
                "speedKmhBefore": data.get("speedKmhBefore"),
                "speedKmhAfter": data.get("speedKmhAfter"),
                "gForce": accel_data.get("gForce") if accel_data else None,
                "detections": detections,
                "weather": weather_info,
                "roadType": road_info.get("roadType") if road_info else None,
                "speedLimitKmh": road_info.get("speedLimitKmh") if road_info else None,
                "timeOfDay": data.get("timeOfDay")
            }
            narrative_result = generate_narrative(facts)
        except Exception as e:
            print(f"[Worker] Narrative xətası (uduldu): {e}")

        ai_result = {
            "detections": detections,
            "riskScore":  score,
            "summary":    summary,
            "weather":    weather_info,
            "roadInfo":   road_info,
            "narrative":  narrative_result
        }

        # 6. DB-yə yaz
        update_event_completed(event_id, ai_result, severity, score, weather_info, road_info, narrative_result)
        print(f"[Worker] DB yeniləndi — COMPLETED")

        # 7. FCM push
        fcm_token = get_driver_fcm_token(event_id)
        if fcm_token:
            send_fcm_notification(fcm_token, event_id, driver_id, severity, event_type)
        else:
            print(f"[Worker] FCM token tapilmadi — eventId: {event_id}, driverId: {driver_id}")

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
            "stalledInterval": 30000,
            "maxStalledCount": 2,
        }
    )

    async def on_failed(job, err, prev_state):
        attempts = job.attempts_made if hasattr(job, 'attempts_made') else 0
        print(f"[Worker] İş uğursuz — eventId: {job.data.get('eventId')}, cəhd: {attempts}, xəta: {err}")
        if attempts >= 3:
            event_id = job.data.get("eventId")
            if event_id:
                update_event_failed(event_id, str(err))
                print(f"[Worker] FAILED yazildi — eventId: {event_id}")

    worker.on("failed", on_failed)

    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())