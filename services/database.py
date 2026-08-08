import os
import psycopg2
import json
from datetime import datetime, timezone


def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def get_event_status(event_id: str) -> str | None:
    """Event-in hazırkı aiStatus-unu DB-dən oxu"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT "aiStatus" FROM trip_events WHERE id = %s', (event_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def update_event_processing(event_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE trip_events SET "aiStatus" = 'PROCESSING' WHERE id = %s""",
                (event_id,)
            )
        conn.commit()
    finally:
        conn.close()


def update_event_completed(event_id: str, ai_result: dict, severity: str, score: int, weather_info: dict = None, road_info: dict = None, narrative_result: dict = None):
    conn = get_connection()
    try:
        weather_data_json = json.dumps(weather_info) if weather_info else None
        road_type = road_info.get("roadType") if road_info else None
        speed_limit = road_info.get("speedLimitKmh") if road_info else None
        narrative_text = narrative_result.get("text") if narrative_result else None
        narrative_gen_by = narrative_result.get("generatedBy") if narrative_result else None

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trip_events
                SET "aiStatus"   = 'COMPLETED',
                    "aiResult"   = %s,
                    severity     = %s,
                    score        = %s,
                    "analyzedAt" = %s,
                    "weatherData" = %s,
                    "roadType"    = %s,
                    "speedLimitKmh" = %s,
                    narrative     = %s,
                    "narrativeGeneratedBy" = %s
                WHERE id = %s
                """,
                (
                    json.dumps(ai_result), severity, score, datetime.now(timezone.utc),
                    weather_data_json, road_type, speed_limit, narrative_text, narrative_gen_by,
                    event_id
                )
            )
        conn.commit()
    finally:
        conn.close()


def update_event_failed(event_id: str, reason: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE trip_events SET "aiStatus" = 'FAILED', "errorReason" = %s WHERE id = %s""",
                (reason, event_id)
            )
        conn.commit()
    finally:
        conn.close()


def get_driver_fcm_token(event_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d."fcmToken"
                FROM trip_events te
                JOIN trips t ON t.id = te."tripId"
                JOIN drivers d ON d.id = t."driverId"
                WHERE te.id = %s
                """,
                (event_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def clear_driver_fcm_token(driver_id: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE drivers
                SET "fcmToken" = NULL
                WHERE id = %s
                """,
                (driver_id,)
            )
            conn.commit()
    finally:
        conn.close()