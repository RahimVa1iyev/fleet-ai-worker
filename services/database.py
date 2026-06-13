import os
import psycopg2
import json
from datetime import datetime, timezone


def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def update_event_processing(event_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE "trip_events" SET "aiStatus" = 'PROCESSING', "updatedAt" = NOW() WHERE id = %s""",
                (event_id,)
            )
        conn.commit()
    finally:
        conn.close()


def update_event_completed(event_id: str, ai_result: dict, severity: str, score: int):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE "trip_events"
                SET "aiStatus"   = 'COMPLETED',
                    "aiResult"   = %s,
                    "severity"   = %s,
                    "score"      = %s,
                    "analyzedAt" = %s,
                    "updatedAt"  = NOW()
                WHERE id = %s
                """,
                (json.dumps(ai_result), severity, score, datetime.now(timezone.utc), event_id)
            )
        conn.commit()
    finally:
        conn.close()


def update_event_failed(event_id: str, reason: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE "trip_events" SET "aiStatus" = 'FAILED', "errorReason" = %s, "updatedAt" = NOW() WHERE id = %s""",
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
                FROM "trip_events" te
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