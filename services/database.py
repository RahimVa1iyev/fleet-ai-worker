import os
import psycopg2
import psycopg2.extras
import json
from datetime import datetime, timezone

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def update_event_processing(event_id: str):
    """Status-u PROCESSING olaraq yenilə"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trip_events
                SET ai_status = 'PROCESSING',
                    updated_at = NOW()
                WHERE id = %s
                """,
                (event_id,)
            )
        conn.commit()
    finally:
        conn.close()

def update_event_completed(event_id: str, ai_result: dict, severity: str, score: int):
    """AI nəticəsini DB-yə yaz"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trip_events
                SET ai_status    = 'COMPLETED',
                    ai_result    = %s,
                    severity     = %s,
                    score        = %s,
                    analyzed_at  = %s,
                    updated_at   = NOW()
                WHERE id = %s
                """,
                (
                    json.dumps(ai_result),
                    severity,
                    score,
                    datetime.now(timezone.utc),
                    event_id,
                )
            )
        conn.commit()
    finally:
        conn.close()

def update_event_failed(event_id: str, reason: str):
    """Xəta halında FAILED yaz"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trip_events
                SET ai_status    = 'FAILED',
                    error_reason = %s,
                    updated_at   = NOW()
                WHERE id = %s
                """,
                (reason, event_id)
            )
        conn.commit()
    finally:
        conn.close()

def get_driver_fcm_token(event_id: str) -> str | None:
    """Hadisəyə aid sürücünün FCM tokenini gətir"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.fcm_token
                FROM trip_events te
                JOIN trips t ON t.id = te.trip_id
                JOIN drivers d ON d.id = t.driver_id
                WHERE te.id = %s
                """,
                (event_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()