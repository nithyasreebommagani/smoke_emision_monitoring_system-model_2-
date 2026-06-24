import sqlite3

DB_NAME = "events.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id INTEGER,
        plate TEXT,
        timestamp TEXT,
        smoke_count INTEGER,
        image_path TEXT,
        video_path TEXT,
        status TEXT,
        synced_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_violation(
    vehicle_id,
    plate,
    timestamp,
    smoke_count,
    image_path,
    video_path
):
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    INSERT INTO violations (
        vehicle_id,
        plate,
        timestamp,
        smoke_count,
        image_path,
        video_path,
        status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        vehicle_id,
        plate,
        timestamp,
        smoke_count,
        image_path,
        video_path,
        "PENDING"
    ))

    conn.commit()
    conn.close()