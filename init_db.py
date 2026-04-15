import csv
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "").strip()
if not DB_URL or DB_URL.lower().startswith("sqlite"):
    raise RuntimeError(
        "Set DATABASE_URL in .env to MySQL, e.g. "
        "mysql+pymysql://root:@127.0.0.1:3310/smartphone"
    )

USERS_CSV = Path("v2.csv")
DATA_CSV = Path("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _chunked(items: list[list[str]], size: int = 1000) -> Iterable[list[list[str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def init_database() -> None:
    if not USERS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {USERS_CSV}")
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {DATA_CSV}")

    users_cols, users_rows = _read_csv_rows(USERS_CSV)
    data_cols, data_rows = _read_csv_rows(DATA_CSV)

    if not users_cols:
        raise ValueError(f"No header found in {USERS_CSV}")
    if not data_cols:
        raise ValueError(f"No header found in {DATA_CSV}")

    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data"))
        conn.execute(text("DROP TABLE IF EXISTS users"))

        conn.execute(
            text(
                """
                CREATE TABLE users (
                    transaction_id VARCHAR(32) NOT NULL PRIMARY KEY,
                    user_id VARCHAR(32) NOT NULL,
                    age INT NULL,
                    gender VARCHAR(32) NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE data (
                    transaction_id VARCHAR(32) NOT NULL PRIMARY KEY,
                    daily_screen_time_hours DOUBLE NULL,
                    social_media_hours DOUBLE NULL,
                    gaming_hours DOUBLE NULL,
                    work_study_hours DOUBLE NULL,
                    sleep_hours DOUBLE NULL,
                    notifications_per_day INT NULL,
                    app_opens_per_day INT NULL,
                    weekend_screen_time DOUBLE NULL,
                    stress_level VARCHAR(32) NULL,
                    academic_work_impact VARCHAR(32) NULL,
                    addiction_level VARCHAR(32) NULL,
                    addicted_label TINYINT NULL,
                    CONSTRAINT fk_data_users
                        FOREIGN KEY (transaction_id) REFERENCES users (transaction_id)
                        ON DELETE CASCADE ON UPDATE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        )

        users_insert_sql = text(
            """
            INSERT INTO users (transaction_id, user_id, age, gender)
            VALUES (:transaction_id, :user_id, :age, :gender)
            """
        )
        for batch in _chunked(users_rows):
            payload = [
                {
                    "transaction_id": row[0],
                    "user_id": row[1],
                    "age": int(row[2]) if row[2] else None,
                    "gender": row[3],
                }
                for row in batch
            ]
            conn.execute(users_insert_sql, payload)

        data_insert_sql = text(
            """
            INSERT INTO data (
                transaction_id,
                daily_screen_time_hours,
                social_media_hours,
                gaming_hours,
                work_study_hours,
                sleep_hours,
                notifications_per_day,
                app_opens_per_day,
                weekend_screen_time,
                stress_level,
                academic_work_impact,
                addiction_level,
                addicted_label
            )
            VALUES (
                :transaction_id,
                :daily_screen_time_hours,
                :social_media_hours,
                :gaming_hours,
                :work_study_hours,
                :sleep_hours,
                :notifications_per_day,
                :app_opens_per_day,
                :weekend_screen_time,
                :stress_level,
                :academic_work_impact,
                :addiction_level,
                :addicted_label
            )
            """
        )
        for batch in _chunked(data_rows):
            payload = [
                {
                    "transaction_id": row[0],
                    "daily_screen_time_hours": float(row[1]) if row[1] else None,
                    "social_media_hours": float(row[2]) if row[2] else None,
                    "gaming_hours": float(row[3]) if row[3] else None,
                    "work_study_hours": float(row[4]) if row[4] else None,
                    "sleep_hours": float(row[5]) if row[5] else None,
                    "notifications_per_day": int(row[6]) if row[6] else None,
                    "app_opens_per_day": int(row[7]) if row[7] else None,
                    "weekend_screen_time": float(row[8]) if row[8] else None,
                    "stress_level": row[9],
                    "academic_work_impact": row[10],
                    "addiction_level": row[11],
                    "addicted_label": int(row[12]) if row[12] else None,
                }
                for row in batch
            ]
            conn.execute(data_insert_sql, payload)


if __name__ == "__main__":
    init_database()
    print("Database initialized with users and data tables (MySQL per DATABASE_URL).")
