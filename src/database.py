"""Database management module for storing patient biometrics and prediction history."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import DATA_DIR

DB_PATH = DATA_DIR / "cardiopredict.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize patient records schema if not exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            predicted_target INTEGER,
            diagnosis TEXT,
            probability REAL,
            risk_tier TEXT,
            clinical_flags TEXT,
            clinician_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_patient_record(
    patient_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    notes: Optional[str] = "",
) -> int:
    """Save an evaluated patient case and its risk prediction into the database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    flags_json = json.dumps(prediction_result.get("clinical_flags", []))

    cursor.execute(
        """
        INSERT INTO patient_history (
            patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
            predicted_target, diagnosis, probability, risk_tier,
            clinical_flags, clinician_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id or f"PT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            patient_name or "Anonymous Patient",
            int(patient_data["age"]),
            int(patient_data["sex"]),
            int(patient_data["cp"]),
            int(patient_data["trestbps"]),
            int(patient_data["chol"]),
            int(patient_data["fbs"]),
            int(patient_data["restecg"]),
            int(patient_data["thalach"]),
            int(patient_data["exang"]),
            float(patient_data["oldpeak"]),
            int(patient_data["slope"]),
            int(patient_data["ca"]),
            int(patient_data["thal"]),
            int(prediction_result["prediction"]),
            prediction_result["diagnosis"],
            float(prediction_result["probability"]),
            prediction_result["risk_tier"],
            flags_json,
            notes or "",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def save_batch_records(batch_df: pd.DataFrame) -> int:
    """Save a batch cohort prediction DataFrame into the database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    inserted_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in batch_df.iterrows():
        pid = row.get("patient_id", f"COHORT-PT-{idx+1:03d}")
        pname = row.get("patient_name", f"Cohort Patient #{idx+1}")

        cursor.execute(
            """
            INSERT INTO patient_history (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                predicted_target, diagnosis, probability, risk_tier,
                clinical_flags, clinician_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(pid),
                str(pname),
                int(row["age"]),
                int(row["sex"]),
                int(row["cp"]),
                int(row["trestbps"]),
                int(row["chol"]),
                int(row["fbs"]),
                int(row["restecg"]),
                int(row["thalach"]),
                int(row["exang"]),
                float(row["oldpeak"]),
                int(row["slope"]),
                int(row["ca"]),
                int(row["thal"]),
                int(row.get("predicted_target", 0)),
                row.get("predicted_diagnosis", "Evaluated"),
                float(row.get("disease_probability", 0.0)),
                row.get("risk_tier", "Unassigned"),
                "[]",
                "Batch Ingestion",
                now_str,
            ),
        )
        inserted_count += 1

    conn.commit()
    conn.close()
    return inserted_count


def get_all_records(limit: int = 200) -> pd.DataFrame:
    """Retrieve historical patient records as a Pandas DataFrame."""
    init_db()
    conn = get_db_connection()
    query = f"SELECT * FROM patient_history ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def delete_record(record_id: int) -> bool:
    """Delete a record by its database ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_history WHERE id = ?", (record_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def get_database_stats() -> Dict[str, Any]:
    """Calculate summary statistics from stored patient records."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM patient_history")
    total_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patient_history WHERE risk_tier = 'High Risk'")
    high_risk_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patient_history WHERE risk_tier = 'Moderate Risk'")
    mod_risk_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patient_history WHERE risk_tier = 'Low Risk'")
    low_risk_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(probability) FROM patient_history")
    avg_prob = cursor.fetchone()[0] or 0.0

    conn.close()
    return {
        "total_records": total_records,
        "high_risk_count": high_risk_count,
        "moderate_risk_count": mod_risk_count,
        "low_risk_count": low_risk_count,
        "avg_probability": round(float(avg_prob), 4),
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
