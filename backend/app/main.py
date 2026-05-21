from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "har.db"

ALLOWED_CLASSES = {
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking",
}


class Sample(BaseModel):
    t: Optional[datetime] = None
    x: float
    y: float
    z: float


class ColetaPayload(BaseModel):
    device: str = Field(min_length=1, max_length=120)
    user_name: str = Field(min_length=1, max_length=120)
    samples: list[Sample] = Field(min_length=200, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    top_class_probability: float = Field(ge=0.0, le=1.0)
    predicted_class: str
    real_class: str
    is_correct: Optional[bool] = None

    @model_validator(mode="after")
    def validate_classes(self) -> "ColetaPayload":
        if self.predicted_class not in ALLOWED_CLASSES:
            raise ValueError("predicted_class invalida")
        if self.real_class not in ALLOWED_CLASSES:
            raise ValueError("real_class invalida")
        return self


class ColetaOut(BaseModel):
    id: int
    created_at: str
    device: str
    user_name: str
    sample_count: int
    confidence: float
    top_class_probability: float
    predicted_class: str
    real_class: str
    is_correct: bool


app = FastAPI(title="HAR Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coletas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                device TEXT NOT NULL,
                user_name TEXT NOT NULL,
                samples_json TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                top_class_probability REAL NOT NULL,
                predicted_class TEXT NOT NULL,
                real_class TEXT NOT NULL,
                is_correct INTEGER NOT NULL
            )
            """
        )
        conn.commit()


@app.post("/v1/coletas")
def create_coleta(payload: ColetaPayload) -> dict:
    inferred_is_correct = payload.predicted_class == payload.real_class
    is_correct = inferred_is_correct if payload.is_correct is None else payload.is_correct

    created_at = datetime.now(timezone.utc).isoformat()
    samples_json = json.dumps([s.model_dump(mode="json") for s in payload.samples])

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO coletas (
                created_at, device, user_name, samples_json, sample_count,
                confidence, top_class_probability, predicted_class, real_class, is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                payload.device,
                payload.user_name,
                samples_json,
                len(payload.samples),
                payload.confidence,
                payload.top_class_probability,
                payload.predicted_class,
                payload.real_class,
                1 if is_correct else 0,
            ),
        )
        conn.commit()

    return {
        "id": cursor.lastrowid,
        "received_at": created_at,
        "stored_samples": len(payload.samples),
        "is_correct": is_correct,
    }


@app.get("/v1/coletas", response_model=list[ColetaOut])
def list_coletas(limit: int = 200) -> list[ColetaOut]:
    safe_limit = max(1, min(limit, 2000))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, device, user_name, sample_count, confidence,
                   top_class_probability, predicted_class, real_class, is_correct
            FROM coletas
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [
        ColetaOut(
            id=row["id"],
            created_at=row["created_at"],
            device=row["device"],
            user_name=row["user_name"],
            sample_count=row["sample_count"],
            confidence=row["confidence"],
            top_class_probability=row["top_class_probability"],
            predicted_class=row["predicted_class"],
            real_class=row["real_class"],
            is_correct=bool(row["is_correct"]),
        )
        for row in rows
    ]


@app.delete("/v1/coletas")
def delete_all_coletas() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM coletas")
        conn.commit()

    return {
        "deleted": cursor.rowcount if cursor.rowcount is not None else 0,
        "message": "Dados apagados com sucesso",
    }


@app.get("/v1/dashboard-summary")
def dashboard_summary() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        by_class_rows = conn.execute(
            """
            SELECT predicted_class, COUNT(*) AS total
            FROM coletas
            GROUP BY predicted_class
            ORDER BY total DESC
            """
        ).fetchall()
        accuracy_rows = conn.execute(
            """
            SELECT is_correct, COUNT(*) AS total
            FROM coletas
            GROUP BY is_correct
            """
        ).fetchall()

    by_class = {row["predicted_class"]: row["total"] for row in by_class_rows}
    accuracy = {
        "correct": sum(row["total"] for row in accuracy_rows if row["is_correct"] == 1),
        "incorrect": sum(row["total"] for row in accuracy_rows if row["is_correct"] == 0),
    }

    return {"by_class": by_class, "accuracy": accuracy}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    index_file = BASE_DIR / "static" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard nao encontrado")
    return FileResponse(index_file)
