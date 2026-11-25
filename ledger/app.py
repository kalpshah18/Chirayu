"""Minimal FastAPI ledger service (IMPLEMENTATION PLAN.md §6.4).

Endpoints: POST /v1/nonce, POST /v1/telemetry/batch, POST /v1/witness/batch,
GET /v1/devices/{id}/logbook, GET /v1/alerts, GET /v1/roots/{hour}.

This first cut keeps everything in process memory, so it runs with no
database for local development (`uvicorn ledger.app:app --reload`); swapping
in the SQLAlchemy models in ledger/models.py for persistence is the next
step once the dashboard needs data to survive a restart.
"""
from __future__ import annotations

import secrets
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.chain import verify_chain
from ledger.verify import ALERT_TYPES
from ledger.merkle import merkle_root

app = FastAPI(title="Chirayu Ledger", version="0.1.0")

# --- in-memory state (dev only) --------------------------------------------
_nonces: dict[str, float] = {}
_device_keys: dict[str, str] = {}          # device_id -> public_key_hex
_device_chains: dict[str, list[dict]] = defaultdict(list)
_witness_streams: dict[str, list[dict]] = defaultdict(list)
_alerts: list[dict] = []
NONCE_TTL_S = 60


class RegisterDevice(BaseModel):
    device_id: str
    public_key_hex: str


class TelemetryBatch(BaseModel):
    device_id: str
    records: list[dict[str, Any]]


class WitnessBatch(BaseModel):
    witness_id: str
    records: list[dict[str, Any]]


@app.post("/v1/devices/register")
def register_device(body: RegisterDevice):
    _device_keys[body.device_id] = body.public_key_hex
    return {"ok": True}


@app.post("/v1/nonce")
def issue_nonce():
    n = secrets.token_hex(16)
    _nonces[n] = time.time()
    return {"nonce": n, "ttl_s": NONCE_TTL_S}


def _nonce_is_fresh(nonce: str) -> bool:
    issued = _nonces.get(nonce)
    return issued is not None and (time.time() - issued) <= NONCE_TTL_S


@app.post("/v1/telemetry/batch")
def post_telemetry_batch(batch: TelemetryBatch):
    public_hex = _device_keys.get(batch.device_id)
    if public_hex is None:
        raise HTTPException(404, f"unknown device {batch.device_id}")

    existing = _device_chains[batch.device_id]
    combined = existing + batch.records
    problems = verify_chain(combined, public_hex)
    if problems:
        for p in problems:
            _alerts.append({
                "device_id": batch.device_id,
                "alert_type": "CHAIN_BREAK",
                "t": datetime.now(timezone.utc).isoformat(),
                "detail": {"reason": p},
            })
        raise HTTPException(409, {"problems": problems})

    _device_chains[batch.device_id] = combined
    return {"accepted": len(batch.records), "chain_length": len(combined)}


@app.post("/v1/witness/batch")
def post_witness_batch(batch: WitnessBatch):
    _witness_streams[batch.witness_id].extend(batch.records)
    return {"accepted": len(batch.records)}


@app.get("/v1/devices/{device_id}/logbook")
def get_logbook(device_id: str, limit: int = 1000):
    return {"device_id": device_id, "records": _device_chains.get(device_id, [])[-limit:]}


@app.get("/v1/alerts")
def get_alerts(alert_type: str | None = None):
    if alert_type and alert_type not in ALERT_TYPES:
        raise HTTPException(400, f"unknown alert_type, expected one of {ALERT_TYPES}")
    rows = _alerts if alert_type is None else [a for a in _alerts if a["alert_type"] == alert_type]
    return {"alerts": rows}


@app.get("/v1/roots/{hour}")
def get_root(hour: str, device_id: str):
    records = _device_chains.get(device_id, [])
    # Simplified: roots over the whole chain, not sliced by wall-clock hour yet.
    leaf_hashes = [r["sig"] for r in records]
    return {"hour": hour, "device_id": device_id, "root": merkle_root(leaf_hashes)}
