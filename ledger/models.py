"""SQLAlchemy models for the ledger service (IMPLEMENTATION PLAN.md §6.4).

SQLite for development, Postgres for the demo (same models, different URL).
Not all tables are wired into ledger/app.py yet in this first cut — the
schema is defined so api/ and the dashboard have a stable contract to build
against, per §7 (data contracts).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    sku: Mapped[str] = mapped_column(String, default="unknown")
    site: Mapped[str] = mapped_column(String, default="unknown")
    public_key_hex: Mapped[str] = mapped_column(String)
    attest_ref: Mapped[str | None] = mapped_column(String, nullable=True)


class Lease(Base):
    __tablename__ = "leases"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    lessee_id: Mapped[str] = mapped_column(String)
    start_date: Mapped[datetime] = mapped_column(DateTime)
    term_months: Mapped[int] = mapped_column(Integer)
    base_rent: Mapped[float] = mapped_column(Float)


class TelemetryRecord(Base):
    __tablename__ = "telemetry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    seq: Mapped[int] = mapped_column(Integer)
    t: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON)  # full signed record, canonical form
    record_hash: Mapped[str] = mapped_column(String)


class WitnessRecord(Base):
    __tablename__ = "witness"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    witness_id: Mapped[str] = mapped_column(String)
    seq: Mapped[int] = mapped_column(Integer)
    t: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON)


class Gap(Base):
    __tablename__ = "gaps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    from_seq: Mapped[int] = mapped_column(Integer)
    to_seq: Mapped[int] = mapped_column(Integer)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    alert_type: Mapped[str] = mapped_column(String)  # see ledger/verify.py ALERT_TYPES
    t: Mapped[datetime] = mapped_column(DateTime)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


class MerkleRootRow(Base):
    __tablename__ = "merkle_roots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hour: Mapped[datetime] = mapped_column(DateTime)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    root_hex: Mapped[str] = mapped_column(String)


class GsuMonthly(Base):
    __tablename__ = "gsu_monthly"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"))
    month: Mapped[str] = mapped_column(String)  # "YYYY-MM"
    gsu: Mapped[float] = mapped_column(Float)
    gsu_lo: Mapped[float] = mapped_column(Float)
    gsu_hi: Mapped[float] = mapped_column(Float)
