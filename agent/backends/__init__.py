"""Telemetry backends.

Both ReplayBackend (finale, driven by real Summit / MIT Supercloud traces) and
NvmlBackend (pilot, driven by a real GPU) implement the same TelemetryBackend
protocol, so nothing downstream of the agent knows or cares which one is live.
See IMPLEMENTATION PLAN.md §6.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class DeviceInfo:
    """Identifies one GPU. For replay this is hostname + slot; for NVML it is the real UUID."""

    id: str                 # e.g. "summit:a01n03:gpu2" or a real NVML UUID
    sku: str = "unknown"     # e.g. "V100", "H100"
    site: str = "unknown"


@dataclass(frozen=True)
class Sample:
    """One point-in-time reading. Fields the source lacks are None, never 0."""

    t: str                              # ISO-8601 UTC timestamp
    temp_core_c: Optional[float]
    temp_mem_c: Optional[float]
    power_w: Optional[float]
    energy_j: Optional[float] = None
    ecc_uncorrected: Optional[int] = None
    remapped_rows: Optional[int] = None
    xid: Optional[int] = None
    utilization_pct: Optional[float] = None


class TelemetryBackend(Protocol):
    """Implemented identically by ReplayBackend and NvmlBackend."""

    def devices(self) -> list[DeviceInfo]: ...

    def sample(self, dev: DeviceInfo) -> Sample: ...
