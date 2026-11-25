"""Signs Summit's power-supply input power as an independent witness stream
(IMPLEMENTATION PLAN.md §6.3). ps0_input_power + ps1_input_power is measured
by the power supplies, a separate hardware path from the on-chip controllers
that produce each GPU's self-reported power — exactly the separation a
physical witness meter provides at the pilot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.backends.replay import ReplayBackend
from agent.chain import canonical_json, sha256_hex, SigningKey

GENESIS_HASH = "0" * 64


@dataclass
class WitnessSample:
    t: str
    node_input_power_w: float


class ReplayWitness:
    """One witness stream per node (not per GPU — the PSUs power the whole node)."""

    def __init__(self, backend: ReplayBackend, signing_key: SigningKey):
        self.backend = backend
        self.key = signing_key
        self.seq = 0
        self.prev_hash = GENESIS_HASH

    def record_at(self, idx: int, timestamp: str) -> dict[str, Any] | None:
        power = self.backend.witness_input_power_w(idx)
        if power is None:
            return None
        body = {
            "v": 1,
            "witness_id": f"witness:{self.backend.hostname}",
            "seq": self.seq,
            "t": timestamp,
            "node_input_power_w": power,
            "prev": self.prev_hash,
        }
        sig = self.key.sign(canonical_json(body))
        full = {**body, "sig": sig}
        self.seq += 1
        self.prev_hash = sha256_hex(canonical_json(full))
        return full

    def record_stream(self) -> list[dict[str, Any]]:
        out = []
        for idx in range(len(self.backend)):
            ts = self.backend.df["timestamp"].iloc[idx]
            t_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rec = self.record_at(idx, t_iso)
            if rec is not None:
                out.append(rec)
        return out
