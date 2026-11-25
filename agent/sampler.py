"""Drives a TelemetryBackend at a fixed cadence and hands samples to a
ChainBuilder, batching every N records for upload (IMPLEMENTATION PLAN.md §6.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .backends import DeviceInfo, TelemetryBackend
from .chain import ChainBuilder, SigningKey


@dataclass
class SamplerConfig:
    batch_size: int = 10          # records per batch (≈10s at 1 Hz, per plan)
    nonce_source: Callable[[], str] | None = None  # ledger-issued; rotates every 60s


def _default_nonce(counter: list[int]) -> Callable[[], str]:
    def _next() -> str:
        counter[0] += 1
        return f"local-nonce-{counter[0]}"

    return _next


class Sampler:
    def __init__(self, backend: TelemetryBackend, device: DeviceInfo, signing_key: SigningKey,
                 config: SamplerConfig | None = None):
        self.backend = backend
        self.device = device
        self.chain = ChainBuilder(device, signing_key)
        self.config = config or SamplerConfig()
        self._nonce = self.config.nonce_source or _default_nonce([0])

    def run_batches(self, n_batches: int) -> Iterator[list[dict[str, Any]]]:
        for _ in range(n_batches):
            batch = []
            for _ in range(self.config.batch_size):
                sample = self.backend.sample(self.device)
                batch.append(self.chain.record(sample, nonce=self._nonce()))
            yield batch

    def run_all(self) -> Iterator[list[dict[str, Any]]]:
        """For ReplayBackend: drains the source, batching until exhausted."""
        try:
            n_total = len(self.backend)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError("run_all() needs a sized backend (e.g. ReplayBackend); use run_batches() otherwise")
        n_batches = (n_total + self.config.batch_size - 1) // self.config.batch_size
        yield from self.run_batches(n_batches)
