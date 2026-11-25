"""Canonical JSON, hash chaining and Ed25519 signatures for telemetry records.

Record shape (IMPLEMENTATION PLAN.md §6.2):

    {
      "v": 1,
      "device": {"id": ..., "attest_ref": null},
      "seq": 184223,
      "t": "...",
      "nonce": "...",
      "s": {"temp_core": ..., "temp_mem": ..., "power_w": ..., "xid": ...},
      "prev": "sha256(prev record)",
      "sig": "ed25519(...)"
    }

The signature covers every field except `sig` itself, computed over the
canonical JSON encoding (sorted keys, no whitespace) so verification is
unambiguous across languages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import nacl.encoding
import nacl.exceptions
import nacl.signing

from .backends import DeviceInfo, Sample

GENESIS_HASH = "0" * 64


def canonical_json(obj: dict[str, Any]) -> bytes:
    """Deterministic encoding: sorted keys, no extraneous whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SigningKey:
    """Wraps an Ed25519 keypair. One per agent (and one per witness)."""

    _sk: nacl.signing.SigningKey = field(repr=False)

    @classmethod
    def generate(cls) -> "SigningKey":
        return cls(nacl.signing.SigningKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "SigningKey":
        return cls(nacl.signing.SigningKey(bytes.fromhex(seed_hex)))

    def seed_hex(self) -> str:
        return self._sk.encode(encoder=nacl.encoding.HexEncoder).decode()

    def public_hex(self) -> str:
        return self._sk.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode()

    def sign(self, payload: bytes) -> str:
        return self._sk.sign(payload).signature.hex()


def verify_signature(public_hex: str, payload: bytes, sig_hex: str) -> bool:
    try:
        vk = nacl.signing.VerifyKey(bytes.fromhex(public_hex))
        vk.verify(payload, bytes.fromhex(sig_hex))
        return True
    except (nacl.exceptions.BadSignatureError, ValueError):
        return False


class ChainBuilder:
    """Builds a signed, hash-chained sequence of telemetry records for one device."""

    def __init__(self, device: DeviceInfo, signing_key: SigningKey, attest_ref: Optional[str] = None):
        self.device = device
        self.key = signing_key
        self.attest_ref = attest_ref
        self.seq = 0
        self.prev_hash = GENESIS_HASH

    def record(self, sample: Sample, nonce: str) -> dict[str, Any]:
        body = {
            "v": 1,
            "device": {"id": self.device.id, "attest_ref": self.attest_ref},
            "seq": self.seq,
            "t": sample.t,
            "nonce": nonce,
            "s": {
                "temp_core": sample.temp_core_c,
                "temp_mem": sample.temp_mem_c,
                "power_w": sample.power_w,
                "xid": sample.xid,
            },
            "prev": self.prev_hash,
        }
        sig = self.key.sign(canonical_json(body))
        full = {**body, "sig": sig}

        self.seq += 1
        self.prev_hash = sha256_hex(canonical_json(full))
        return full


def verify_chain(records: list[dict[str, Any]], public_hex: str) -> list[str]:
    """Returns a list of problems found; empty list means the chain verifies 100%.

    Checks, in order: monotonic seq, correct prev-hash linkage, and a valid
    signature on every record. This is what §6.2's acceptance test exercises:
    a single-field edit, a deletion, or a replayed batch each surface here.
    """
    problems: list[str] = []
    expected_prev = GENESIS_HASH
    expected_seq = 0

    for i, rec in enumerate(records):
        rec = dict(rec)
        sig = rec.pop("sig", None)
        if sig is None:
            problems.append(f"record {i}: missing signature")
            continue

        if rec.get("seq") != expected_seq:
            problems.append(f"record {i}: seq {rec.get('seq')} != expected {expected_seq} (gap or reorder)")

        if rec.get("prev") != expected_prev:
            problems.append(f"record {i}: prev hash mismatch — chain break or tamper")

        payload = canonical_json(rec)
        if not verify_signature(public_hex, payload, sig):
            problems.append(f"record {i}: signature does not verify")

        full = {**rec, "sig": sig}
        expected_prev = sha256_hex(canonical_json(full))
        expected_seq = rec.get("seq", expected_seq) + 1

    return problems
