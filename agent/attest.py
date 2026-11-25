"""Hardware-identity binding design (IMPLEMENTATION PLAN.md §6.2).

NOT DEMONSTRATED in the finale. This module documents and stubs the pilot
design: binding the agent's Ed25519 key to an NVIDIA attestation report on
confidential-computing-capable H100s, so a lessor can trust that signed
telemetry actually came from the physical device it claims to.

Design (pilot stage):
  1. The agent generates its Ed25519 signing key inside the attested
     environment (e.g. an H100 confidential-compute VM).
  2. It requests an NVIDIA Attestation SDK report that includes the public
     key's hash as application data (nonce-bound, to prevent replay of an
     old report).
  3. The ledger verifies the attestation report against NVIDIA's reference
     measurements (via the NVIDIA Remote Attestation Service) before
     accepting any telemetry signed by that key, and records `attest_ref`
     on every subsequent record (see agent/chain.py).
  4. If attestation ever fails revalidation (key rotated outside the
     attested flow, device swapped), the ledger raises IDENTITY_CHANGE.

This module ships only the binding *logic* (steps 3-4, which do not require
real hardware) with tests. Step 2 is a stub that would call the NVIDIA SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AttestationReport:
    """A stand-in for what NVIDIA's Attestation SDK would return."""

    device_id: str
    public_key_hex: str
    nonce: str
    measurement_hash: str   # stand-in for GPU firmware/driver measurement
    signature_valid: bool   # whether the report's own signature checked out


class AttestationError(Exception):
    pass


def request_attestation_report(device_id: str) -> AttestationReport:
    """Stub. At the pilot, this calls the NVIDIA Attestation SDK on an
    H100 confidential-compute instance. Not available without real hardware.
    """
    raise NotImplementedError(
        "request_attestation_report requires the NVIDIA Attestation SDK on a "
        "confidential-computing-capable GPU. This is a pilot-stage integration, "
        "not demonstrated at the finale (IMPLEMENTATION PLAN.md §1.2)."
    )


def verify_binding(report: AttestationReport, expected_public_key_hex: str,
                    expected_nonce: str, known_good_measurements: set[str]) -> Optional[str]:
    """Pure verification logic the ledger runs. Returns None if the binding
    holds, else a human-readable reason it failed (so it maps cleanly onto
    the IDENTITY_CHANGE alert type in ledger/divergence.py's sibling checks).
    """
    if report.public_key_hex.lower() != expected_public_key_hex.lower():
        return "attestation reports a different public key than the one signing telemetry"
    if report.nonce != expected_nonce:
        return "attestation nonce does not match — possible replay of an old report"
    if not report.signature_valid:
        return "attestation report signature is invalid"
    if report.measurement_hash not in known_good_measurements:
        return "attestation measurement does not match any known-good firmware/driver build"
    return None
