"""Re-exports the pure chain-verification logic for use by the ledger service
and by eval/tamper_eval.py. Kept separate from agent/chain.py so the ledger
package has no import-time dependency on the agent's own internals beyond
this function and the alert vocabulary below.
"""
from __future__ import annotations

from agent.chain import verify_chain  # noqa: F401

ALERT_TYPES = (
    "POWER_DIVERGENCE",
    "THERMAL_INCONSISTENT",
    "CHAIN_BREAK",
    "GAP",
    "IDENTITY_CHANGE",
    "WITNESS_SILENT",
)
