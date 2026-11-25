"""Hourly Merkle roots over a device's records, so anyone can audit a single
record's inclusion without trusting the ledger operator with the whole log
(IMPLEMENTATION PLAN.md §6.4).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


@dataclass
class MerkleProof:
    leaf_index: int
    siblings: list[str]  # hex, bottom to top


def merkle_root(leaf_hashes_hex: list[str]) -> str:
    if not leaf_hashes_hex:
        return "0" * 64
    level = [bytes.fromhex(h) for h in leaf_hashes_hex]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # duplicate last node on an odd layer
        level = [_h(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0].hex()


def merkle_proof(leaf_hashes_hex: list[str], index: int) -> MerkleProof:
    level = [bytes.fromhex(h) for h in leaf_hashes_hex]
    siblings = []
    idx = index
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        sib_idx = idx ^ 1
        siblings.append(level[sib_idx].hex())
        level = [_h(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return MerkleProof(leaf_index=index, siblings=siblings)


def verify_proof(leaf_hash_hex: str, proof: MerkleProof, root_hex: str) -> bool:
    node = bytes.fromhex(leaf_hash_hex)
    idx = proof.leaf_index
    for sib_hex in proof.siblings:
        sib = bytes.fromhex(sib_hex)
        node = _h(node + sib) if idx % 2 == 0 else _h(sib + node)
        idx //= 2
    return node.hex() == root_hex
