"""Provider scrapers for public GPU rental prices (IMPLEMENTATION PLAN.md §6.7).

One module per provider, each exposing `fetch() -> list[RawQuote]`. Respect
robots.txt and each site's terms; cache raw responses under data/raw/market/
so a parsing bug doesn't force re-scraping. Not implemented in this first cut
beyond the interface and a synthetic provider for local development — wiring
real providers (E2E Networks, Yotta, RunPod, Lambda, Vast.ai) is M0/M1 work
per the milestone table (§8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RawQuote:
    provider: str
    sku: str
    term: str          # e.g. "on-demand", "1-month reserved"
    region: str
    price: float
    currency: str
    fetched_at: str    # ISO-8601 UTC


class ProviderScraper(Protocol):
    name: str

    def fetch(self) -> list[RawQuote]: ...
