"""Normalises raw provider quotes to USD per GPU-hour by SKU, term and region
(IMPLEMENTATION PLAN.md §6.7). INR conversion uses the RBI reference rate of
the day, passed in rather than fetched here to keep this module pure and
testable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .scrapers import RawQuote


@dataclass
class NormalizedQuote:
    date: str
    sku: str
    provider: str
    term: str
    region: str
    usd_per_gpu_hr: float


def normalize(quote: RawQuote, usd_inr_rate: float | None = None) -> NormalizedQuote:
    if quote.currency.upper() == "USD":
        usd = quote.price
    elif quote.currency.upper() == "INR":
        if not usd_inr_rate:
            raise ValueError(f"INR quote from {quote.provider} needs usd_inr_rate")
        usd = quote.price / usd_inr_rate
    else:
        raise ValueError(f"unsupported currency: {quote.currency}")

    return NormalizedQuote(
        date=quote.fetched_at[:10],
        sku=quote.sku,
        provider=quote.provider,
        term=quote.term,
        region=quote.region,
        usd_per_gpu_hr=usd,
    )
