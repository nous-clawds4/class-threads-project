"""Deterministic RNG seeding for reproducible experiments."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def make_rng(*parts: Any) -> np.random.Generator:
    """A numpy ``Generator`` seeded deterministically from a tuple of factors.

    The same ``parts`` always yield the same stream, so a
    ``(dataset, arm, rho, seed, ...)`` tuple reproduces a trial exactly across
    machines and runs. Uses BLAKE2b over ``repr(parts)`` so ordering and types
    are part of the seed.
    """
    payload = repr(parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "big") % (2 ** 32)
    return np.random.default_rng(seed)
