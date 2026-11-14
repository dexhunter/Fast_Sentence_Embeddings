#!/usr/bin/env python

# Author: intake harness (baseline only; does not change fse)
# SPDX-License-Identifier: MIT

"""Deterministic baselines for the osspr intake targets.

Each target times one existing function on a realistic workload. Data is built
outside the timer. One warmup call, then five timed calls. Prints one JSON
object per target: median milliseconds, the five times, and the full spread
``(max - min) / median * 100``.

    uv run python osspr-intake/harness/bench.py hamming
    uv run python osspr-intake/harness/bench.py quantize
    uv run python osspr-intake/harness/bench.py strings
    uv run python osspr-intake/harness/bench.py tokens
    uv run python osspr-intake/harness/bench.py scales
    uv run python osspr-intake/harness/bench.py all
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

REPEATS = 5
WARMUP = 1

# Ordinary English tokens, reused the way a real vocabulary is reused.
_WORDS = (
    "the a of and to in is it that for on with as was at by an be this from or "
    "are were not you your we our they their his her has have had but if can "
    "will would about into over after before under between during without "
    "people time year day work world life city water small large new old good "
    "long little own other right still such here there when where what which "
    "who how than then them these those just also only even because through "
    "sentence document query vector model word token text search result"
).split()


def _times(fn) -> list[float]:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _report(name: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    spread = (ordered[-1] - ordered[0]) / median * 100.0
    print(
        json.dumps(
            {
                "target": name,
                "repeats": samples,
                "baseline_median_ms": median,
                "baseline_min_ms": ordered[0],
                "baseline_max_ms": ordered[-1],
                "baseline_noise_pct": spread,
            }
        ),
        flush=True,
    )


def _sentences(n: int, length: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(_WORDS), size=(n, length))
    return [" ".join(_WORDS[i] for i in row) for row in ids]


def bench_hamming() -> None:
    """Sign bits of pooled document vectors against a small query batch.

    Same shape binary search feeds ``fse._reference_bits.hamming``: many rows,
    a few dozen queries, 384 dimensions (48 packed bytes).
    """
    from fse._reference_bits import hamming, query_bits, sign_bits

    rng = np.random.default_rng(0)
    stored = rng.standard_normal((120_000, 384)).astype(np.float32)
    queries = rng.standard_normal((24, 384)).astype(np.float32)
    bits = sign_bits(stored)
    qbits = query_bits(queries)
    _report("hamming", _times(lambda: hamming(bits, qbits)))


def bench_quantize() -> None:
    """Int8 quantization of a static embedding table (200k rows, 384 dims)."""
    from fse._codes import quantize_into

    rng = np.random.default_rng(1)
    rows = rng.standard_normal((200_000, 384)).astype(np.float32)
    # A few exact halves, so rounding is the real half-to-even path.
    rows[::4096, 0] = np.float32(0.5)
    codes = np.empty(rows.shape, np.int8)
    scales = np.empty(len(rows), np.float32)

    def run() -> None:
        quantize_into(rows, codes, scales)

    _report("quantize", _times(run))


def bench_strings() -> None:
    """``StringSource`` on a 1-d numpy unicode array, as ``np.asarray(texts)``."""
    from fse._text._strings import StringSource

    texts = np.asarray(_sentences(250_000, 12, seed=2))
    _report("strings", _times(lambda: StringSource(texts)))


def bench_tokens() -> None:
    """``TokenListSource`` on a 1-d object array of token lists.

    That is the array a column of pretokenized lists becomes (``dtype=object``
    holding the lists themselves, not a 2-d array of tokens).
    """
    from fse._text._strings import TokenListSource

    rng = np.random.default_rng(3)
    n, length = 160_000, 12
    ids = rng.integers(0, len(_WORDS), size=(n, length))
    lists = [[_WORDS[i] for i in row] for row in ids]
    sentences = np.empty(n, dtype=object)
    for i, tokens in enumerate(lists):
        sentences[i] = tokens
    _report("tokens", _times(lambda: TokenListSource(sentences)))


def bench_scales() -> None:
    """The int8 scale check used when a quantized table is built or verified.

    400k rows by 300 dimensions, codes in -127..127, positive finite scales.
    """
    from fse._wordvectors_format import _check_scale_values

    rng = np.random.default_rng(4)
    rows = rng.integers(-127, 128, size=(400_000, 300), dtype=np.int8)
    scales = np.abs(rng.standard_normal(400_000)).astype(np.float32) * np.float32(0.02)
    scales[scales == 0] = np.float32(0.02)

    def run() -> None:
        _check_scale_values("vectors", "scales", rows, scales)

    _report("scales", _times(run))


_TARGETS = {
    "hamming": bench_hamming,
    "quantize": bench_quantize,
    "strings": bench_strings,
    "tokens": bench_tokens,
    "scales": bench_scales,
}


def main(argv: list[str]) -> None:
    name = argv[1] if len(argv) > 1 else "all"
    if name == "all":
        for fn in _TARGETS.values():
            fn()
        return
    if name not in _TARGETS:
        raise SystemExit(f"unknown target {name!r}; choose from {', '.join(_TARGETS)} or all")
    _TARGETS[name]()


if __name__ == "__main__":
    main(sys.argv)
