# Compact LM-QCS

A Python research implementation of the **Short Lee-Metric Code-based Signature (LM-QCS)** scheme of Tan and Prabowo (VCRIS 2024), extended with **Falcon-style Gaussian compression** of the signature responses.

This repository accompanies my master's dissertation, where the compression technique is proposed as an improvement on the original LM-QCS signature size. It supports the three parameter sets of the original paper (128-, 192-, and 256-bit classical security).

> **Disclaimer.** This is research code. It is not the authors' official implementation, it has not been independently audited, and it is **not constant-time**. Do not use it to protect real data.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Shared utilities](#shared-utilities)
- [Parameter sets](#parameter-sets)
- [Signature format and compression](#signature-format-and-compression)
- [Benchmarks](#benchmarks)
- [Implementation choices](#implementation-choices)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Overview

LM-QCS is a hash-and-sign style signature scheme built on quasi-cyclic codes in the Lee metric. Keys and signatures live in the ring `R_q = F_q[x]/(x^n - 1)`:

- **KeyGen** samples `e1, e2` with coefficients in `[-ell_e, ell_e]`, requires `e1` to be invertible, and publishes `h = e1^{-1} e2`.
- **Sign** applies a random cyclic shift to the secret key, samples a sparse ternary vector `e_bar` and two box vectors `u1, u2`, derives a Fiat-Shamir challenge `c`, and outputs `(c, s1, s2)`. The scheme is specified *without aborts*, so signing makes a single ephemeral draw.
- **Verify** reconstructs `t = s1·h - s2`, computes `c^{-1} t`, recomputes the challenge, and checks the infinity-norm conditions.

The original paper reports signature sizes assuming fixed-width coefficients. Since the coefficients of `s1` and `s2` are approximately Gaussian, most of them are much smaller than the worst-case bound `gamma`. This implementation therefore encodes them with the variable-length compression used by [Falcon](https://falcon-sign.info/), which makes the signature shorter on average.

## Features

- Complete `KeyGen` → `Sign` → `Verify` pipeline for the three parameter sets.
- Falcon-style compression and decompression of `s1` and `s2`, with a per-level LSB width `b`.
- Enumerative (combination-rank plus sign bits) encoding of the challenge `c`.
- Fixed-size serialization of public and secret keys.
- Strict signature parsing: truncated input, trailing bytes, and out-of-bound coefficients are rejected.
- Reproducible runs through a seedable SHAKE-based random number generator.
- Numba-accelerated ring arithmetic (with a pure-Python fallback).
- Benchmark script that excludes JIT warm-up from the measurements.

## Repository layout

```
.
├── common/               # Shared utilities (see "Shared utilities" below)
│   ├── __init__.py       # Exports ShakeRNG and SigningFailure
│   ├── rng.py            # ShakeRNG: SHAKE256-based RNG (seedable or OS-seeded)
│   └── errors.py         # SigningFailure exception (not used by LM-QCS itself)
└── lm_qcs_python/
    ├── __init__.py       # Public API
    ├── parameters.py     # Parameter sets, derived sizes, consistency checks
    ├── core.py           # KeyGen, Sign, Verify, key/signature (de)serialization
    ├── ring.py           # Arithmetic in F_q[x]/(x^n - 1), polynomial inversion
    ├── sampling.py       # Box, sparse ternary, and cyclic-shift sampling
    ├── challenge.py      # Combinatorial rank/unrank encoding of the challenge
    ├── hashing.py        # SHAKE256 hash-to-challenge
    ├── encoding.py       # Bit packing and Falcon compression/decompression
    ├── encoding_base.py  # Fixed-width bit-packing helpers (not used by core.py)
    ├── run.py            # Command-line demo
    └── benchmark.py      # Timing and size benchmark
```

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install numpy numba
```

Python 3.8 or later is required (the code uses `math.comb`). Numba is optional but strongly recommended: without it, the ring arithmetic falls back to plain Python and is much slower.

## Quick start

### Command line

Run the commands below from the repository root:

```bash
python -m lm_qcs_python.run --level 128
python -m lm_qcs_python.run --level 192
python -m lm_qcs_python.run --level 256
```

Options: `--level {128,192,256}`, `--message TEXT`, and `--seed INT` (the default seed is `12345`). The script prints timings, the verification result, and the public key, secret key, and signature sizes in bytes.

### Python API

```python
from lm_qcs_python import LMQCS_I, keygen, sign, verify, Signature

par = LMQCS_I                      # or LMQCS_II, LMQCS_III
message = b"hello, post-quantum world"

keys = keygen(par)
sig = sign(message, keys.secret_key, keys.public_key, par)
assert verify(message, sig, keys.public_key, par)

# Serialize with Falcon compression and parse it back
sig_bytes = sig.to_bytes(par)
restored = Signature.from_bytes(sig_bytes, par)
assert verify(message, restored, keys.public_key, par)

print(len(keys.public_key.to_bytes(par)),
      len(keys.secret_key.to_bytes(par)),
      len(sig_bytes))
```

For reproducible runs, pass a seeded generator:

```python
from common.rng import ShakeRNG

rng = ShakeRNG.from_int(2024)
keys = keygen(par, rng)
sig = sign(message, keys.secret_key, keys.public_key, par, rng)
```

If no generator is given, a system-seeded one is used.

## Shared utilities

The `common/` package holds code that is shared across scheme implementations and is imported as `from common.rng import ShakeRNG` (so scripts must be run from the repository root).

**`ShakeRNG`** is a deterministic random source built on SHAKE256 in counter mode. It is constructed from a non-empty byte-string seed, or through two helpers:

- `ShakeRNG.from_int(seed)` creates a reproducible generator from a non-negative integer;
- `ShakeRNG.from_system()` seeds the generator from the operating system (`os.urandom`, 32 bytes by default); this is what `keygen` and `sign` use when no generator is passed.

Besides `random_bytes(size)`, it provides `randbelow(upper_bound)` (uniform integers by rejection sampling) and `sample_positions(population_size, weight)` (uniform sampling without replacement through a partial Fisher-Yates shuffle). The latter is used to draw the support of the sparse vector `e_bar`.

**`SigningFailure`** is a `RuntimeError` subclass that records the number of attempts made. LM-QCS does not use it, because signing is specified without aborts and raises a plain `RuntimeError` on the negligible norm-failure event.

## Parameter sets

The sets below are those of Tables I and II of the original paper. `parameters.py` checks at import time that the computed public key, secret key, and uncompressed signature sizes match the reported ones.

| Set | Security | `n` | `q` | `ell_e` | `ell_u` | `omega_e_bar` | `omega_c` | `t` |
|---|---|---|---|---|---|---|---|---|
| `LMQCS_I` | 128 bits | 691 | 5479 | 2 | 28 | 345 | 41 | 10.1 |
| `LMQCS_II` | 192 bits | 1009 | 8623 | 3 | 35 | 455 | 61 | 11.0 |
| `LMQCS_III` | 256 bits | 1201 | 12569 | 4 | 41 | 705 | 83 | 10.1 |

The response bound is `gamma = ceil(t · sigma_s)`, where `sigma_s` is derived from the parameters in `parameters.py`.

Fixed-size objects (bytes):

| Set | Public key | Secret key | Signature (uncompressed baseline) | Signature (compressed, expected) |
|---|---|---|---|---|
| `LMQCS_I` | 1123 | 519 | 2106 | 1570 |
| `LMQCS_II` | 1766 | 757 | 3076 | 2467 |
| `LMQCS_III` | 2102 | 1201 | 3968 | 3056 |

The last column is an analytical estimate (`expected_signature_bytes_compressed` in `parameters.py`). It models each response coefficient as a zero-mean Gaussian with standard deviation `sigma_s` and adds the expected unary length to the fixed per-coefficient bits. It ignores the byte padding of each vector, so it is only an approximation.

The actual compressed signature has a **variable length**, so the size of a given signature differs from this value. See the next section and the [benchmarks](#benchmarks) for measured sizes.

## Signature format and compression

A serialized signature is the concatenation

```
signature = c_bytes || s1_bytes || s2_bytes
```

**Challenge `c`.** The ternary challenge with `omega_c` non-zero entries is encoded as a single integer, `rank(support) · 2^omega_c + sign_bits`, where `rank` is the lexicographic rank of the support among all `omega_c`-subsets of `{0, …, n-1}`. It occupies `ceil(challenge_bits / 8)` bytes.

**Responses `s1`, `s2`.** Each coefficient `s_i` is written, bit by bit (least-significant bit first within each byte), as:

1. one sign bit (`1` if `s_i > 0`, otherwise `0`);
2. the `b` low-order bits of `|s_i|`;
3. the high part `k = floor(|s_i| / 2^b)` in unary: `k` zeros followed by a terminating `1`.

Each of the two vectors is padded to a byte boundary. The decoder reads `n` coefficients per vector, and `Signature.from_bytes` raises an error if the number of bytes consumed differs from the input length or if any coefficient exceeds `gamma`.

The LSB width `b` is chosen per parameter set and is configured through the `falcon_lsb_bits` field in `parameters.py`:

| Set | `b` |
|---|---|
| `LMQCS_I` | 6 |
| `LMQCS_II` | 7 |
| `LMQCS_III` | 7 |

Larger `b` spends more fixed bits per coefficient but shortens the unary part; the best value depends on `sigma_s`.

## Benchmarks

```python
from lm_qcs_python import LMQCS_I
from lm_qcs_python.benchmark import benchmark

print(benchmark(LMQCS_I, repetitions=30))
```

The returned dictionary contains mean, median, standard deviation, minimum, and maximum for:

- `keygen_seconds`;
- `sign_seconds` (signing plus serialization with compression);
- `verify_seconds` (deserialization with decompression plus verification);
- `compressed_signature_bytes`, the measured size over all repetitions.

It also reports the fixed key sizes and the uncompressed signature size for comparison. A warm-up run (key generation, signing, and verification) is executed first, so Numba compilation time is excluded. Timings depend on the hardware and on the Python and Numba versions.

## Implementation choices

The original paper is six pages long and does not fix a hash-to-challenge algorithm or a wire format, so the following choices are specific to this implementation:

1. **Hash-to-challenge:** SHAKE256 with a domain-separation prefix (`LM-QCS-H-v1`), taking the message, the vector `v`, and the public key `h` as input.
2. **Uniform challenges:** exact rejection sampling over the set of signed, weight-`omega_c` ternary vectors.
3. **Invertibility:** non-invertible challenges are rejected, as the paper requires.
4. **Challenge encoding:** enumerative combination ranking plus sign bits, matching the challenge-size term of the paper's signature-size formula.
5. **Response encoding:** Falcon-style compression (this repository's contribution) instead of fixed-width packing.
6. **No aborts:** `sign()` performs a single ephemeral draw. If the negligible norm-failure event occurs, it raises a `RuntimeError` rather than silently introducing rejection sampling.

## Limitations

- Not constant-time; variable-time arithmetic and variable-length encoding may leak information through timing.
- Pure research code: not audited, and not validated by the scheme's authors.
- `ShakeRNG` is a simple SHAKE256 counter-mode generator, OS-seeded by default; it has not been reviewed for production cryptographic use.
- Ring multiplication is schoolbook (`O(n^2)`), accelerated only by Numba.
- The compressed signature size varies from signature to signature.

## Citation

If you use this code, please cite the original scheme and, for the compression technique, the dissertation:

```bibtex
@inproceedings{tan2024lmqcs,
  author    = {Tan, Chik How and Prabowo, Theo Fanuela},
  title     = {Short {L}ee-Metric Code-Based Signature},
  booktitle = {VCRIS 2024},
  year      = {2024},
  doi       = {10.1109/VCRIS63677.2024.10813379}
}

@mastersthesis{Santos2026comparative,
  author = {Pablo H. Santos Moreira},
  title  = {Quasi-Cyclic Code-Based Digital Signature Schemes: Comparative Analysis under Hamming and Lee Metrics},
  school = {National Laboratory of Scientific Computation - LNCC},
  year   = {2026}
}
```

When available, please cite the associated article. GitHub reads the `CITATION.cff` file at the repository root and shows a "Cite this repository" button. Its contents are:

```yaml
cff-version: "1.2.0"
message: >-
  If you use this software, please cite this repository and, when available,
  the associated article.
title: "Comparative Evaluation of Quasi-Cyclic Code-Based Signature Schemes under Hamming and Lee Metrics"
type: software
license: BSD-3-Clause
authors:
  - family-names: "Moreira"
    given-names: "Pablo H. Santos"
  - family-names: "Pereira"
    given-names: "Diogo"
  - family-names: "Borges"
    given-names: "Fábio"
```

The compression follows the Falcon specification: Prest et al., *Falcon: Fast-Fourier Lattice-based Compact Signatures over NTRU*, <https://falcon-sign.info/>.

## License

See the `LICENSE` file.
