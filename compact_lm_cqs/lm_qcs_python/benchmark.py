from __future__ import annotations

from statistics import mean, median, stdev
from time import perf_counter
from common.rng import ShakeRNG

from .core import Signature, keygen, sign, verify
from .parameters import LMQCS_I, LMQCSParameters


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def benchmark(
    par: LMQCSParameters = LMQCS_I,
    repetitions: int = 10,
    message: bytes = b"LM-QCS benchmark",
    seed: int = 12345,
) -> dict[str, object]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    rng = ShakeRNG.from_int(seed)

    # Warm up Numba paths and serialization routines (excluded from measurements)
    warm_keys = keygen(par, rng)
    warm_sig = sign(b"warmup", warm_keys.secret_key, warm_keys.public_key, par, rng)
    warm_bytes = warm_sig.to_bytes(par)
    warm_deser = Signature.from_bytes(warm_bytes, par)
    if not verify(b"warmup", warm_deser, warm_keys.public_key, par):
        raise RuntimeError("Warm-up signature failed verification.")

    keygen_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []
    sig_sizes: list[float] = []

    for i in range(repetitions):
        # 1. KeyGen
        start = perf_counter()
        keys = keygen(par, rng)
        keygen_times.append(perf_counter() - start)

        # 2. Sign + Serialize (FALCON Compression)
        start = perf_counter()
        sig = sign(message + i.to_bytes(4, "little"), keys.secret_key, keys.public_key, par, rng)
        sig_bytes = sig.to_bytes(par)  # Executes compress_falcon
        sign_times.append(perf_counter() - start)

        # Record actual compressed byte length
        sig_sizes.append(float(len(sig_bytes)))

        # 3. Deserialize (FALCON Decompression) + Verify
        start = perf_counter()
        deserialized_sig = Signature.from_bytes(sig_bytes, par)  # Executes decompress_falcon
        ok = verify(message + i.to_bytes(4, "little"), deserialized_sig, keys.public_key, par)
        verify_times.append(perf_counter() - start)

        if not ok:
            raise RuntimeError("Generated signature failed verification.")

    return {
        "parameter_set": par.name,
        "repetitions": repetitions,
        "keygen_seconds": _stats(keygen_times),
        "sign_seconds": _stats(sign_times),
        "verify_seconds": _stats(verify_times),
        "public_key_bytes": par.public_key_bytes,
        "secret_key_bytes": par.secret_key_bytes,
        "uncompressed_signature_bytes": par.signature_bytes_uncompressed,
        "compressed_signature_bytes": _stats(sig_sizes),
    }