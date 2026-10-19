from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .challenge import decode_challenge, encode_challenge
from .encoding import (
    compress_falcon,
    decompress_falcon,
    pack_values,
    unpack_values,
)
from .hashing import hash_to_challenge
from .parameters import LMQCS_I, LMQCSParameters
from .ring import cyclic_mul, norm_inf, poly_inverse, sparse_cyclic_mul, symmetric
from common.rng import ShakeRNG
from .sampling import circular_shift, sample_box, sample_sparse_ternary


# Optimal LSB width b per security level
FALCON_B_MAP: dict[str, int] = {
    "LM-QCS-I": 6,
    "LM-QCS-II": 7,
    "LM-QCS-III": 7,
}


@dataclass(frozen=True)
class PublicKey:
    h: np.ndarray
    parameter_name: str

    def to_bytes(self, par: LMQCSParameters = LMQCS_I) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Parameter mismatch.")
        out = pack_values((self.h % par.q).tolist(), par.q_bits)
        if len(out) != par.public_key_bytes:
            raise RuntimeError("Unexpected public key size.")
        return out

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS_I) -> "PublicKey":
        if len(data) != par.public_key_bytes:
            raise ValueError("Invalid public key size.")
        h = unpack_values(data, par.n, par.q_bits)
        if np.any(h >= par.q):
            raise ValueError("Public key coefficient outside F_q.")
        return cls(h, par.name)


@dataclass(frozen=True)
class SecretKey:
    e1: np.ndarray
    e2: np.ndarray
    parameter_name: str

    def to_bytes(self, par: LMQCSParameters = LMQCS_I) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Parameter mismatch.")
        values = np.concatenate((self.e1, self.e2)) + par.ell_e
        out = pack_values(values.tolist(), par.secret_coeff_bits)
        if len(out) != par.secret_key_bytes:
            raise RuntimeError("Unexpected secret key size.")
        return out

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS_I) -> "SecretKey":
        if len(data) != par.secret_key_bytes:
            raise ValueError("Invalid secret key size.")
        values = unpack_values(data, 2 * par.n, par.secret_coeff_bits) - par.ell_e
        if np.any(np.abs(values) > par.ell_e):
            raise ValueError("Secret key coefficient outside U_ell_e.")
        return cls(values[:par.n], values[par.n:], par.name)


@dataclass(frozen=True)
class Signature:
    c: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    parameter_name: str
    elapsed_seconds: float = 0.0
    attempts: int = 1

    def to_bytes(self, par: LMQCSParameters = LMQCS_I) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Parameter mismatch.")
        
        b = FALCON_B_MAP.get(par.name, 7)

        # 1. Encode combinatorial challenge integer
        c_value = encode_challenge(self.c, par.n, par.omega_c, par.challenge_bits)
        c_bytes = pack_values([c_value], par.challenge_bits)

        # 2. Variable-length FALCON Gaussian compression for s1 and s2 using level-specific b
        s1_bytes = compress_falcon(self.s1, b=b)
        s2_bytes = compress_falcon(self.s2, b=b)

        return c_bytes + s1_bytes + s2_bytes

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS_I) -> "Signature":
        c_bytes_len = (par.challenge_bits + 7) // 8
        if len(data) <= c_bytes_len:
            raise ValueError("Signature payload too short.")

        b = FALCON_B_MAP.get(par.name, 7)

        # 1. Unpack challenge c
        c_encoded = unpack_values(data[:c_bytes_len], 1, par.challenge_bits)
        c = decode_challenge(int(c_encoded[0]), par.n, par.omega_c)

        # 2. Decompress s1 starting after challenge bytes using level-specific b
        s1, s1_consumed = decompress_falcon(data, par.n, b=b, byte_offset=c_bytes_len)

        # 3. Decompress s2 starting immediately after s1 using level-specific b
        s2, s2_consumed = decompress_falcon(data, par.n, b=b, byte_offset=c_bytes_len + s1_consumed)

        # 4. Strict byte count validation
        total_consumed = c_bytes_len + s1_consumed + s2_consumed
        if total_consumed != len(data):
            raise ValueError(
                f"Signature byte mismatch: payload has {len(data)} bytes, "
                f"decompression consumed {total_consumed} bytes."
            )

        if norm_inf(s1) > par.gamma or norm_inf(s2) > par.gamma:
            raise ValueError("Signature coefficient outside bound.")

        return cls(c, s1, s2, par.name)


@dataclass(frozen=True)
class KeyPair:
    public_key: PublicKey
    secret_key: SecretKey
    elapsed_seconds: float = 0.0
    attempts: int = 1


def keygen(
    par: LMQCSParameters = LMQCS_I,
    rng: ShakeRNG | None = None,
) -> KeyPair:
    """Generate e1,e2 in U_ell_e, require e1 invertible, and h=e1^-1 e2."""
    rng = rng or ShakeRNG.from_system()
    attempts = 0
    while True:
        attempts += 1
        e1 = sample_box(par.n, par.ell_e, rng)
        try:
            e1_inv = poly_inverse(e1, par.q)
            break
        except ValueError:
            continue
    e2 = sample_box(par.n, par.ell_e, rng)
    h = cyclic_mul(e1_inv, e2, par.q)
    return KeyPair(
        public_key=PublicKey(h, par.name),
        secret_key=SecretKey(e1, e2, par.name),
        attempts=attempts,
    )


def sign(
    message: bytes,
    secret_key: SecretKey,
    public_key: PublicKey,
    par: LMQCSParameters = LMQCS_I,
    rng: ShakeRNG | None = None,
) -> Signature:
    if secret_key.parameter_name != par.name or public_key.parameter_name != par.name:
        raise ValueError("Parameter mismatch.")
    rng = rng or ShakeRNG.from_system()

    r = rng.randbelow(par.n)
    sign_bit = rng.randbelow(2)
    sign_factor = -1 if sign_bit else 1
    e_bar = sample_sparse_ternary(par.n, par.omega_e_bar, rng)
    u1 = sample_box(par.n, par.ell_u, rng)
    u2 = sample_box(par.n, par.ell_u, rng)

    u1h_minus_u2 = symmetric(cyclic_mul(u1, public_key.h, par.q) - u2, par.q)
    hash_vector = sign_factor * u1h_minus_u2
    c, _ = hash_to_challenge(message, hash_vector, public_key.h, par)

    shifted_e1 = circular_shift(secret_key.e1, r)
    shifted_e2 = circular_shift(secret_key.e2, r)
    s1 = symmetric(
        sparse_cyclic_mul(e_bar, shifted_e1, par.q)
        + sign_factor * sparse_cyclic_mul(c, u1, par.q),
        par.q,
    )
    s2 = symmetric(
        sparse_cyclic_mul(e_bar, shifted_e2, par.q)
        + sign_factor * sparse_cyclic_mul(c, u2, par.q),
        par.q,
    )

    if norm_inf(s1) > par.gamma or norm_inf(s2) > par.gamma:
        raise RuntimeError(
            "The negligible LM-QCS norm-failure event occurred; no rejection "
            "sampling was applied because the paper specifies without aborts."
        )

    return Signature(c, s1, s2, par.name, attempts=1)


def verify(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: LMQCSParameters = LMQCS_I,
) -> bool:
    try:
        if signature.parameter_name != par.name or public_key.parameter_name != par.name:
            return False
        if np.count_nonzero(signature.c) != par.omega_c:
            return False
        if np.any((signature.c < -1) | (signature.c > 1)):
            return False
        if norm_inf(signature.s1) > par.gamma or norm_inf(signature.s2) > par.gamma:
            return False

        c_inv = poly_inverse(signature.c, par.q)
        t = symmetric(cyclic_mul(signature.s1, public_key.h, par.q) - signature.s2, par.q)
        recovered_hash_vector = symmetric(cyclic_mul(c_inv, t, par.q), par.q)
        expected_c, _ = hash_to_challenge(message, recovered_hash_vector, public_key.h, par)
        if not np.array_equal(expected_c, signature.c):
            return False

        w1 = symmetric(cyclic_mul(c_inv, signature.s1, par.q), par.q)
        w2 = symmetric(cyclic_mul(c_inv, signature.s2, par.q), par.q)
        if norm_inf(w1) <= par.gamma or norm_inf(w2) <= par.gamma:
            return False
        return True
    except (ValueError, RuntimeError, OverflowError):
        return False