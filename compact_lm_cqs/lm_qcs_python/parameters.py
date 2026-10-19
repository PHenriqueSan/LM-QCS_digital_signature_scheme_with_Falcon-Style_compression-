from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, erfc, log2, sqrt


@dataclass(frozen=True)
class LMQCSParameters:
    name: str
    security_level: int
    n: int
    q: int
    ell_e: int
    ell_u: int
    omega_e_bar: int
    omega_c: int
    t: float
    public_key_bytes_reported: int
    secret_key_bytes_reported: int
    signature_bytes_reported: int

    # FALCON Compression constant (LSB bits)
    falcon_lsb_bits: int = 7

    @property
    def sigma_s(self) -> float:
        return sqrt(
            self.omega_e_bar * self.ell_e * (self.ell_e + 1) / 3
            + self.omega_c * self.ell_u * (self.ell_u + 1) / 3
        )

    @property
    def gamma(self) -> int:
        return ceil(self.t * self.sigma_s)

    @property
    def q_bits(self) -> int:
        return ceil(log2(self.q))

    @property
    def secret_coeff_bits(self) -> int:
        return ceil(log2(2 * self.ell_e + 1))

    @property
    def s_bits(self) -> int:
        """Fixed uncompressed bit-width per response coefficient."""
        return ceil(log2(2 * self.gamma + 1))

    @property
    def challenge_bits(self) -> int:
        return ceil(log2((1 << self.omega_c) * comb(self.n, self.omega_c)))

    @property
    def public_key_bytes(self) -> int:
        return ceil(self.n * self.q_bits / 8)

    @property
    def secret_key_bytes(self) -> int:
        return ceil(2 * self.n * self.secret_coeff_bits / 8)

    @property
    def signature_bytes_uncompressed(self) -> int:
        """Baseline uncompressed signature size (fixed width s_bits)."""
        return ceil((2 * self.n * self.s_bits + self.challenge_bits) / 8)

    @property
    def expected_bits_per_coeff(self) -> float:
        """
        Expected bit length per coefficient under FALCON compression:
        1 sign bit + b LSBs + 1 stop bit + E[k] (where k = floor(|s_i| / 2^b)).
        Approximates E[k] using Gaussian tail probabilities for sigma_s.
        """
        b = self.falcon_lsb_bits
        modulus = 1 << b  # 2^b

        # Tail sum formula E[k] = sum_{m=1}^infty P(|s_i| >= 2^b * m)
        expected_k = 0.0
        for m in range(1, 10):
            bound = modulus * m
            # P(|X| >= bound) for X ~ N(0, sigma_s^2)
            prob = erfc(bound / (self.sigma_s * sqrt(2)))
            expected_k += prob
            if prob < 1e-6:
                break

        # 1 sign bit + b LSBs + 1 stop bit + expected_k
        return (2 + b) + expected_k

    @property
    def expected_signature_bytes_compressed(self) -> int:
        """Expected signature size in bytes using FALCON compression."""
        total_compressed_bits = (
            2 * self.n * self.expected_bits_per_coeff + self.challenge_bits
        )
        return ceil(total_compressed_bits / 8)

    @property
    def signature_bytes(self) -> int:
        """Defaults to expected compressed signature size for testing."""
        return self.expected_signature_bytes_compressed

    def validate_reported_sizes(self) -> None:
        actual_uncompressed = (
            self.public_key_bytes,
            self.secret_key_bytes,
            self.signature_bytes_uncompressed,
        )
        reported = (
            self.public_key_bytes_reported,
            self.secret_key_bytes_reported,
            self.signature_bytes_reported,
        )
        if actual_uncompressed != reported:
            raise ValueError(
                f"Computed uncompressed sizes {actual_uncompressed} differ from paper {reported}."
            )


# Parameter Sets with explicit falcon_lsb_bits configuration
LMQCS_I = LMQCSParameters(
    name="LM-QCS-I", security_level=128, n=691, q=5479,
    ell_e=2, ell_u=28, omega_e_bar=345, omega_c=41, t=10.1,
    public_key_bytes_reported=1123, secret_key_bytes_reported=519,
    signature_bytes_reported=2106,
    falcon_lsb_bits=7,
)

LMQCS_II = LMQCSParameters(
    name="LM-QCS-II", security_level=192, n=1009, q=8623,
    ell_e=3, ell_u=35, omega_e_bar=455, omega_c=61, t=11.0,
    public_key_bytes_reported=1766, secret_key_bytes_reported=757,
    signature_bytes_reported=3076,
    falcon_lsb_bits=7,
)

LMQCS_III = LMQCSParameters(
    name="LM-QCS-III", security_level=256, n=1201, q=12569,
    ell_e=4, ell_u=41, omega_e_bar=705, omega_c=83, t=10.1,
    public_key_bytes_reported=2102, secret_key_bytes_reported=1201,
    signature_bytes_reported=3968,
    falcon_lsb_bits=8,  # Updated to optimal b for Level 256
)

PARAMETER_SETS = {
    128: LMQCS_I,
    192: LMQCS_II,
    256: LMQCS_III,
    1: LMQCS_I,
    2: LMQCS_II,
    3: LMQCS_III,
}

for _p in (LMQCS_I, LMQCS_II, LMQCS_III):
    _p.validate_reported_sizes()