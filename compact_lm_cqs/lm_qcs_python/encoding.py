from __future__ import annotations

from typing import Iterable
import numpy as np


def pack_values(values: Iterable[int], bits: int) -> bytes:
    out = bytearray(); acc = 0; acc_bits = 0; limit = 1 << bits
    for raw in values:
        value = int(raw)
        if not 0 <= value < limit:
            raise ValueError(f"Value {value} does not fit in {bits} bits.")
        acc |= value << acc_bits; acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF); acc >>= 8; acc_bits -= 8
    if acc_bits: out.append(acc & 0xFF)
    return bytes(out)


def unpack_values(data: bytes, count: int, bits: int) -> np.ndarray:
    out = np.zeros(count, dtype=object if bits > 62 else np.int64)
    acc = 0; acc_bits = 0; di = 0; mask = (1 << bits) - 1
    for i in range(count):
        while acc_bits < bits:
            if di >= len(data): raise ValueError("Truncated data.")
            acc |= data[di] << acc_bits; acc_bits += 8; di += 1
        out[i] = acc & mask; acc >>= bits; acc_bits -= bits
    return out


def pack_fields(fields: list[tuple[Iterable[int], int]]) -> bytes:
    out = bytearray(); acc = 0; acc_bits = 0
    for values, bits in fields:
        limit = 1 << bits
        for raw in values:
            value = int(raw)
            if not 0 <= value < limit: raise ValueError("Field value out of range.")
            acc |= value << acc_bits; acc_bits += bits
            while acc_bits >= 8:
                out.append(acc & 0xFF); acc >>= 8; acc_bits -= 8
    if acc_bits: out.append(acc & 0xFF)
    return bytes(out)


def unpack_fields(data: bytes, specs: list[tuple[int, int]]) -> list[np.ndarray]:
    acc = 0; acc_bits = 0; di = 0; results=[]
    for count, bits in specs:
        vals = np.zeros(count, dtype=object if bits > 62 else np.int64); mask = (1 << bits) - 1
        for i in range(count):
            while acc_bits < bits:
                if di >= len(data): raise ValueError("Truncated data.")
                acc |= data[di] << acc_bits; acc_bits += 8; di += 1
            vals[i] = acc & mask; acc >>= bits; acc_bits -= bits
        results.append(vals)
    return results


# ==============================================================================
# FALCON Variable-Length Gaussian Compression / Decompression (Parameterized)
# ==============================================================================

def compress_falcon(s: Iterable[int], b: int = 7) -> bytes:
    """
    Compresses a polynomial/vector s using FALCON Algorithm 21 into raw bytes:
      - 1 sign bit ('1' for > 0, '0' for <= 0)
      - b LSBs of |s_i|
      - Unary code for MSBs k = floor(|s_i| / 2^b): k zeros followed by '1'
      
    Args:
        s: Iterable of integer coefficients.
        b: Number of raw LSB bits to store uncompressed (default: 7).
        
    Flushes accumulator to byte boundary at the end.
    """
    if b <= 0:
        raise ValueError("Number of LSB bits 'b' must be positive.")

    out = bytearray()
    acc = 0
    acc_bits = 0
    lsb_mask = (1 << b) - 1

    for raw in s:
        s_i = int(raw)

        # 1. Sign bit
        sign_bit = 1 if s_i > 0 else 0
        acc |= (sign_bit << acc_bits)
        acc_bits += 1
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

        # 2. b LSBs of |s_i|
        abs_s = abs(s_i)
        lsb_val = abs_s & lsb_mask
        acc |= (lsb_val << acc_bits)
        acc_bits += b
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

        # 3. Unary coding for MSBs k = floor(|s_i| / 2^b)
        k = abs_s >> b
        acc_bits += k
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

        # Terminating '1'
        acc |= (1 << acc_bits)
        acc_bits += 1
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

    # Flush remaining bits
    if acc_bits > 0:
        out.append(acc & 0xFF)

    return bytes(out)


def decompress_falcon(
    data: bytes, n: int, b: int = 7, byte_offset: int = 0
) -> tuple[np.ndarray, int]:
    """
    Decompresses 'n' coefficients from 'data' starting at 'byte_offset'
    following FALCON Algorithm 22 using b-bit LSB resolution.
    
    Args:
        data: Raw compressed byte array.
        n: Expected number of coefficients to decompress.
        b: Number of raw LSB bits used during compression (default: 7).
        byte_offset: Starting byte offset in `data`.
        
    Returns:
        (reconstructed_vector, bytes_consumed)
    """
    if b <= 0:
        raise ValueError("Number of LSB bits 'b' must be positive.")

    s = np.zeros(n, dtype=np.int64)
    bit_idx = byte_offset * 8
    total_bits = len(data) * 8

    def read_bit(idx: int) -> int:
        if idx >= total_bits:
            raise ValueError("Truncated compressed data stream.")
        return (data[idx // 8] >> (idx % 8)) & 1

    curr_bit = bit_idx
    for i in range(n):
        # 1. Sign bit
        sign_bit = read_bit(curr_bit)
        curr_bit += 1

        # 2. b LSBs
        s_prime = 0
        for j in range(b):
            s_prime |= (read_bit(curr_bit) << j)
            curr_bit += 1

        # 3. Unary MSBs (count consecutive zeros)
        k = 0
        while read_bit(curr_bit) == 0:
            k += 1
            curr_bit += 1
        curr_bit += 1  # Consume terminating '1'

        # Reconstruct coefficient: s_i = sign * (s_prime + k * 2^b)
        sign_multiplier = 1 if sign_bit == 1 else -1
        magnitude = s_prime + (k << b)
        s[i] = sign_multiplier * magnitude

    bits_consumed = curr_bit - bit_idx
    bytes_consumed = (bits_consumed + 7) // 8
    return s, bytes_consumed