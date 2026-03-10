# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GF(2^8) Arithmetic for Erasure Coding
======================================

Galois Field GF(2^8) arithmetic using the AES irreducible polynomial
x^8 + x^4 + x^3 + x + 1 (0x11B) with primitive element g=3.

Provides scalar operations (multiply, inverse, power) via log/exp tables,
vectorized tensor multiplication for efficient byte-level operations, and
matrix inversion via Gauss-Jordan elimination for Reed-Solomon decoding.
"""

import functools
from typing import List

import torch

# AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1
GF_POLY: int = 0x11B

# Primitive element (generator) of GF(2^8) under 0x11B.
# 3 has order 255 (generates the full multiplicative group).
# Note: 2 has order 51 under 0x11B and is NOT primitive.
GF_GENERATOR: int = 3

# Build log and exp tables at module load time.
# _gf_exp[i] = g^i mod GF_POLY for i in [0, 255)
# _gf_log[x] = i such that g^i = x, for x in [1, 255]
# _gf_log[0] is undefined (set to 0 as sentinel).
_gf_exp: List[int] = [0] * 256
_gf_log: List[int] = [0] * 256


def _gf_mul_no_table(a: int, b: int) -> int:
    """GF(2^8) multiply without tables (used during table construction)."""
    result = 0
    while b > 0:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= GF_POLY
        b >>= 1
    return result


def _build_tables() -> None:
    """Populate the exp and log tables for GF(2^8) using generator g=3."""
    x = 1
    for i in range(255):
        _gf_exp[i] = x
        _gf_log[x] = i
        x = _gf_mul_no_table(x, GF_GENERATOR)
    # g^255 == g^0 == 1 (the group has order 255), so set exp[255] = 1
    # for convenience in modular arithmetic. Do NOT overwrite _gf_log[1].
    _gf_exp[255] = _gf_exp[0]


_build_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two GF(2^8) elements using log/exp tables."""
    if a == 0 or b == 0:
        return 0
    return _gf_exp[(_gf_log[a] + _gf_log[b]) % 255]


def gf_inv(a: int) -> int:
    """Multiplicative inverse of a nonzero GF(2^8) element."""
    if a == 0:
        raise ValueError("Cannot invert zero in GF(2^8)")
    return _gf_exp[(255 - _gf_log[a]) % 255]


def gf_pow(base: int, exp: int) -> int:
    """Raise a GF(2^8) element to a non-negative integer power."""
    if base == 0:
        return 0 if exp > 0 else 1
    return _gf_exp[(_gf_log[base] * exp) % 255]


@functools.lru_cache(maxsize=256)
def _mul_table(coeff: int) -> torch.Tensor:
    """
    Build a 256-entry uint8 lookup table: table[x] = gf_mul(coeff, x).
    Cached per coefficient value.
    """
    table = torch.zeros(256, dtype=torch.uint8)
    for x in range(256):
        table[x] = gf_mul(coeff, x)
    return table


def gf_mul_vec(coeff: int, t: torch.Tensor) -> torch.Tensor:
    """
    Vectorized GF(2^8) multiplication: multiply every byte in tensor ``t``
    by scalar ``coeff``.

    Uses a precomputed 256-entry lookup table for the coefficient, then
    indexes into it with the tensor values.

    Args:
        coeff: GF(2^8) scalar (0..255).
        t: uint8 tensor of arbitrary shape.

    Returns:
        uint8 tensor of same shape, each byte multiplied by coeff in GF(2^8).
    """
    if coeff == 0:
        return torch.zeros_like(t)
    if coeff == 1:
        return t.clone()
    table = _mul_table(coeff).to(t.device)
    return table[t.long()]


def gf_matrix_inv(matrix: List[List[int]], size: int) -> List[List[int]]:
    """
    Invert a ``size x size`` matrix over GF(2^8) using Gauss-Jordan elimination.

    Args:
        matrix: 2D list of GF(2^8) elements (will not be modified).
        size: dimension of the square matrix.

    Returns:
        The inverse matrix as a 2D list of GF(2^8) elements.

    Raises:
        ValueError: if the matrix is singular.
    """
    # Augment with identity: [M | I]
    aug = [[0] * (2 * size) for _ in range(size)]
    for i in range(size):
        for j in range(size):
            aug[i][j] = matrix[i][j]
        aug[i][size + i] = 1

    # Forward elimination with partial pivoting
    for col in range(size):
        # Find pivot row
        pivot = -1
        for row in range(col, size):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            raise ValueError("Singular matrix in GF(2^8)")

        # Swap rows
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        # Scale pivot row so that aug[col][col] == 1
        inv_diag = gf_inv(aug[col][col])
        for j in range(2 * size):
            aug[col][j] = gf_mul(aug[col][j], inv_diag)

        # Eliminate column in all other rows
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(2 * size):
                aug[row][j] ^= gf_mul(factor, aug[col][j])

    # Extract the inverse from the right half
    result = [[aug[i][size + j] for j in range(size)] for i in range(size)]
    return result
