# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchft.gf256 import (
    _gf_exp,
    _gf_log,
    gf_inv,
    gf_matrix_inv,
    gf_mul,
    gf_mul_vec,
    gf_pow,
    GF_GENERATOR,
)


class TestGF256(unittest.TestCase):
    def test_exp_log_roundtrip(self) -> None:
        """exp[log[x]] == x for all nonzero x."""
        for x in range(1, 256):
            log_x = _gf_log[x]
            self.assertEqual(
                _gf_exp[log_x], x, f"roundtrip failed for x={x}, log={log_x}"
            )

    def test_mul_identity_zero(self) -> None:
        """a * 1 == a and a * 0 == 0 for all a."""
        for a in range(256):
            self.assertEqual(gf_mul(a, 1), a, f"a*1 != a for a={a}")
            self.assertEqual(gf_mul(a, 0), 0, f"a*0 != 0 for a={a}")
            self.assertEqual(gf_mul(0, a), 0, f"0*a != 0 for a={a}")
            self.assertEqual(gf_mul(1, a), a, f"1*a != a for a={a}")

    def test_mul_commutativity(self) -> None:
        """a * b == b * a for a sample of values."""
        for a in range(0, 256, 17):
            for b in range(0, 256, 13):
                self.assertEqual(
                    gf_mul(a, b), gf_mul(b, a), f"commutativity failed: {a}*{b}"
                )

    def test_inv_all_nonzero(self) -> None:
        """a * inv(a) == 1 for all nonzero a."""
        for a in range(1, 256):
            inv_a = gf_inv(a)
            self.assertEqual(
                gf_mul(a, inv_a), 1, f"a * inv(a) != 1 for a={a}, inv={inv_a}"
            )

    def test_inv_zero_raises(self) -> None:
        """Inverting zero should raise ValueError."""
        with self.assertRaises(ValueError):
            gf_inv(0)

    def test_pow_basic(self) -> None:
        """Test gf_pow for known cases."""
        g = GF_GENERATOR
        # g^0 == 1
        self.assertEqual(gf_pow(g, 0), 1)
        # g^1 == g
        self.assertEqual(gf_pow(g, 1), g)
        # g^8 should match exp table
        self.assertEqual(gf_pow(g, 8), _gf_exp[8])
        # 0^k == 0 for k > 0
        self.assertEqual(gf_pow(0, 5), 0)
        # 0^0 == 1
        self.assertEqual(gf_pow(0, 0), 1)

    def test_mul_vec_matches_scalar(self) -> None:
        """Vectorized multiply matches element-wise scalar for all coefficients."""
        data = torch.arange(256, dtype=torch.uint8)
        for coeff in [0, 1, 2, 37, 128, 255]:
            result = gf_mul_vec(coeff, data)
            for x in range(256):
                expected = gf_mul(coeff, x)
                self.assertEqual(
                    result[x].item(),
                    expected,
                    f"mismatch: coeff={coeff}, x={x}",
                )

    def test_mul_vec_preserves_shape(self) -> None:
        """gf_mul_vec preserves tensor shape."""
        t = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
        result = gf_mul_vec(42, t)
        self.assertEqual(result.shape, t.shape)
        self.assertEqual(result.dtype, torch.uint8)

    def test_matrix_inv_2x2(self) -> None:
        """Invert a known 2x2 Vandermonde matrix and verify M * M^-1 == I."""
        g = GF_GENERATOR
        # Vandermonde for rows j=0,1 and columns i=0,1:
        # row j, col i: g^(i*j)
        matrix = [
            [gf_pow(g, 0 * 0), gf_pow(g, 1 * 0)],  # [1, 1]
            [gf_pow(g, 0 * 1), gf_pow(g, 1 * 1)],  # [1, g]
        ]
        inv = gf_matrix_inv(matrix, 2)

        # Verify M * M^-1 == I
        for i in range(2):
            for j in range(2):
                val = 0
                for k in range(2):
                    val ^= gf_mul(matrix[i][k], inv[k][j])
                expected = 1 if i == j else 0
                self.assertEqual(val, expected, f"M*M^-1 [{i}][{j}] != I")

    def test_matrix_inv_3x3(self) -> None:
        """Invert a 3x3 Vandermonde matrix and verify M * M^-1 == I."""
        g = GF_GENERATOR
        matrix = [
            [gf_pow(g, i * j) for i in range(3)] for j in range(3)
        ]
        inv = gf_matrix_inv(matrix, 3)

        for i in range(3):
            for j in range(3):
                val = 0
                for k in range(3):
                    val ^= gf_mul(matrix[i][k], inv[k][j])
                expected = 1 if i == j else 0
                self.assertEqual(val, expected, f"M*M^-1 [{i}][{j}] != I")

    def test_vandermonde_any_submatrix_invertible(self) -> None:
        """
        Key erasure coding property: any k-row submatrix of the m-row
        Vandermonde matrix (with N columns) is invertible, for small N and m.
        """
        g = GF_GENERATOR
        N = 6  # number of data columns
        m = 4  # number of syndrome rows

        # Full Vandermonde: m rows x N cols, entry [j][i] = g^(i*j)
        vand = [[gf_pow(g, i * j) for i in range(N)] for j in range(m)]

        # For each k in 1..m, pick all C(m,k) subsets of rows and
        # all C(N,k) subsets of columns, and verify the k×k submatrix
        # is invertible.
        from itertools import combinations

        for k in range(1, m + 1):
            for row_subset in combinations(range(m), k):
                for col_subset in combinations(range(N), k):
                    sub = [
                        [vand[r][c] for c in col_subset] for r in row_subset
                    ]
                    try:
                        inv = gf_matrix_inv(sub, k)
                    except ValueError:
                        self.fail(
                            f"Vandermonde submatrix rows={row_subset} "
                            f"cols={col_subset} is singular"
                        )
                    # Verify sub * inv == I
                    for i in range(k):
                        for j in range(k):
                            val = 0
                            for kk in range(k):
                                val ^= gf_mul(sub[i][kk], inv[kk][j])
                            expected = 1 if i == j else 0
                            self.assertEqual(
                                val,
                                expected,
                                f"k={k} rows={row_subset} cols={col_subset} "
                                f"[{i}][{j}] != I",
                            )

    def test_singular_matrix_raises(self) -> None:
        """A matrix with duplicate rows should raise ValueError."""
        matrix = [
            [1, 2],
            [1, 2],  # duplicate row
        ]
        with self.assertRaises(ValueError):
            gf_matrix_inv(matrix, 2)


if __name__ == "__main__":
    unittest.main()
