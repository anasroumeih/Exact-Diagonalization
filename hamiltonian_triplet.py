from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from basis import Basis


def sigma_z_value(s: int, j: int) -> int:
    """Return +1 for spin up (bit 1), -1 for spin down (bit 0)."""
    return 1 if ((s >> j) & 1) else -1


def flip_bit(s: int, j: int) -> int:
    return s ^ (1 << j)


def exchange_bits_if_opposite(s: int, j: int, jp: int) -> int | None:
    """Implements the two-site flip |10> <-> |01|."""
    bj = (s >> j) & 1
    bjp = (s >> jp) & 1
    if bj == bjp:
        return None
    return s ^ (1 << j) ^ (1 << jp)


class SpinChainHamiltonian:
    """
    Same API as the LIL builder, but accumulates COO triplets (row, col, value)
    and converts to CSR only at the end.
    """

    def __init__(self, basis: Basis, dtype=complex):
        self.basis = basis
        self.L = basis.L
        self.dim = basis.dim
        self.dtype = dtype
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[complex] = []

    def _add_operator_from_local_action(self, action_fn):
        for ket_idx in range(self.dim):
            ket_expansion = self.basis.expand_basis_state(ket_idx)

            for s, c_in in ket_expansion:
                for s_out, amp in action_fn(s):
                    for bra_idx, c_out in self.basis.decompose_computational_state(s_out):
                        val = amp * c_in * np.conjugate(c_out)
                        if abs(val) > 0:
                            self.rows.append(bra_idx)
                            self.cols.append(ket_idx)
                            self.vals.append(val)

    def add_zz(self, J: float, periodic: bool = True):
        def action(s: int):
            val = 0.0
            max_j = self.L if periodic else self.L - 1
            for j in range(max_j):
                jp = (j + 1) % self.L
                val += sigma_z_value(s, j) * sigma_z_value(s, jp)
            return [(s, J * val)]

        self._add_operator_from_local_action(action)

    def add_z_field(self, h: float):
        def action(s: int):
            val = 0.0
            for j in range(self.L):
                val += sigma_z_value(s, j)
            return [(s, h * val)]

        self._add_operator_from_local_action(action)

    def add_x_field(self, g: float):
        def action(s: int):
            out = []
            for j in range(self.L):
                out.append((flip_bit(s, j), g))
            return out

        self._add_operator_from_local_action(action)

    def add_xx_yy(self, J: float, periodic: bool = True):
        def action(s: int):
            out = []
            max_j = self.L if periodic else self.L - 1
            for j in range(max_j):
                jp = (j + 1) % self.L
                sp = exchange_bits_if_opposite(s, j, jp)
                if sp is not None:
                    out.append((sp, J))
            return out

        self._add_operator_from_local_action(action)

    def add_heisenberg(self, Jxy: float, Jz: float | None = None, periodic: bool = True):
        if Jz is None:
            Jz = Jxy
        self.add_xx_yy(Jxy, periodic=periodic)
        self.add_zz(Jz, periodic=periodic)

    def build(self) -> csr_matrix:
        if not self.vals:
            return csr_matrix((self.dim, self.dim), dtype=self.dtype)

        H = coo_matrix(
            (np.asarray(self.vals, dtype=self.dtype),
             (np.asarray(self.rows, dtype=np.int32), np.asarray(self.cols, dtype=np.int32))),
            shape=(self.dim, self.dim),
            dtype=self.dtype,
        )
        return H.tocsr()
