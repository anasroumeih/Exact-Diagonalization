from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

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
    def __init__(self, basis: Basis, dtype=complex):
        self.basis = basis
        self.L = basis.L
        self.dim = basis.dim
        self.dtype = dtype
        self.H = lil_matrix((self.dim, self.dim), dtype=dtype)

    def _add_operator_from_local_action(self, action_fn):
        for ket_idx in range(self.dim):
            ket_expansion = self.basis.expand_basis_state(ket_idx)
            accum = defaultdict(complex)

            for s, c_in in ket_expansion:
                for s_out, amp in action_fn(s):
                    for bra_idx, c_out in self.basis.decompose_computational_state(s_out):
                        accum[bra_idx] += amp * c_in * np.conjugate(c_out)

            for bra_idx, val in accum.items():
                if abs(val) > 0:
                    self.H[bra_idx, ket_idx] += val

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
        return csr_matrix(self.H)
