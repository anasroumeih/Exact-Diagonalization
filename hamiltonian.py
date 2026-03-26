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
    """
    Implements the two-site flip:
        |10> <-> |01>
    If the two bits are equal, return None.
    """
    bj = (s >> j) & 1
    bjp = (s >> jp) & 1
    if bj == bjp:
        return None
    return s ^ (1 << j) ^ (1 << jp)


class SpinChainHamiltonian:
    """
    Generic Hamiltonian builder acting inside a Basis object.

    Important:
    ----------
    This class does not 'discover' symmetries by itself.
    It assumes the chosen basis is compatible with the Hamiltonian terms you add.

    Examples:
    ---------
    - FullBasis: everything is allowed
    - SzBasis: only use terms preserving total Sz
    - TranslationBasis / MomentumBasis:
        only use translation-invariant sums over sites
    """
    def __init__(self, basis: Basis, dtype=complex):
        self.basis = basis
        self.L = basis.L
        self.dim = basis.dim
        self.dtype = dtype
        self.H = lil_matrix((self.dim, self.dim), dtype=dtype)

    def _add_operator_from_local_action(self, action_fn):
        """
        Generic helper.

        action_fn(s) should return a list of pairs:
            [(s1, amp1), (s2, amp2), ...]

        representing:
            O |s> = sum_n amp_n |s_n>

        Then this method lifts that action from computational basis
        to the chosen Basis object.
        """
        for ket_idx in range(self.dim):
            # Expand basis ket into computational states
            ket_expansion = self.basis.expand_basis_state(ket_idx)

            accum = defaultdict(complex)

            for s, c_in in ket_expansion:
                for s_out, amp in action_fn(s):
                    # Project computational output back into the chosen basis sector
                    for bra_idx, c_out in self.basis.decompose_computational_state(s_out):
                        accum[bra_idx] += amp * c_in * np.conjugate(c_out)

            for bra_idx, val in accum.items():
                if abs(val) > 0:
                    self.H[bra_idx, ket_idx] += val

    # -----------------------
    # Common translation-summed terms
    # -----------------------

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
        """
        Uniform longitudinal field: h * sum_j sigma^z_j
        Preserves:
            - Sz
            - translation (if uniform)
        """
        def action(s: int):
            val = 0.0
            for j in range(self.L):
                val += sigma_z_value(s, j)
            return [(s, h * val)]
        self._add_operator_from_local_action(action)

    def add_x_field(self, g: float):
        """
        Uniform transverse field: g * sum_j sigma^x_j
        Preserves:
            - translation (if uniform)
        Breaks:
            - total Sz
        """
        def action(s: int):
            out = []
            for j in range(self.L):
                out.append((flip_bit(s, j), g))
            return out
        self._add_operator_from_local_action(action)

    def add_xx_yy(self, J: float, periodic: bool = True):
        """
        XY exchange:
            J * sum_j (S^+_j S^-_{j+1} + S^-_j S^+_{j+1})

        In bit language, it swaps 10 <-> 01 on each bond.
        Preserves:
            - total Sz
            - translation (if summed uniformly)
        """
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
        """
        XXZ / Heisenberg:
            Jxy * sum_j (S^+_j S^-_{j+1} + h.c.) + Jz * sum_j sigma^z_j sigma^z_{j+1}

        If Jz is None, set Jz = Jxy.
        """
        if Jz is None:
            Jz = Jxy
        self.add_xx_yy(Jxy, periodic=periodic)
        self.add_zz(Jz, periodic=periodic)

    def build(self) -> csr_matrix:
        return csr_matrix(self.H)