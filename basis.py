# basis.py
from __future__ import annotations
from abc import ABC, abstractmethod
import cmath
import math


def rotate_left(s: int, L: int, r: int = 1) -> int:
    """Cyclic left shift of an L-bit integer by r sites."""
    r %= L
    mask = (1 << L) - 1
    return ((s << r) & mask) | (s >> (L - r))


def reflect_bits(s: int, L: int) -> int:
    """Bit reflection: site j -> L-1-j."""
    out = 0
    for j in range(L):
        if (s >> j) & 1:
            out |= 1 << (L - 1 - j)
    return out


def popcount(s: int) -> int:
    return s.bit_count()


def translate_orbit(s: int, L: int) -> list[int]:
    """Full translation orbit of state s."""
    orbit = []
    t = s
    while t not in orbit:
        orbit.append(t)
        t = rotate_left(t, L, 1)
    return orbit


def canonical_representative(s: int, L: int) -> tuple[int, int]:
    """
    Return:
        rep   = minimal integer in the translation orbit
        shift = n such that s = T^n rep
    """
    orbit = translate_orbit(s, L)
    rep = min(orbit)
    shift = orbit.index(s)
    return rep, shift


class Basis(ABC):
    """
    Abstract basis API.

    The Hamiltonian builder only needs these two maps:

    1. expand_basis_state(i):
       |basis_i> = sum_{s} c_s |s_computational>

    2. decompose_computational_state(s):
       projection of |s> into this basis sector:
       P_sector |s> = sum_i a_i |basis_i>

    This makes the Hamiltonian builder generic enough to work for:
    - full basis
    - fixed-Sz basis
    - translation k=0 basis
    - momentum basis
    """
    def __init__(self, L: int):
        self.L = L
        self.dim = 0

    @abstractmethod
    def expand_basis_state(self, i: int) -> list[tuple[int, complex]]:
        pass

    @abstractmethod
    def decompose_computational_state(self, s: int) -> list[tuple[int, complex]]:
        pass

    def __len__(self):
        return self.dim


class FullBasis(Basis):
    def __init__(self, L: int):
        super().__init__(L)
        self.states = list(range(2**L))
        self.index = {s: i for i, s in enumerate(self.states)}
        self.dim = len(self.states)

    def expand_basis_state(self, i: int) -> list[tuple[int, complex]]:
        return [(self.states[i], 1.0)]

    def decompose_computational_state(self, s: int) -> list[tuple[int, complex]]:
        idx = self.index.get(s)
        return [] if idx is None else [(idx, 1.0)]


class SzBasis(Basis):
    def __init__(self, L: int, n_up: int):
        super().__init__(L)
        self.n_up = n_up
        self.states = [s for s in range(2**L) if popcount(s) == n_up]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.dim = len(self.states)

    def expand_basis_state(self, i: int) -> list[tuple[int, complex]]:
        return [(self.states[i], 1.0)]

    def decompose_computational_state(self, s: int) -> list[tuple[int, complex]]:
        idx = self.index.get(s)
        return [] if idx is None else [(idx, 1.0)]


class MomentumBasis(Basis):
    """
    Translation-momentum basis.

    Parameters
    ----------
    L : int
        Chain length
    m : int
        Momentum quantum number, k = 2*pi*m/L
    n_up : int | None
        If not None, restrict to fixed magnetization sector.
        If None, use the full Hilbert space.

    Notes
    -----
    Basis states are:
        |rep; k> = (1/sqrt(R)) sum_{n=0}^{R-1} exp(-i k n) T^n |rep>
    where R is the translation orbit length of rep.

    A representative rep is allowed in momentum sector m iff:
        exp(i k R) = 1
    i.e.
        (m * R) % L == 0
    """
    def __init__(self, L: int, m: int, n_up: int | None = None):
        super().__init__(L)
        self.m = m % L
        self.k = 2.0 * math.pi * self.m / L
        self.n_up = n_up

        if n_up is None:
            raw_states = list(range(2**L))
        else:
            raw_states = [s for s in range(2**L) if popcount(s) == n_up]

        # Build unique translation orbits
        rep_data = {}
        state_to_rep_shift = {}

        for s in raw_states:
            rep, shift = canonical_representative(s, L)
            if rep not in rep_data:
                orbit = translate_orbit(rep, L)
                rep_data[rep] = orbit
                for n, t in enumerate(orbit):
                    state_to_rep_shift[t] = (rep, n)

        # Keep only reps compatible with momentum m
        self.reps = []
        self.orbits = {}
        self.orbit_len = {}
        self.rep_to_index = {}
        self.state_to_rep_shift = {}

        for rep, orbit in rep_data.items():
            R = len(orbit)
            if (self.m * R) % L == 0:
                idx = len(self.reps)
                self.reps.append(rep)
                self.orbits[rep] = orbit
                self.orbit_len[rep] = R
                self.rep_to_index[rep] = idx
                for n, t in enumerate(orbit):
                    self.state_to_rep_shift[t] = (rep, n)

        self.dim = len(self.reps)

    def expand_basis_state(self, i: int) -> list[tuple[int, complex]]:
        rep = self.reps[i]
        orbit = self.orbits[rep]
        R = self.orbit_len[rep]
        norm = math.sqrt(R)

        out = []
        for n, s in enumerate(orbit):
            coeff = cmath.exp(-1j * self.k * n) / norm
            out.append((s, coeff))
        return out

    def decompose_computational_state(self, s: int) -> list[tuple[int, complex]]:
        """
        Projection of |s> into this momentum sector:
            P_k |s> = e^{i k r} / sqrt(R) |rep; k>
        where s = T^r rep.
        """
        data = self.state_to_rep_shift.get(s)
        if data is None:
            return []

        rep, shift = data
        idx = self.rep_to_index.get(rep)
        if idx is None:
            return []

        R = self.orbit_len[rep]
        coeff = cmath.exp(1j * self.k * shift) / math.sqrt(R)
        return [(idx, coeff)]


class TranslationBasis(MomentumBasis):
    """
    Translation-invariant basis = momentum k=0 sector.
    This is just MomentumBasis with m=0.
    """
    def __init__(self, L: int, n_up: int | None = None):
        super().__init__(L=L, m=0, n_up=n_up)
