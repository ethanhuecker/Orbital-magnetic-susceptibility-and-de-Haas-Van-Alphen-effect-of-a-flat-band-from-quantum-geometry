# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""qwz_cross.py -- nearly-flat-band crossover extension of flatqwz.py.

Model: H(k) = dhat(k).sigma + delta * g(k) * identity,
  g(k) = 2 - cos kx - cos ky   (>= 0, minimum at Gamma).

Because the scalar term is proportional to the identity in orbital space it
commutes with the flattened QWZ part: at B = 0 the exact lower band is
  E_0(k) = -1 + delta * g(k),
with eigenstates (hence Berry curvature, m_orb, c2, contour tables of
flatqwz) COMPLETELY UNCHANGED.  Under a field the composite semiclassical
symbol of the lower band is
  eps(k; B) = -1 + delta * g(k) + B * m_orb(k) + O(B^2 c2).

Peierls: the scalar channel enters the Harper matrix through the same
(m, n) hopping loop and phase convention as the sigma channels; g(k) is
band-limited (harmonics 0, +-1 only) so its hopping set is exact.
"""

import numpy as np
import flatqwz as fq


def harper_matrix_cross(hops, q, delta, k1=fq.K1_DEFAULT, k2=fq.K2_DEFAULT):
    """2q x 2q magnetic Bloch Hamiltonian of dhat.sigma + delta*g(k)*1.

    Scalar channel added analytically: g(k) = 2 - cos kx - cos ky has
    hoppings  c0_{0,0} = 2, c0_{+-1,0} = c0_{0,+-1} = -1/2,
    inserted with the identical phase convention as harper_matrix.
    """
    H = fq.harper_matrix(hops, q, k1, k2)
    if delta == 0.0:
        return H
    s = np.arange(q)
    # (m, n) = (0, 0): onsite 2*delta
    idx = np.arange(2 * q)
    H[idx, idx] += 2.0 * delta
    # (m, n) = (+-1, 0): coefficient -delta/2, phase exp(1j*k1*c) with
    # c = (s + m - sp)//q, sp = (s + m) % q ; n = 0 so ph = 1.
    for m in (1, -1):
        sp = (s + m) % q
        c = (s + m - sp) // q
        pk = -0.5 * delta * np.exp(1j * k1 * c)
        H[2 * sp, 2 * s] += pk
        H[2 * sp + 1, 2 * s + 1] += pk
    # (m, n) = (0, +-1): coefficient -delta/2, phase exp(1j*n*(2*pi/q*s + k2))
    for n in (1, -1):
        ph = -0.5 * delta * np.exp(1j * n * (2 * np.pi / q * s + k2))
        H[2 * s, 2 * s] += ph
        H[2 * s + 1, 2 * s + 1] += ph
    return H


def harper_spectrum_cross(hops, q, delta, k1=fq.K1_DEFAULT, k2=fq.K2_DEFAULT):
    H = harper_matrix_cross(hops, q, delta, k1, k2)
    herm = float(np.max(np.abs(H - H.conj().T)))
    return np.linalg.eigvalsh(H), herm


def g_native(kx, ky):
    return 2.0 - np.cos(kx) - np.cos(ky)


def composite_symbol(kx, ky, B, delta, u=fq.U):
    """eps(k;B) + 1 = delta*g + B*m_orb (grade-0 + grade-1 of lower band)."""
    return delta * g_native(kx, ky) + B * fq.berry_curvature(kx, ky, u)


def composite_tables(B, delta, Nfine=1024, u=fq.U):
    """Sorted cumulative structures of the composite symbol h = delta*g + B*m_orb.

    Same layout as fq.contour_tables but sorted by h instead of m_orb:
      h_s    : sorted composite values
      cumA   : occupied area A(h) = |{k : h(k) < h}|
      cumPhi : Integral_D m_orb d^2k   (for the O(B) phase correction)
    """
    k = -np.pi + 2 * np.pi * (np.arange(Nfine) + 0.5) / Nfine
    KX, KY = np.meshgrid(k, k, indexing="ij")
    OM = fq.berry_curvature(KX, KY, u)
    HH = delta * g_native(KX, KY) + B * OM
    cell = (2 * np.pi / Nfine) ** 2
    h_flat = HH.ravel()
    order = np.argsort(h_flat, kind="stable")
    h_s = h_flat[order]
    cumA = np.arange(1, h_s.size + 1) * cell
    cumPhi = np.cumsum(OM.ravel()[order]) * cell
    return dict(h_s=h_s, cumA=cumA, cumPhi=cumPhi, cell=cell,
                h_min=float(h_s[0]), h_max=float(h_s[-1]))


def A_of_h(tab, h):
    idx = np.searchsorted(tab["h_s"], h, side="left")
    return 0.0 if idx == 0 else float(tab["cumA"][idx - 1])


def Phi_of_h(tab, h):
    idx = np.searchsorted(tab["h_s"], h, side="left")
    return 0.0 if idx == 0 else float(tab["cumPhi"][idx - 1])


def h_of_A(tab, area):
    idx = np.searchsorted(tab["cumA"], area, side="left")
    idx = min(idx, tab["h_s"].size - 1)
    return float(tab["h_s"][idx])


def composite_levels(B, delta, dmu, gamma=0.5, Nfine=1024, u=fq.U,
                     tab=None, nmax=None):
    """Semiclassical Landau levels of the composite symbol at flux B:
    A(h_n) = 2*pi*B*(n + gamma), returning h_n (energies above -1).
    dmu = mu + 1 sets the highest occupied level for counting."""
    if tab is None:
        tab = composite_tables(B, delta, Nfine, u)
    out = []
    n = 0
    while True:
        area = 2 * np.pi * B * (n + gamma)
        if area >= tab["cumA"][-1]:
            break
        h = h_of_A(tab, area)
        out.append(h)
        n += 1
        if nmax is not None and n >= nmax:
            break
    return np.array(out)
