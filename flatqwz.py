# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""flatqwz.py -- library for the flattened-QWZ dHvA production pipeline (Part III).

Model: flattened QWZ, H(k) = dhat(k) . sigma,
  d = (sin kx, sin ky, u + cos kx + cos ky), u = -1.2.
Flat bands at +-1; the lower band (eps0 = -1) is the subject.
Units: e = hbar = a = 1.  Flux phi = 1/q per plaquette, B = 2*pi*phi = 2*pi/q.

Field-induced dispersion of the lower flat band:
  eps(k) = -1 + B * m_orb(k) + B^2 * c2(k) + ...
with m_orb(k) equal (numerically, for this model) to the lower-band Berry
curvature; closed form in `berry_curvature` (alias `m_orb`).

Legacy helper default (the published waveform instead uses mu=-1): mu(B) = -1 + m0*B, m0 = -1/2.

Phase-offset conventions (locked):
  gamma0 = 1/2
  gamma1 = -(1/2 pi) * Integral_D m_orb d^2k        (D = occupied region)
  gamma2 = (1/2 pi) * d/dm0 [ Integral_D c2 d^2k ]  (windowed slope, window 0.09)
Quantization: A(m_n) = 2*pi*B*(n + gamma0 + gamma1 + gamma2).

All numerical machinery below is ported from the validated Part-III scratch
sources (ed_sweep.py, compute_ingredients.py, verify_eps2.py, phase_refine.py,
finiteT protocol); do not "improve" the recipes without re-validating.
"""

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

# ---------------------------------------------------------------- constants
U = -1.2
K1_DEFAULT, K2_DEFAULT = 0.51, 1.13
N_FFT_DEFAULT = 1024
R_TRUNC_DEFAULT = 30

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ==================================================================
#  Hopping construction (N = 1024 FFT, r = 24, tail bound)
# ==================================================================
def build_hoppings(N=N_FFT_DEFAULT, r=R_TRUNC_DEFAULT, u=U):
    """Real-space hoppings t_{mn} = cx*sx + cy*sy + cz*sz of the flattened QWZ
    Hamiltonian, from an N x N FFT of dhat(k).sigma.

    Returns dict with coefficient blocks cxs, cys, czs of shape (2r+1, 2r+1)
    indexed [m+r, n+r], and the operator-norm tail bound
        T(r) = sum_{max(|m|,|n|) > r} (|cx|+|cy|+|cz|)_{mn}
    over the full N x N coefficient set (centered indices).
    """
    k = 2 * np.pi * np.arange(N) / N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    ad = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2
                 + (u + np.cos(KX) + np.cos(KY)) ** 2)
    cx = np.fft.ifft2(np.sin(KX) / ad)
    cy = np.fft.ifft2(np.sin(KY) / ad)
    cz = np.fft.ifft2((u + np.cos(KX) + np.cos(KY)) / ad)

    # tail bound over dropped coefficients (centered indices in [-N/2, N/2))
    w = np.abs(cx) + np.abs(cy) + np.abs(cz)
    wsh = np.fft.fftshift(w)
    idx = np.arange(-N // 2, N // 2)
    MM, NN = np.meshgrid(idx, idx, indexing="ij")
    kept = (np.abs(MM) <= r) & (np.abs(NN) <= r)
    tail = float(wsh[~kept].sum())

    ms = np.arange(-r, r + 1)
    cxs = cx[np.ix_(ms % N, ms % N)]
    cys = cy[np.ix_(ms % N, ms % N)]
    czs = cz[np.ix_(ms % N, ms % N)]
    return dict(r=r, N=N, cxs=cxs, cys=cys, czs=czs, tail=tail)


# ==================================================================
#  Harper matrix at phi = 1/q (straight-line Peierls, exact magnetic
#  Bloch folding); validated recipe, phases identical to ed_sweep.py:
#      phase(m, n; s) = exp(i*(2*pi/q * n * (s + m/2) + k2*n + k1*c))
# ==================================================================
def harper_matrix(hops, q, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """2q x 2q magnetic Bloch Hamiltonian at flux phi = 1/q.

    Vectorized over the inner n-sum (matmul); phase convention identical to
    the validated per-(m, n) loop.
    """
    r = hops["r"]
    cxs, cys, czs = hops["cxs"], hops["cys"], hops["czs"]
    ns = np.arange(-r, r + 1)
    s = np.arange(q)
    H = np.zeros((2 * q, 2 * q), dtype=complex)
    for mi, m in enumerate(ns):
        sp = (s + m) % q
        c = (s + m - sp) // q
        # ph[j, i] = exp(1j * n_j * (2*pi/q*(s_i + m/2) + k2))
        ph = np.exp(1j * np.outer(ns, 2 * np.pi / q * (s + m / 2) + k2))
        # Accelerate BLAS raises spurious FP flags on complex matmul; the
        # output is validated to 7e-15 against the reference per-(m,n) loop.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            fx = cxs[mi] @ ph
            fy = cys[mi] @ ph
            fz = czs[mi] @ ph
        pk = np.exp(1j * k1 * c)
        H[2 * sp, 2 * s] += fz * pk
        H[2 * sp, 2 * s + 1] += (fx - 1j * fy) * pk
        H[2 * sp + 1, 2 * s] += (fx + 1j * fy) * pk
        H[2 * sp + 1, 2 * s + 1] += -fz * pk
    return H


def harper_spectrum(hops, q, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """Eigenvalues (ascending) and hermiticity error of the Harper matrix."""
    H = harper_matrix(hops, q, k1, k2)
    herm = float(np.max(np.abs(H - H.conj().T)))
    return np.linalg.eigvalsh(H), herm


# ==================================================================
#  Closed-form Berry curvature of the lower band  (= m_orb numerically)
# ==================================================================
def berry_curvature(kx, ky, u=U):
    """Omega_-(k) closed form; anchors Omega(Gamma) = -2, Omega(X) = -2/9."""
    sx, sy = np.sin(kx), np.sin(ky)
    cx, cy = np.cos(kx), np.cos(ky)
    dz = u + cx + cy
    d = np.sqrt(sx ** 2 + sy ** 2 + dz ** 2)
    num = sx ** 2 * cy + cx * sy ** 2 + dz * cx * cy
    return -num / (2.0 * d ** 3)


m_orb = berry_curvature  # physical name for the O(B) dispersion coefficient


# ==================================================================
#  c2(k) evaluator -- ported verbatim from verify_eps2.py
#  (4th-order 5-point stencils, h = 1e-3, parallel-transport gauge,
#   c2 = Re eps2_19 at B = 1)
# ==================================================================
_OFF = [-2, -1, 0, 1, 2]
_W1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0     # d/dk / h
_W2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0  # d2/dk2 / h^2


def _H_qwz(k, u=U):
    kx, ky = k
    d = np.array([np.sin(kx), np.sin(ky), u + np.cos(kx) + np.cos(ky)])
    d = d / np.linalg.norm(d)
    return d[0] * SX + d[1] * SY + d[2] * SZ


def _flat_vec(k, u=U):
    w, V = np.linalg.eigh(_H_qwz(k, u))
    i = int(np.argmin(np.abs(w - (-1.0))))  # lower flat band, eps0 = -1
    return w[i], V[:, i], w, V, i


def _align(v, vref):
    ov = vref.conj() @ v
    return v * (ov.conjugate() / abs(ov))


def c2_flat(kx, ky, h=1e-3, u=U, B=1.0):
    """c2(k) = Re eps2_19 at B = 1 for the lower flat band (eq. 19 machinery).

    eps2_19 = -(B^2/2) M0 Om - (B^2/4) T2 + (B^2/8) T3 - (B^2/4) T4
    with all derivatives from 5-point stencils in the parallel-transport
    (aligned) gauge; ported from verify_eps2.full_quantities.
    """
    k0 = np.array([kx, ky], float)
    e0, vref, w, V, iflat = _flat_vec(k0, u)
    H0 = _H_qwz(k0, u)
    n = H0.shape[0]

    Vg, Hg = {}, {}
    for a, oa in enumerate(_OFF):
        for b, ob in enumerate(_OFF):
            kk = k0 + np.array([oa * h, ob * h])
            Hg[(a, b)] = _H_qwz(kk, u)
            _, v, _, _, _ = _flat_vec(kk, u)
            Vg[(a, b)] = _align(v, vref)

    dxv = sum(_W1[a] * Vg[(a, 2)] for a in range(5)) / h
    dyv = sum(_W1[b] * Vg[(2, b)] for b in range(5)) / h
    dxxv = sum(_W2[a] * Vg[(a, 2)] for a in range(5)) / h ** 2
    dyyv = sum(_W2[b] * Vg[(2, b)] for b in range(5)) / h ** 2
    dxyv = sum(_W1[a] * _W1[b] * Vg[(a, b)]
               for a in range(5) for b in range(5)) / h ** 2

    Hx = sum(_W1[a] * Hg[(a, 2)] for a in range(5)) / h
    Hy = sum(_W1[b] * Hg[(2, b)] for b in range(5)) / h
    Hxx = sum(_W2[a] * Hg[(a, 2)] for a in range(5)) / h ** 2
    Hyy = sum(_W2[b] * Hg[(2, b)] for b in range(5)) / h ** 2
    Hxy = sum(_W1[a] * _W1[b] * Hg[(a, b)]
              for a in range(5) for b in range(5)) / h ** 2

    dv = {0: dxv, 1: dyv}
    ddv = {(0, 0): dxxv, (0, 1): dxyv, (1, 0): dxyv, (1, 1): dyyv}
    dH = {0: Hx, 1: Hy}
    ddH = {(0, 0): Hxx, (0, 1): Hxy, (1, 0): Hxy, (1, 1): Hyy}
    Hm = H0 - e0 * np.eye(n)

    Om = float((1j * (dv[0].conj() @ dv[1] - dv[1].conj() @ dv[0])).real)
    M0 = float((-0.5j * (dv[0].conj() @ (Hm @ dv[1])
                         - dv[1].conj() @ (Hm @ dv[0]))).real)

    combos = [(0, 1, 0, 1, 1.0), (0, 1, 1, 0, -1.0),
              (1, 0, 0, 1, -1.0), (1, 0, 1, 0, 1.0)]
    T2 = 0.0 + 0j
    T3 = 0.0 + 0j
    T4 = 0.0 + 0j
    others = [mm for mm in range(len(w)) if mm != iflat]
    for (i, j, kk_, l, sgn) in combos:
        T2 += sgn * (dv[i].conj() @ (ddH[(j, kk_)] @ dv[l]))
        T3 += sgn * (ddv[(kk_, i)].conj() @ (Hm @ ddv[(j, l)]))
        for mm in others:
            um = V[:, mm]
            T4 += sgn * ((dv[l].conj() @ (dH[kk_] @ um))
                         * (um.conj() @ (dH[i] @ dv[j])) / (w[mm] - e0))

    eps2_19 = -(B ** 2 / 2) * M0 * Om - (B ** 2 / 4) * T2 \
        + (B ** 2 / 8) * T3 - (B ** 2 / 4) * T4
    return float(eps2_19.real)


def c2_grid(Ng=96, h=1e-3, u=U):
    """c2 on the offset grid k_j = -pi + 2*pi*(j + 1/2)/Ng (matches c2grid96)."""
    kg = -np.pi + 2 * np.pi * (np.arange(Ng) + 0.5) / Ng
    out = np.empty((Ng, Ng))
    for i in range(Ng):
        for j in range(Ng):
            out[i, j] = c2_flat(kg[i], kg[j], h=h, u=u)
    return out


# ==================================================================
#  Contour geometry:  A(m), Phi(m), dm_orb/dA, gamma1(m), gamma2(m)
# ==================================================================
def bilinear_periodic(grid, kxf, kyf, Ng):
    """Periodic bilinear interpolation; grid on k_j = -pi + 2*pi*(j+0.5)/Ng."""
    x = (kxf + np.pi) / (2 * np.pi) * Ng - 0.5
    y = (kyf + np.pi) / (2 * np.pi) * Ng - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    x0m = np.mod(x0, Ng)
    x1m = np.mod(x0 + 1, Ng)
    y0m = np.mod(y0, Ng)
    y1m = np.mod(y0 + 1, Ng)
    return (grid[x0m, y0m] * (1 - fx) * (1 - fy)
            + grid[x1m, y0m] * fx * (1 - fy)
            + grid[x0m, y1m] * (1 - fx) * fy
            + grid[x1m, y1m] * fx * fy)


def contour_tables(Nfine=1024, c2grid=None, Ng_c2=96, u=U):
    """Sorted cumulative structures on a fine BZ grid.

    Returns dict with:
      om_s   : sorted m_orb values (fine grid)
      cumA   : A(m) just above om_s[i]  (occupied area, D = {m_orb < m})
      cumPhi : Integral_D m_orb d^2k
      cumI   : Integral_D c2 d^2k  (if c2grid given)
      cell   : fine-grid cell area
    """
    k = -np.pi + 2 * np.pi * (np.arange(Nfine) + 0.5) / Nfine
    KX, KY = np.meshgrid(k, k, indexing="ij")
    OM = berry_curvature(KX, KY, u)
    cell = (2 * np.pi / Nfine) ** 2

    om_flat = OM.ravel()
    order = np.argsort(om_flat, kind="stable")
    om_s = om_flat[order]
    cumA = np.arange(1, om_s.size + 1) * cell
    cumPhi = np.cumsum(om_s) * cell

    tab = dict(om_s=om_s, cumA=cumA, cumPhi=cumPhi, cell=cell,
               om_min=float(om_s[0]), om_max=float(om_s[-1]))
    if c2grid is not None:
        C2 = bilinear_periodic(c2grid, KX, KY, Ng_c2)
        tab["cumI"] = np.cumsum(C2.ravel()[order]) * cell
    return tab


def _cum_at(tab, key, m):
    idx = np.searchsorted(tab["om_s"], m, side="left")
    if idx == 0:
        return 0.0
    return float(tab[key][idx - 1])


def A_of_m(tab, m):
    """Occupied area A(m) = |{k : m_orb(k) < m}|."""
    return _cum_at(tab, "cumA", m)


def Phi_of_m(tab, m):
    """Integral of m_orb over the occupied region D."""
    return _cum_at(tab, "cumPhi", m)


def I_of_m(tab, m):
    """Integral of c2 over the occupied region D."""
    return _cum_at(tab, "cumI", m)


def m_of_A(tab, area):
    """Inverse of A(m): contour level enclosing the given area."""
    idx = np.searchsorted(tab["cumA"], area, side="left")
    idx = min(idx, tab["om_s"].size - 1)
    return float(tab["om_s"][idx])


def windowed_slope(tab, key, m0, window=0.09):
    """Slope d(cum)/dm at m0 from an LSQ linear fit over [m0-w/2, m0+w/2]."""
    half = window / 2.0
    om_s = tab["om_s"]
    i0 = np.searchsorted(om_s, m0 - half, side="left")
    i1 = np.searchsorted(om_s, m0 + half, side="left")
    x = om_s[i0:i1]
    y = tab[key][i0:i1]
    xm = x - x.mean()
    return float(np.dot(xm, y - y.mean()) / np.dot(xm, xm))


def dm_orb_dA(tab, m0, window=0.09):
    """dm_orb/dA on the contour at level m0 (inverse of windowed dA/dm).

    dA/dm is convex near m0 = -1/2 (dm_orb/dA varies ~25% over +-0.06), so a
    plain windowed LINEAR slope is biased by the window average; fit A(m)
    with a QUADRATIC over the window and take the derivative at the center
    (unbiased to O(w^3)).  Stable to <0.5% for window in [0.03, 0.09].
    """
    half = window / 2.0
    om_s = tab["om_s"]
    i0 = np.searchsorted(om_s, m0 - half, side="left")
    i1 = np.searchsorted(om_s, m0 + half, side="left")
    mm = om_s[i0:i1] - m0
    A = tab["cumA"][i0:i1]
    M = np.column_stack([np.ones_like(mm), mm, mm ** 2])
    coef, *_ = np.linalg.lstsq(M, A, rcond=None)
    return float(1.0 / coef[1])


def gamma1_of_m(tab, m0):
    """gamma1 = -(1/2 pi) * Integral_D m_orb d^2k."""
    return -Phi_of_m(tab, m0) / (2 * np.pi)


def gamma2_of_m(tab, m0, window=0.09):
    """gamma2 = (1/2 pi) * d/dm0 Integral_D c2 d^2k (windowed slope)."""
    return windowed_slope(tab, "cumI", m0, window) / (2 * np.pi)


# ==================================================================
#  Scaled-mu grand potential per site at mu(B) = -1 + m0*B
# ==================================================================
def grand_potential_per_site(evals, q, m0=-0.5, T=0.0):
    """Omega_gp per site from a single-(k1,k2) spectrum at flux 1/q.

    T = 0:      (1/q) sum_{e < mu} (e - mu)
    T > 0:      -(T/q) sum_all logaddexp(0, -(e - mu)/T)
    """
    e = np.asarray(evals, float)
    e = e[~np.isnan(e)]
    mu = -1.0 + 2 * np.pi * m0 / q
    if T <= 0:
        occ = e < mu
        return float(np.sum(e[occ] - mu) / q)
    return float(-(T / q) * np.sum(np.logaddexp(0.0, -(e - mu) / T)))


def grand_potential_sweep(evals_pad, qs, m0=-0.5, T=0.0):
    """Vector of Omega_gp per site over a q sweep (rows of padded evals)."""
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        out[i] = grand_potential_per_site(evals_pad[i], float(q), m0, T)
    return out


# ==================================================================
#  Detrending: RESCALED basis (x0/x)^p, p = 0..5, with x = 2*pi/B.
#  (The raw {1, 1/x, ..., 1/x^5} basis is ill-conditioned, cond ~ 5e15.)
# ==================================================================
def detrend_rescaled(x, y, pmax=5, x0=None):
    """LSQ fit of y on span{(x0/x)^p} and residual.  Returns (resid, cond)."""
    x = np.asarray(x, float)
    if x0 is None:
        x0 = np.median(x)
    uu = x0 / x
    basis = np.vstack([uu ** p for p in range(pmax + 1)]).T
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    cond = float(np.linalg.cond(basis))
    return y - basis @ coef, cond


def scaled_residual(x, y, pmax=5):
    """R = (2*pi/B)^2 * residual = x^2 * residual with x = 2*pi/B."""
    resid, cond = detrend_rescaled(x, y, pmax)
    return x ** 2 * resid, cond


# ==================================================================
#  Fundamental cos/sin extraction
# ==================================================================
def cos_sin_fit(x, R, F):
    """LSQ a*cos(2 pi F x) + b*sin(2 pi F x).

    Returns (a, b, amp, theta) with R ~ amp*cos(2 pi F x - 2 pi theta),
    theta = atan2(b, a)/(2 pi) mod 1.
    """
    M = np.column_stack([np.cos(2 * np.pi * F * x), np.sin(2 * np.pi * F * x)])
    (a, b), *_ = np.linalg.lstsq(M, R, rcond=None)
    amp = float(np.hypot(a, b))
    theta = float((np.arctan2(b, a) / (2 * np.pi)) % 1.0)
    return float(a), float(b), amp, theta


def periodogram_peak(Rw, npad=1 << 18, dx=1.0):
    """Zero-padded periodogram peak frequency of an evenly sampled series."""
    R0 = Rw - Rw.mean()
    P = np.abs(np.fft.rfft(R0, npad)) ** 2
    fr = np.fft.rfftfreq(npad, d=dx)
    ipk = int(np.argmax(P[1:]) + 1)
    return float(fr[ipk]), P, fr


# ==================================================================
#  Spline + lowpass background for amplitudes (ported from phase_refine.py)
# ==================================================================
def spline_lowpass_background(x, y, F, knot_lo=75.0, knot_hi=306.0,
                              knot_step=15.0):
    """Cubic LSQ spline in t = 1/x fit to a one-period moving-average lowpass.

    x is 2*pi/B (uniformly spaced flux denominators); F the fundamental
    frequency per unit x.  Knots every `knot_step` in x.
    """
    period = 1.0 / F
    halfw = period / 2.0
    y_lp = np.empty_like(y)
    for i, xi in enumerate(x):
        m = (x >= xi - halfw) & (x <= xi + halfw)
        y_lp[i] = y[m].mean()
    t_all = 1.0 / x
    order = np.argsort(t_all)
    ts, ys = t_all[order], y_lp[order]
    knots_x = np.arange(knot_lo, knot_hi, knot_step)
    knots_t = np.sort(1.0 / knots_x)
    knots_t = knots_t[(knots_t > ts[0]) & (knots_t < ts[-1])]
    spl = LSQUnivariateSpline(ts, ys, knots_t, k=3)
    return spl(1.0 / x)


# ==================================================================
#  Joint background + harmonic-comb windowed fit (primary extractor)
#
#  Design: rescaled polynomial (x0/x)^p, p = 0..pmax (x0 = window mean),
#  PLUS cos(2*pi*k*nu*x)/x and sin(2*pi*k*nu*x)/x for k = 1..kmax, all in
#  a SINGLE lstsq.  The 1/x scaling of the comb columns matches the
#  1/q envelope of the R-units fundamental, so the fitted coefficient is
#  an LSQ average of x * (local R amplitude) over the window (compare it
#  against the identically fitted theory envelope, not a point value).
#  Validated 2026-07-16 on the production spectra: theta stable to
#  ~1.7e-3 across windows [80,180]..[220,320] (vs 0.017 for the
#  sequential detrend-then-fit), resid_rms/amp <= 2e-3.
# ==================================================================
def joint_comb_design(x, nu, pmax=5, kmax=4, x0=None):
    """Design matrix of the joint fit; returns (M, x0).

    Columns: (x0/x)^p for p = 0..pmax, then cos(2*pi*k*nu*x)/x for
    k = 1..kmax, then sin(2*pi*k*nu*x)/x for k = 1..kmax.
    """
    x = np.asarray(x, float)
    if x0 is None:
        x0 = float(x.mean())
    uu = x0 / x
    cols = [uu ** p for p in range(pmax + 1)]
    for k in range(1, kmax + 1):
        cols.append(np.cos(2 * np.pi * k * nu * x) / x)
    for k in range(1, kmax + 1):
        cols.append(np.sin(2 * np.pi * k * nu * x) / x)
    return np.vstack(cols).T, x0


def joint_comb_fit(x, y, nu, pmax=5, kmax=4):
    """Single-lstsq joint background+comb fit of y(x) on a window.

    Returns dict with:
      coef       : all pmax+1+2*kmax coefficients
      background : polynomial part of the model (same length as y)
      resid      : y - full model
      resid_rms  : rms of resid
      cond       : condition number of the design
      amp        : fundamental coefficient hypot(c_cos1, c_sin1); the local
                   R amplitude at x is amp/x (comb columns carry 1/x)
      theta      : fundamental ~ (amp/x) * cos(2*pi*(nu*x - theta)), mod 1
    """
    y = np.asarray(y, float)
    M, x0 = joint_comb_design(x, nu, pmax, kmax)
    # Accelerate BLAS raises spurious FP flags on some matmuls (see
    # harper_matrix); outputs are validated.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        resid = y - M @ coef
        bg = M[:, :pmax + 1] @ coef[:pmax + 1]
    c1 = float(coef[pmax + 1])
    s1 = float(coef[pmax + 1 + kmax])
    return dict(
        coef=coef, background=bg, resid=resid,
        resid_rms=float(np.sqrt(np.mean(resid ** 2))),
        cond=float(np.linalg.cond(M)),
        amp=float(np.hypot(c1, s1)),
        theta=float((np.arctan2(s1, c1) / (2 * np.pi)) % 1.0),
        x0=x0, pmax=pmax, kmax=kmax,
    )


# ==================================================================
#  Synthetic ideal spectra:  A(m_n) = 2*pi*B*(n + gamma)
# ==================================================================
def synthetic_levels(tab, q, gamma, m_max=-0.30):
    """Ideal Landau levels m_n solving A(m_n) = 2*pi*B*(n + gamma), n = 0,1,...

    Returns the array of m_n with m_n <= m_max (upper cutoff well above the
    committed m0 = -1/2 so grand-potential sums are unaffected).
    """
    B = 2 * np.pi / q
    A_max = A_of_m(tab, m_max)
    n_max = int(np.floor(A_max / (2 * np.pi * B) - gamma))
    n = np.arange(0, n_max + 1)
    targets = 2 * np.pi * B * (n + gamma)
    idx = np.searchsorted(tab["cumA"], targets, side="left")
    idx = np.minimum(idx, tab["om_s"].size - 1)
    return tab["om_s"][idx]


def synthetic_grand_potential(tab, qs, gamma, m0=-0.5, m_max=-0.30):
    """Omega_gp per site of the ideal spectrum (each level weight 1/q)."""
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        q = float(q)
        B = 2 * np.pi / q
        mn = synthetic_levels(tab, q, gamma, m_max)
        occ = mn < m0
        out[i] = (B / q) * np.sum(mn[occ] - m0)
    return out


# ==================================================================
#  Smooth-area ideal ladder (calibration companion of joint_comb_fit)
#
#  The staircase area table A(m) carries grid-counting noise that does
#  NOT converge away at reachable Nfine (theta_syn scatter ~0.01-0.03
#  per 100-wide window at Nfine = 1024..4096, both with level snapping
#  and with linear interpolation).  A(m) is analytic on m in (-2, -2/9)
#  (only critical values: band minimum at Gamma, saddle at X = -2/9,
#  max at M), so on the working range m <= -0.295 a global Chebyshev
#  smooth of the table is exact up to grid noise.  Validated 2026-07-16:
#  per-window theta_syn scatter drops to ~2e-4, d(theta_syn)/d(gamma)
#  = 1.00 +- 0.06, results identical for Nfine = 2048 vs 4096 and
#  shifted < 5e-4 for deg = 15 vs 21.
# ==================================================================
def smooth_area_ladder(tab, m_lo=-1.999, m_hi=-0.295, deg=15, ndense=20000):
    """Chebyshev-smoothed A(m) on [m_lo, m_hi], densely resampled.

    Returns dict with m_dense, A_dense (monotone), and fitdev = max
    absolute deviation of the smooth from the table (grid-noise scale).
    """
    om_s, cumA = tab["om_s"], tab["cumA"]
    i0 = np.searchsorted(om_s, m_lo)
    i1 = np.searchsorted(om_s, m_hi)
    step = max(1, (i1 - i0) // 200000)
    mm, AA = om_s[i0:i1:step], tab["cumA"][i0:i1:step]
    cf = np.polynomial.chebyshev.Chebyshev.fit(mm, AA, deg)
    fitdev = float(np.max(np.abs(cf(mm) - AA)))
    m_dense = np.linspace(mm[0], mm[-1], ndense)
    A_dense = cf(m_dense)
    if not np.all(np.diff(A_dense) > 0):
        raise ValueError("smoothed A(m) is not monotone")
    return dict(m_dense=m_dense, A_dense=A_dense, fitdev=fitdev,
                deg=deg, m_lo=m_lo, m_hi=m_hi)


def synthetic_grand_potential_smooth(lad, qs, gamma, m0=-0.5):
    """Omega_gp per site of the ideal ladder on the smoothed A(m).

    Levels m_n solve A(m_n) = 2*pi*B*(n + gamma) by interpolation on the
    dense smooth curve (upper cutoff lad["m_hi"], well above m0 = -1/2).
    """
    m_dense, A_dense = lad["m_dense"], lad["A_dense"]
    A_max = float(A_dense[-1])
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        q = float(q)
        B = 2 * np.pi / q
        n_max = int(np.floor(A_max / (2 * np.pi * B) - gamma))
        n = np.arange(0, n_max + 1)
        mn = np.interp(2 * np.pi * B * (n + gamma), A_dense, m_dense)
        occ = mn < m0
        out[i] = (B / q) * np.sum(mn[occ] - m0)
    return out


# ==================================================================
#  Stable first-harmonic estimator (finite-T damping):
#  Hann-weighted local fit at F, width 60, with quadratic BASELINE and
#  quadratic amplitude ENVELOPE; D(T) via phase-coherent projection onto
#  the T = 0 phasor.
#
#  The baseline columns keep smooth background-mismatch leakage out of the
#  phasor (an injected pure tone of known amplitude is recovered to 0.6%,
#  where a bare cos/sin fit on the same residual is biased by -11%).
#  The envelope columns return the phasor AT THE WINDOW CENTER, removing
#  the window-averaging (convexity) bias when the amplitude varies strongly
#  across the window -- essential at strong damping, X ~ 4, where
#  D(X(q)) changes by an order of magnitude over the 60-wide window.
# ==================================================================
def local_phasor(x, R, xc, F, width=60.0):
    """Hann-weighted local fit -> fundamental phasor (a, b) at x = xc.

    Design: [1, dx, dx^2] baseline + [1, dx, dx^2]*cos + [1, dx, dx^2]*sin.
    """
    m = np.abs(x - xc) <= width / 2.0
    xw, y = x[m], R[m]
    dx = xw - xc
    wts = 0.5 * (1.0 + np.cos(2 * np.pi * dx / width))
    C = np.cos(2 * np.pi * F * xw)
    S = np.sin(2 * np.pi * F * xw)
    cols = np.column_stack([np.ones_like(dx), dx, dx ** 2,
                            C, dx * C, dx ** 2 * C,
                            S, dx * S, dx ** 2 * S])
    sw = np.sqrt(wts)
    coef, *_ = np.linalg.lstsq(cols * sw[:, None], y * sw, rcond=None)
    return float(coef[3]), float(coef[6])


def phase_coherent_ratio(ab_T, ab_0):
    """D = Re(z_T conj(z_0)) / |z_0|^2 (projection onto the T = 0 phasor)."""
    aT, bT = ab_T
    a0, b0 = ab_0
    return float((aT * a0 + bT * b0) / (a0 ** 2 + b0 ** 2))


# ==================================================================
#  Small helpers
# ==================================================================
def wrap_unit(x):
    """Wrap to (-1/2, 1/2]."""
    return (x + 0.5) % 1.0 - 0.5


def hbar_omega_eff(B, dmdA):
    """Effective cyclotron quantum: hbar*omega_eff = 2*pi*(dm_orb/dA)*B^2."""
    return 2 * np.pi * dmdA * B ** 2


# ==================================================================
#  Parked chemical potential (mu = -1 exactly): island staircase
# ==================================================================
def parked_count(evals_pad, qs, mu=-1.0):
    """Integer count of levels below mu per row of a padded sweep."""
    n = np.empty(len(qs), dtype=int)
    for i in range(len(qs)):
        e = evals_pad[i]
        n[i] = int(np.sum(e[np.isfinite(e)] < mu))
    return n


def pocket_geometry(u=U, N=2048):
    """Area and Berry flux of the pocket P = {m_orb > 0} (closed-form grid).

    Returns dict with A_pocket, Phi_pocket (= area integral of Omega_- over
    P; equals the m_orb integral in this model), ne = 1 - A_pocket/(4 pi^2).
    """
    k = -np.pi + 2 * np.pi * (np.arange(N) + 0.5) / N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    m = m_orb(KX, KY, u)
    dA = (2 * np.pi / N) ** 2
    pos = m > 0
    A_p = float(pos.sum() * dA)
    Phi_p = float(m[pos].sum() * dA)
    return dict(A_pocket=A_p, Phi_pocket=Phi_p,
                ne=1.0 - A_p / (4 * np.pi ** 2))


def zero_contour_c2(u=U, Nc=512, n_c2=40, h=1e-5):
    """J = (1/2pi) oint_{m_orb=0} c2/|grad m_orb| dl (unsigned dl), plus
    coarea perimeter integral P = oint dl/|grad m_orb| = |dA_pocket/dm|.

    Contour extracted on [0, 2pi]^2 (pocket around M is contiguous there);
    c2 evaluated pointwise on n_c2 equally-spaced contour points and
    interpolated linearly in arclength onto segment midpoints.
    """
    import matplotlib.pyplot as _plt
    kf = 2 * np.pi * np.arange(Nc + 1) / Nc
    KX, KY = np.meshgrid(kf, kf, indexing="ij")
    cs = _plt.contour(KX, KY, m_orb(KX, KY, u), levels=[0.0])
    _plt.close()
    segs = [s for lev in cs.allsegs for s in lev]
    c = max(segs, key=len)
    kxc, kyc = c[:, 0], c[:, 1]
    gx = (m_orb(kxc + h, kyc, u) - m_orb(kxc - h, kyc, u)) / (2 * h)
    gy = (m_orb(kxc, kyc + h, u) - m_orb(kxc, kyc - h, u)) / (2 * h)
    gn = np.hypot(gx, gy)
    dl = np.hypot(np.diff(kxc), np.diff(kyc))
    s_mid = np.concatenate([[0.0], np.cumsum(dl)])[:-1] + dl / 2
    L = float(dl.sum())
    # c2 at n_c2 anchor points along arclength, linear interp to midpoints
    s_anchor = np.linspace(0, L, n_c2, endpoint=False)
    idx = np.searchsorted(np.cumsum(dl), s_anchor)
    idx = np.clip(idx, 0, len(kxc) - 1)
    c2_anchor = np.array([c2_flat(float(kxc[i]), float(kyc[i]), u=u)
                          for i in idx])
    c2_mid = np.interp(s_mid, s_anchor, c2_anchor, period=L)
    gnm = 0.5 * (gn[1:] + gn[:-1])
    P = float(np.sum(dl / gnm))
    J = float(np.sum(c2_mid * dl / gnm) / (2 * np.pi))
    return dict(J=J, P_coarea=P, perimeter=L,
                c2_mean=float(2 * np.pi * J / P))


def staircase_envelope(qs, h, nu, qmin, qmax):
    """Envelope of g(q) = nu*q - h(q) on a q-window: gamma in (max g - 1, min g].

    Returns (lo, hi, slack); the interval is nonempty iff slack >= 0,
    slack = 1 - (max g - min g).
    """
    sel = (qs >= qmin) & (qs <= qmax)
    g = nu * qs[sel] - h[sel]
    lo = float(g.max() - 1.0)
    hi = float(g.min())
    return lo, hi, float(1.0 - (g.max() - g.min()))


def staircase_drift_band(qs, h, nu, c_lo=-1.0, c_hi=4.0, nc=2001):
    """Best 1/q-drift-corrected envelope: gamma_eff(q) = gamma_inf + c/q.

    Scans c, returns the c minimizing the spread of g(q) - c/q and the
    resulting gamma_inf interval (lo, hi].
    """
    g = nu * qs - h
    best = (np.inf, None)
    for c in np.linspace(c_lo, c_hi, nc):
        z = g - c / qs
        wdt = z.max() - z.min()
        if wdt < best[0]:
            best = (wdt, c)
    wdt, c = best
    z = g - c / qs
    return dict(c=float(c), width=float(wdt),
                lo=float(z.max() - 1.0), hi=float(z.min()))


def pocket_area_polygon(u=U, N=4096):
    """Pocket area A~(0) by shoelace on the marching-squares zero contour.

    More accurate than pixel counting (grid bias ~1e-4 at N=4096 is the
    dominant systematic of the parked-mu phase estimator).
    """
    import matplotlib.pyplot as _plt
    kf = 2 * np.pi * np.arange(N + 1) / N
    KX, KY = np.meshgrid(kf, kf, indexing="ij")
    cs = _plt.contour(KX, KY, m_orb(KX, KY, u), levels=[0.0])
    _plt.close()
    c = max([s for lev in cs.allsegs for s in lev], key=len)
    x, y = c[:, 0], c[:, 1]
    A = abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(4 * np.pi ** 2 - A if A > 2 * np.pi ** 2 else A)


def spectroscopy_gamma(evals_pad, qs, A0):
    """Sub-level phase estimator at mu = -1: for each q, the fractional
    ladder index of the m_orb = 0 crossing, gamma_est(q) =
    frac(A~(0) q / 4 pi^2 - n_zero).  Linear interpolation in m between
    the two levels bracketing zero (bias O(1/q^2), removed by the drift
    fit).  Returns array over qs."""
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        e = evals_pad[i]
        e = e[np.isfinite(e)]
        B = 2 * np.pi / q
        m = (e + 1.0) / B
        m_lo = m[m < 0].max()
        m_hi = m[m > 0].min()
        nstar = float((m > 0).sum())
        n_zero = (nstar - 1) + m_hi / (m_hi - m_lo)
        out[i] = (A0 * q / (4 * np.pi ** 2) - n_zero) % 1.0
    return out


def drift_fit(qs, gam, qmin=100, deg=1):
    """LSQ fit gamma_est(q) = gamma_inf + c1/q (+ c2/q^2).  Returns dict."""
    qs = np.asarray(qs, float)
    s = qs >= qmin
    X = np.vstack([qs[s] ** (-float(p)) for p in range(deg + 1)]).T
    cf, *_ = np.linalg.lstsq(X, gam[s], rcond=None)
    r = gam[s] - X @ cf
    return dict(gamma_inf=float(cf[0]), c=float(cf[1]),
                rms=float(r.std()), coef=[float(v) for v in cf])


# ==================================================================
#  Canonical ensemble (fixed N): T = 0 energy, mu staircase, arc
# ==================================================================
def canonical_energy(evals_pad, qs, ne):
    """T=0 canonical energy per site at filling ne, and s = frac(ne*q).

    E = phi * [sum_{n<K} eps_n + s * eps_K], K = floor(ne*q):
    K filled levels plus fractional occupation s of the next.
    """
    E = np.empty(len(qs))
    S = np.empty(len(qs))
    for i, q in enumerate(qs):
        e = np.sort(evals_pad[i][np.isfinite(evals_pad[i])])
        x = ne * q
        K = int(np.floor(x))
        S[i] = x - K
        E[i] = (e[:K].sum() + S[i] * e[K]) / q
    return E, S


def mu_staircase(evals_pad, qs, ne):
    """T->0 canonical mu(N,B): the partially filled level's energy."""
    mu = np.empty(len(qs))
    for i, q in enumerate(qs):
        e = np.sort(evals_pad[i][np.isfinite(evals_pad[i])])
        mu[i] = e[int(np.floor(ne * q))]
    return mu


def gamma_from_heights(evals_pad, qs, ne, m_grid, A_m,
                       qmin=100, smin=0.1, smax=0.9):
    """Exact-area staircase-height estimator of the graded gamma.

    gamma_imp(q) = A(m_mu)/(2 pi B) - K with m_mu = (eps_K + 1)/B read
    through the zero-field area function (exact in the ladder
    nonlinearity; the linearized form mu = -1 + m0 B + hw(gamma - s)
    is biased by A(m) curvature).  Excludes near-integer fillings.
    Returns (q_sel, gamma_pts).
    """
    out_q, out_g = [], []
    for i, q in enumerate(qs):
        if q < qmin:
            continue
        x = ne * q
        K = int(np.floor(x))
        s = x - K
        if s < smin or s > smax:
            continue
        e = np.sort(evals_pad[i][np.isfinite(evals_pad[i])])
        B = 2 * np.pi / q
        m_mu = (e[K] + 1.0) / B
        out_q.append(float(q))
        out_g.append(float(np.interp(m_mu, m_grid, A_m) / (2 * np.pi * B) - K))
    return np.array(out_q), np.array(out_g)


def arc_prediction(qs, S, dmdA):
    """Zero-mean flat-band Peierls arc in q^3*E units:
    q^3 * (phi * hbar_omega_eff / 2) * (s(1-s) - 1/6)."""
    qs = np.asarray(qs, float)
    hw = 2 * np.pi * (2 * np.pi / qs) ** 2 * dmdA
    return qs ** 3 * ((1.0 / qs) * hw / 2 * (S * (1 - S) - 1.0 / 6.0))


def detrend_mixed(x, y, powers=(3, 2, 1, 0, -1, -2)):
    """LSQ residual on span{(x/mean)^p} for mixed positive/negative powers
    (needed for q^3-scaled canonical energy; inverse-only basis cannot
    absorb the growing background)."""
    x = np.asarray(x, float)
    u = x / x.mean()
    M = np.vstack([u ** float(p) for p in powers]).T
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    return y - M @ coef, M
