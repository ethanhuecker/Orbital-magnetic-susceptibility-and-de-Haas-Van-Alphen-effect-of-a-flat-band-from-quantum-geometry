# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Sparse Harper assembly adapted from the supplied sweep_counts4.py.
# Added checked inertia and shift-invert spectroscopy for reproducibility.
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh, LinearOperator
import flatqwz as fq

def harper_sparse_cross(hops, q, delta, k1=fq.K1_DEFAULT, k2=fq.K2_DEFAULT):
    r = hops["r"]
    cxs, cys, czs = hops["cxs"], hops["cys"], hops["czs"]
    ns = np.arange(-r, r + 1)
    s = np.arange(q)
    rows, cols, vals = [], [], []

    def add(rr, cc, vv):
        rows.append(np.asarray(rr).ravel())
        cols.append(np.asarray(cc).ravel())
        vals.append(np.asarray(vv, dtype=complex).ravel())

    for mi, m in enumerate(ns):
        sp_ = (s + m) % q
        c = (s + m - sp_) // q
        ph = np.exp(1j * np.outer(ns, 2 * np.pi / q * (s + m / 2) + k2))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            fx = cxs[mi] @ ph
            fy = cys[mi] @ ph
            fz = czs[mi] @ ph
        pk = np.exp(1j * k1 * c)
        add(2 * sp_, 2 * s, fz * pk)
        add(2 * sp_, 2 * s + 1, (fx - 1j * fy) * pk)
        add(2 * sp_ + 1, 2 * s, (fx + 1j * fy) * pk)
        add(2 * sp_ + 1, 2 * s + 1, -fz * pk)
    if delta != 0.0:
        idx = np.arange(2 * q)
        add(idx, idx, np.full(2 * q, 2.0 * delta))
        for m in (1, -1):
            sp_ = (s + m) % q
            c = (s + m - sp_) // q
            pk = np.broadcast_to(-0.5 * delta * np.exp(1j * k1 * c), s.shape)
            add(2 * sp_, 2 * s, pk)
            add(2 * sp_ + 1, 2 * s + 1, pk)
        for n in (1, -1):
            ph = -0.5 * delta * np.exp(1j * n * (2 * np.pi / q * s + k2))
            add(2 * s, 2 * s, ph)
            add(2 * s + 1, 2 * s + 1, ph)
    return sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * q, 2 * q))


def real_inertia(A):
    """Real symmetric representation: every eigenvalue occurs twice.

    Interleaved (Re z, Im z) keeps the sparse band structure. This fallback
    avoids using signs of complex pivots whose imaginary roundoff grew large.
    """
    C=A.tocoo();n=A.shape[0]
    rr=np.repeat(2*C.row,4)+np.tile([0,0,1,1],C.nnz)
    cc=np.repeat(2*C.col,4)+np.tile([0,1,0,1],C.nnz)
    vv=np.column_stack((C.data.real,-C.data.imag,C.data.imag,C.data.real)).ravel()
    R=sp.csc_matrix((vv,(rr,cc)),shape=(2*n,2*n));R.eliminate_zeros()
    # Remove only floating-point anti-Hermitian noise; recorded before this step.
    R=(R+R.T)*.5
    lu=splu(R,permc_spec='NATURAL',diag_pivot_thresh=0.,
            options=dict(SymmetricMode=True,Equil=False))
    if not np.array_equal(lu.perm_r,lu.perm_c):
        raise ArithmeticError('Real inertia requires a symmetric pivot permutation')
    d=lu.U.diagonal()
    neg=int(np.sum(d<0))
    if not np.isfinite(d).all() or neg%2 or np.min(np.abs(d))<1e-12:
        raise ArithmeticError('Real inertia has ambiguous pivots or odd eigenvalue count')
    rng=np.random.default_rng(21);rhs=rng.normal(size=2*n);sol=lu.solve(rhs)
    normR=float(np.max(np.asarray(np.abs(R).sum(axis=1)).ravel()))
    back=float(np.linalg.norm(R@sol-rhs,np.inf)/(normR*np.linalg.norm(sol,np.inf)+np.linalg.norm(rhs,np.inf)))
    # Check that the computed factors have the LDL^T relation on a probe vector.
    v=rng.normal(size=2*n);uv=lu.U@v;dlv=d*(lu.L.T@v)
    congruence=float(np.linalg.norm(uv-dlv,np.inf)/max(1.,np.linalg.norm(uv,np.inf)))
    if back>1e-9 or congruence>1e-6:
        raise ArithmeticError(f'Real LDL accuracy failed: {back}, {congruence}')
    return neg//2,dict(real_backward_error=back,ldlt_probe_error=congruence,
                       real_min_abs_pivot=float(np.min(np.abs(d))))


def checked_factor(H,mu):
    """Use unpivoted LU as LDL^H only after verifying its congruence conditions.

    SuperLU is an LU solver, NOT a general Hermitian-inertia routine.
    Merely counting real(U.diagonal()) is invalid if row/column permutations
    differ, pivots have appreciable imaginary parts, or factor accuracy fails.
    Such cases raise; they must never be silently counted as successful runs.
    """
    err=H-H.conj().T
    herm=float(np.max(np.abs(err.data))) if err.nnz else 0.
    if herm>1e-11:
        raise ArithmeticError(f'Hermiticity residual {herm}')
    A=H-mu*sp.eye(H.shape[0],format='csc',dtype=complex)
    lu=splu(A,permc_spec='NATURAL',diag_pivot_thresh=0.,
            options=dict(SymmetricMode=True,Equil=False))
    if not np.array_equal(lu.perm_r,lu.perm_c):
        raise ArithmeticError('LU pivot permutations are not a Hermitian congruence')
    piv=lu.U.diagonal()
    imag=float(np.max(np.abs(piv.imag)/np.maximum(1.,np.abs(piv.real))))
    extra={};method='complex_unpivoted_LDLH'
    neg=int(np.sum(piv.real<0))
    if imag>1e-6 or np.any(~np.isfinite(piv)):
        # An independent real symmetric inertia count is required in this case.
        del lu
        neg,extra=real_inertia(A)
        lu=splu(A)  # Stable pivoted solver, not used to infer inertia.
        method='real_symmetric_doubled_LDLT'
    # Reproducible backward-error probe, independent of any expected count.
    rng=np.random.default_rng(19)
    rhs=rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0])
    sol=lu.solve(rhs)
    normA=float(np.max(np.asarray(np.abs(A).sum(axis=1)).ravel()))
    back=float(np.linalg.norm(A@sol-rhs,np.inf)/(normA*np.linalg.norm(sol,np.inf)+np.linalg.norm(rhs,np.inf)))
    if back>1e-9:
        raise ArithmeticError(f'Factor backward error {back}')
    return lu,dict(hermiticity=herm,pivot_imag_relative=imag,method=method,**extra,
                   min_abs_pivot=float(np.min(np.abs(piv.real))),
                   backward_error=back,negative=neg)


def count_levels(hops,q,W=2.79e-4,k1=fq.K1_DEFAULT,k2=fq.K2_DEFAULT):
    H=harper_sparse_cross(hops,q,W,k1,k2)
    _,info=checked_factor(H,-1+3.5*W)
    return info


def near_mu(hops,q,W=0.,mu=None,k=6):
    if mu is None: mu=-1+3.5*W
    H=harper_sparse_cross(hops,q,W)
    lu,info=checked_factor(H,mu)
    op=LinearOperator(H.shape,matvec=lu.solve,dtype=complex)
    v0=np.random.default_rng(4).normal(size=H.shape[0])
    ev,V=eigsh(H,k=k,sigma=mu,which='LM',OPinv=op,tol=1e-11,v0=v0)
    res=float(np.max(np.linalg.norm(H@V-V*ev,axis=0)))
    if res>1e-9: raise ArithmeticError(f'Eigenpair residual {res}')
    ev=np.sort(ev)
    lo=ev[ev<mu][-1];hi=ev[ev>mu][0]
    info.update(q=int(q),W=W,mu=mu,gap=float(hi-lo),lo=float(lo),hi=float(hi),
                eigenpair_residual=res,eigenvalues=ev.tolist())
    # Lower-band cluster contains q-1 levels (C=-1). Count includes all
    # occupied states; none from the remote band for these chemical potentials.
    hole_count=q-1-info['negative']
    info['hole_fractional_index']=float(hole_count-1+(hi-mu)/(hi-lo))
    return info

