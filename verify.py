"""Independent numerical checks and explicit tolerances (not self-consistency only).

Targets rounded in the paper are compared separately from implementation checks.
Validation does not claim a proof of the general multiband analytic derivation.
"""
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
import flatqwz as fq
import qwz_cross as qc
import geometry as geo
import sparse_spectrum as spc
from reproduce import ROOT,CONFIG,get_hops,fingerprint,write_json
from analyze import spectrum,grand,chi_closed

def main():
    checks=[]
    def check(name,value,tolerance,details=None):
        good=bool(np.isfinite(value) and abs(value)<=tolerance)
        row=dict(name=name,value=float(value),tolerance=tolerance,passed=good)
        if details is not None:row['details']=details
        checks.append(row)
        print(('PASS' if good else 'FAIL'),name,f'{value:.6g}',flush=True)
    h=get_hops();u=CONFIG['u'];rng=np.random.default_rng(261012193)
    g=json.loads((ROOT/'data/geometry.json').read_text())
    flat=json.loads((ROOT/'data/flat_analysis.json').read_text())
    chi=json.loads((ROOT/'data/susceptibility.json').read_text())
    cross=json.loads((ROOT/'data/crossover_analysis.json').read_text())
    check('Chern number -1',g['chern']+1,1e-10)
    check('BZ integral of U2 vanishes',g['U2_mean'],1e-10)
    # Grid convergence of smooth zone integrals.
    gl=geo.global_geometry(512,u)
    check('I convergence 512 to 1024',gl['I']-g['I'],1e-11)
    # Independent pointwise formula from aligned eigenvectors (Eq.17) vs Eq.35.
    pts=rng.uniform(-np.pi,np.pi,(20,2))
    diffs=[];fd=[]
    for x,y in pts:
        xx=geo.X_closed(x,y,u)
        a=2*fq.c2_flat(x,y,h=1e-3,u=u);b=2*fq.c2_flat(x,y,h=2e-3,u=u)
        diffs.append(abs(a-xx));fd.append(abs(a-b))
    check('X Eq.35 vs independent eigenvector Eq.17',max(diffs),2e-7)
    check('Eigenvector finite-difference step convergence',max(fd),2e-7)
    # Constant energy shifts and arbitrary smooth eigenvector phases cannot change X.
    oldH=fq._H_qwz;oldV=fq._flat_vec
    base=np.array([fq.c2_flat(x,y,u=u) for x,y in pts[:5]])
    try:
        fq._H_qwz=lambda k,u=u:oldH(k,u)+.37*np.eye(2)
        shift=np.array([fq.c2_flat(x,y,u=u) for x,y in pts[:5]])
    finally:fq._H_qwz=oldH
    def gauged(k,u=u):
        w0,v,w,V,i=oldV(k,u);phase=np.exp(1j*(.7*np.sin(k[0])+1.3*np.cos(k[1])))
        V=V.copy();V[:,i]*=phase
        return w0,v*phase,w,V,i
    try:
        fq._flat_vec=gauged
        changed=np.array([fq.c2_flat(x,y,u=u) for x,y in pts[:5]])
    finally:fq._flat_vec=oldV
    check('Constant Hamiltonian energy-shift invariance',np.max(abs(shift-base)),2e-7)
    check('Smooth eigenvector gauge invariance',np.max(abs(changed-base)),2e-7)
    # Analytic Berry curvature vs independently differentiated Bloch projector.
    n,nx,ny,*_=geo.n_derivatives(pts[:,0],pts[:,1],u)
    projector_om=-.5*np.sum(n*np.cross(nx,ny),axis=-1)
    check('Berry curvature analytic vs projector derivatives',np.max(abs(projector_om-fq.m_orb(pts[:,0],pts[:,1],u))),1e-12)
    # Fourier transform and truncation convergence; no magnetic calculation involved.
    h512=fq.build_hoppings(N=512,r=30,u=u)
    check('FFT hopping convergence 512 to 1024',max(np.max(abs(h[k]-h512[k])) for k in ['cxs','cys','czs']),1e-13)
    h36=fq.build_hoppings(N=1024,r=36,u=u)
    ids=np.arange(-36,37);mask=np.maximum(abs(ids[:,None]),abs(ids[None,:]))>30
    maxdrop=max(np.max(abs(h36[k][mask])) for k in ['cxs','cys','czs'])
    check('Largest discarded Fourier coefficient outside radius 30',maxdrop,1e-12)
    ms=np.arange(-30,31);err=0.
    for k in pts[:8]:
        ph=np.exp(-1j*(ms[:,None]*k[0]+ms[None,:]*k[1]))
        HH=sum(np.sum(h[key]*ph)*sigma for key,sigma in zip(['cxs','cys','czs'],[fq.SX,fq.SY,fq.SZ]))
        err=max(err,float(np.max(abs(HH-fq._H_qwz(k,u)))))
    check('Zero-field Fourier reconstruction',err,1e-10)
    # Slow independent per-hop assembly tests wrapping for q smaller than radius.
    q=17;ref=np.zeros((2*q,2*q),complex);s=np.arange(q)
    for mi,m in enumerate(ms):
        target=(s+m)%q;c=(s+m-target)//q
        for ni,nv in enumerate(ms):
            t=h['cxs'][mi,ni]*fq.SX+h['cys'][mi,ni]*fq.SY+h['czs'][mi,ni]*fq.SZ
            phase=np.exp(1j*(2*np.pi/q*nv*(s+m/2)+CONFIG['k2']*nv+CONFIG['k1']*c))
            for a in [0,1]:
                for b in [0,1]:ref[2*target+a,2*s+b]+=t[a,b]*phase
    check('Harper phases vs independent per-hop assembly',np.max(abs(ref-fq.harper_matrix(h,q))),1e-12)
    maxmatrix=maxcount=maxsolver=maxnear=0.
    for q in [60,150,400]:
        H=qc.harper_matrix_cross(h,q,CONFIG['W'])
        S=spc.harper_sparse_cross(h,q,CONFIG['W'])
        maxmatrix=max(maxmatrix,float(np.max(abs(H-S.toarray()))))
        # Request vectors so EVD uses divide-and-conquer, not the eigenvalues-only
        # tridiagonal shortcut that can coincide with EVR's internal path.
        ev,_=eigh(H,eigvals_only=False,driver='evd');mu=-1+3.5*CONFIG['W']
        count=spc.count_levels(h,q,CONFIG['W'])['negative']
        realcount,_=spc.real_inertia(S-mu*spc.sp.eye(2*q,format='csc'))
        maxcount=max(maxcount,abs(count-np.sum(ev<mu)),abs(realcount-count))
        near=spc.near_mu(h,q,CONFIG['W'])
        gap=ev[ev>mu][0]-ev[ev<mu][-1]
        maxnear=max(maxnear,abs(near['gap']/gap-1))
        alt=eigh(H,eigvals_only=True,driver='evr')
        maxsolver=max(maxsolver,float(np.max(abs(ev-alt))))
    check('Sparse vs dense Harper matrices',maxmatrix,1e-12)
    check('Complex/real inertia counts vs dense ED',maxcount,0.)
    check('Independent dense LAPACK drivers',maxsolver,1e-11)
    check('Shift-invert spacing vs full diagonalization',maxnear,1e-7)
    # Magnetic Bloch dependence at a representative plotted q.
    q=150;reference=spectrum(q);reference_gap=reference[reference>-1][0]-reference[reference<-1][-1]
    twist=[]
    for k1,k2 in [(0.,0.),(1.7,2.1),(np.pi,np.pi)]:
        ev,_=fq.harper_spectrum(h,q,k1,k2)
        sel=np.argsort(abs(reference+1))[:6]
        twist.append(np.max(abs(ev[sel]-reference[sel]))/reference_gap)
    check('Magnetic Bloch dispersion near pinned mu / spacing',max(twist),1e-6)
    # Resolves the concrete risk that finite hopping tails dominate tiny high-q gaps.
    near30=spc.near_mu(h,2560);near36=spc.near_mu(h36,2560)
    check('Radius 30 to 36 convergence at q=2560',near36['gap']/near30['gap']-1,1e-5)
    # Independent one-dimensional zero-contour integrals using cos ky explicitly.
    x0=np.arccos(1/(1-u))
    def yl(x):return np.arccos(np.clip(-np.cos(x)/(1+u*np.cos(x)),-1,1))
    A=4*quad(lambda x:np.pi-yl(x),x0,np.pi,epsabs=2e-10,epsrel=2e-12)[0]
    P=4*quad(lambda x:1/abs(geo.moment_gradient(x,yl(x),u)[1]),x0,np.pi,epsabs=1e-8,epsrel=2e-11)[0]
    check('Pocket area polar vs independent Cartesian quadrature',A-g['pinned']['area'],1e-8)
    check('Coarea polar vs independent Cartesian quadrature',P-g['pinned']['coarea'],1e-7)
    small=geo.contour(ntheta=256,nrad=40)
    check('Contour quadrature 256 to 1024 angles',max(abs(small[k]-g['pinned'][k]) for k in ['area','coarea','gamma']),1e-8)
    check('Fixed moment contour area is independent of B',g['inband']['dA_dB'],1e-10)
    # Derivative in the chirp frequency is not the instantaneous area itself.
    deriv=[]
    for B in [.0005,.005,.03]:
        dB=B*1e-4;c=geo.contour(B=B,W=CONFIG['W'])
        lo=geo.contour(B=B-dB,W=CONFIG['W'])['area'];hi=geo.contour(B=B+dB,W=CONFIG['W'])['area']
        deriv.append(abs((hi-lo)/(2*dB)-c['dA_dB'])/max(1,abs(c['dA_dB'])))
    check('Composite area derivative vs field finite difference',max(deriv),1e-7)
    # Provenance and all-run diagnostics.
    specfiles=list((ROOT/'data/spectra').glob('q*.npz'));bad=0;herm=0.
    for path in specfiles:
        with np.load(path) as z:
            bad+=str(z['source_fingerprint'])!=fingerprint()
            herm=max(herm,float(z['hermiticity']))
    check('No stale spectrum cache',bad,0.,{'spectra':len(specfiles)})
    check('Full sweep Hermiticity',herm,1e-11)
    counts=[json.loads(p.read_text()) for p in (ROOT/'data/count_cache').glob('q*.json')]
    check('No stale inertia cache',sum(r['source_fingerprint']!=fingerprint() for r in counts),0.,{'counts':len(counts)})
    check('All count solve backward errors',max(r['backward_error'] for r in counts),1e-9)
    methods={m:sum(r['method']==m for r in counts) for m in sorted(set(r['method'] for r in counts))}
    # Paper comparisons. Finite-field deviations are expected, so quote them.
    check('Paper I=0.0732 (rounding interval)',g['I']-.0732,.00005)
    check('Paper gamma=0.874 (rounding interval)',g['pinned']['gamma']-.874,.0005)
    check('Paper frequency=2.26 (rounding interval)',g['pinned']['Fobs']-2.26,.005)
    check('Bandwidth weak-field slope relative error',flat['bandwidth_slope']/flat['bandwidth_slope_prediction']-1,.002)
    check('Spacing relative error at q=2560',flat['spacing_relative_error_at_q2560'],.003)
    check('Spacing B exponent',flat['spacing_log_slope']-2,.01)
    check('Phase extrapolation q>=400 linear vs geometry',flat['phase_fits']['qmin400_degree1']['gamma_infinity']-g['pinned']['gamma'],.001)
    check('Susceptibility relative errors',max(abs(r['chi_ED']/r['chi_closed']-1) for r in chi['mu_scan_T0.1']),.0001)
    check('Susceptibility independent fit window stability',max(abs(r['chi_ED']-r['chi_window_480_1200']) for r in chi['mu_scan_T0.1']),2e-6)
    check('Smooth magnetization absolute error',max(abs(r['M_ED']-r['M_closed']) for r in chi['mu_scan_T0.1']),1e-7)
    check('Pinned waveform relative RMS error',flat['waveform_rms_error_over_theory_rms'],.10)
    check('Pinned damping max relative error above 0.05',flat['damping_pinned']['max_relative_error_pred_gt_005'],.05)
    check('Caption damping max relative error above 0.05',flat['damping_caption']['max_relative_error_pred_gt_005'],.10)
    check('Crossover count frequency median error per q',cross['median_frequency_absolute_error_per_q'],.015)
    check('Crossover differential phase max absolute error',cross['max_differential_phase_error'],.02)
    # Check conservation of the physical conclusion rather than imposing a fit to it.
    ss=cross['spectroscopy'];B=np.array([r['B'] for r in ss]);gap=np.array([r['cross']['gap'] for r in ss])
    order=np.argsort(B);B=B[order];gap=gap[order]
    powerlow=float(np.polyfit(np.log(B[:4]),np.log(gap[:4]),1)[0])
    powerhigh=float(np.polyfit(np.log(B[-4:]),np.log(gap[-4:]),1)[0])
    check('Crossover weak-field spacing exponent near 1',powerlow-1,.15)
    # The finite plotted range need not yet have exponent exactly 2.
    check('Crossover spacing exponent rises toward 2',max(0.,powerlow+.4-powerhigh),0.,
          {'low_field_exponent':powerlow,'high_field_exponent':powerhigh})
    # The paper's separate phase-scale experiment extends beyond Fig. 5's fields.
    from phase_audit import phase_fingerprint
    phase=json.loads((ROOT/'data/phase_scale_analysis.json').read_text())
    with np.load(ROOT/'data/phase_scale_spectra.npz') as rawphase:
        stale=(str(rawphase['source_fingerprint'])!=phase_fingerprint()
               or phase['source_fingerprint']!=phase_fingerprint())
    check('No stale separate phase-scale data',int(stale),0.)
    check('Phase-scale differential phase absolute error',phase['max_differential_phase_error'],.03)
    cross_error=phase['max_crossing_relative_error']
    check('All measured phase-scale crossings vs Eq.77',1e99 if cross_error is None else cross_error,.05,
          {'crossings':phase['crossings'],'B_gamma_prediction':phase['B_gamma_prediction']})
    out=dict(all_passed=all(c['passed'] for c in checks),checks=checks,
        count_methods=methods,source_fingerprint=fingerprint(),
        limitations=['Original missing data caches and exact extended window list were not supplied.',
          'Figure 4b source uses pinned mu; its published caption specifies an in-band contour.',
          'Finite-field LK waveform has a measurable residual; agreement is not exact.',
          'The supplied q=88..112 first count window extends slightly above x=0.10.',
          'Separate phase-scale fields extend beyond the Fig. 5 weak-field window; all crossing brackets are reported.',
          'Checks verify this two-band numerical model, not the full multiband proof.'])
    write_json(ROOT/'validation/checks.json',out)
    from report import write_report
    write_report(out)
    if not out['all_passed']:raise SystemExit('Some validation checks failed; see validation/checks.json')
    print(f'All {len(checks)} checks passed.',flush=True)

if __name__=='__main__':main()
