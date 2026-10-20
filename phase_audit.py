"""Separate phase-content test described on page 14, beyond Fig. 5's fields.

The missing original phase-sweep grid cannot be recovered. This explicit grid
straddles the predicted B_gamma. It uses full dense spectra, not the contour
prediction, to obtain each fractional level index. The difference between the
W=0 and W>0 intercepts cancels the common finite-field phase error.
"""
import hashlib
import json
import numpy as np
from scipy.linalg import eigh
import geometry as geo
import qwz_cross as qc
from reproduce import ROOT,CONFIG,get_hops,fingerprint,write_json

TAU=2*np.pi

def phase_fingerprint():
    h=hashlib.sha256(fingerprint().encode())
    for name in ['phase_config.json','phase_audit.py']:
        h.update(name.encode());h.update((ROOT/name).read_bytes())
    return h.hexdigest()

def fractional_hole_index(ev,mu,q):
    cluster=ev[ev<0]
    if len(cluster)!=q-1:
        raise ArithmeticError('Unexpected cluster size in phase audit')
    hi=cluster[cluster>mu][0];lo=cluster[cluster<mu][-1]
    # The highest level is index zero; interpolate between adjacent levels.
    return float(np.sum(cluster>mu)-1+(hi-mu)/(hi-lo))

def main():
    cfg=json.loads((ROOT/'phase_config.json').read_text())
    qs=np.arange(cfg['q_min'],cfg['q_max']+1,cfg['q_step'])
    g=json.loads((ROOT/'data/geometry.json').read_text())
    W=CONFIG['W'];am=g['pinned']['area'];gam=g['pinned']['gamma']
    content=abs(g['pinned']['gamma1']+g['pinned']['gamma2'])
    lam=g['pinned']['Lambda_g'];bgamma=np.sqrt(W*lam/(TAU*content))
    rawflat=np.full((len(qs),2*max(qs)),np.nan)
    rawcross=rawflat.copy();rows=[];herm=0.
    for i,q in enumerate(qs):
        B=TAU/q;c=geo.contour(B=B,W=W)
        indices=[]
        for w,store in [(0.,rawflat),(W,rawcross)]:
            H=qc.harper_matrix_cross(get_hops(),int(q),w,CONFIG['k1'],CONFIG['k2'])
            herm=max(herm,float(np.max(abs(H-H.conj().T))))
            ev=eigh(H,eigvals_only=True,driver='evr',check_finite=False)
            store[i,:len(ev)]=ev
            indices.append(fractional_hole_index(ev,-1+CONFIG['g0']*w,q))
        n0,nw=indices
        # With fixed frequency A_M/(2*pi), intercept = A_M/(2*pi*B)-n.
        # Subtracting the two intercepts gives n0-nw with no modulo ambiguity.
        measured=n0-nw
        area_residual=(am-c['area'])/(TAU*B)
        predicted=area_residual+c['gamma']-gam
        rows.append(dict(q=int(q),B=float(B),flat_fractional_index=n0,
            narrow_fractional_index=nw,contamination_ED=measured,
            contamination_composite=float(predicted),
            contamination_asymptotic=float(W*lam/(TAU*B**2)),
            area_residual=float(area_residual),
            gamma_flat_ED=float(am/(TAU*B)-n0),
            gamma_narrow_ED=float(c['area']/(TAU*B)-nw),
            gamma_narrow_theory=float(c['gamma']),
            differential_phase_error=float(measured-predicted)))
    # Keep every crossing: discrete-level interpolation is slightly oscillatory.
    # Never select the crossing closest to the theoretical answer.
    crossings=[]
    for a,b in zip(rows[:-1],rows[1:]):
        ya=a['contamination_ED']-content;yb=b['contamination_ED']-content
        if ya*yb<0:
            qr=a['q']-ya*(b['q']-a['q'])/(yb-ya)
            crossings.append(dict(q_bracket=[a['q'],b['q']],
                B_bracket=sorted([a['B'],b['B']]),q_interpolated=qr,
                B_interpolated=float(TAU/qr)))
    np.savez_compressed(ROOT/'data/phase_scale_spectra.npz',qs=qs,
        evals_flat=rawflat,evals_narrow=rawcross,
        source_fingerprint=phase_fingerprint())
    out=dict(configuration=cfg,rows=rows,crossings=crossings,
        geometric_content=content,B_gamma_prediction=float(bgamma),
        max_differential_phase_error=max(abs(r['differential_phase_error']) for r in rows),
        max_crossing_relative_error=max((abs(r['B_interpolated']/bgamma-1) for r in crossings),default=None),
        B_min=float(TAU/max(qs)),B_max=float(TAU/min(qs)),
        max_expansion_parameter=float(TAU/min(qs)*g['xi2_grid']),
        hermiticity=herm,source_fingerprint=phase_fingerprint(),
        method='Paired full spectra; linear interpolation of level indices; retain all crossing brackets.',
        limitation='The grid is reconstructed. Its largest fields are outside the Fig. 5 x<=0.10 window.')
    write_json(ROOT/'data/phase_scale_analysis.json',out)
    print('Separate phase-scale audit:',{k:out[k] for k in [
        'B_gamma_prediction','max_differential_phase_error','max_crossing_relative_error']},flush=True)
    return out

if __name__=='__main__':main()
