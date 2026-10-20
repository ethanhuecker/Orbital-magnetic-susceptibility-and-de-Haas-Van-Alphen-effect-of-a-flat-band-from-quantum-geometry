"""Derive all plotted observables from saved raw spectra and level counts."""
import json
import numpy as np
from scipy.special import expit
import flatqwz as fq
import geometry as geo
import sparse_spectrum as spc
from reproduce import ROOT,CONFIG,write_json,get_hops

TAU=2*np.pi

def spectrum(q):
    with np.load(ROOT/'data/spectra'/f'q{int(q):05d}.npz') as d:return d['evals']

def grand(ev,q,mu,T):
    if T==0:return np.minimum(ev-mu,0).sum()/q
    return -T*np.logaddexp(0,(mu-ev)/T).sum()/q

def chi_closed(mu,T,I):
    fm,fp=expit((mu+1)/T),expit((mu-1)/T)
    return -I*(fm-fp-(fm*(1-fm)+fp*(1-fp))/T)

def susceptibility(g):
    qs=np.array(CONFIG['chi_q']);B=TAU/qs;scale=np.max(B)
    ev=[spectrum(q) for q in qs]
    T=CONFIG['chi_T']; rows=[]
    for mu in CONFIG['chi_mu']:
        y=np.array([grand(e,q,mu,T) for e,q in zip(ev,qs)])
        y0=-T*np.logaddexp(0,(mu-np.array([-1.,1.]))/T).sum()
        cf=np.polynomial.polynomial.polyfit(B/scale,y-y0,4)
        pred=y0+np.polynomial.polynomial.polyval(B/scale,cf)
        cc=float(chi_closed(mu,T,g['I']))
        fm,fp=expit((mu+1)/T),expit((mu-1)/T)
        FTm,FTp=-T*np.logaddexp(0,(mu+1)/T),-T*np.logaddexp(0,(mu-1)/T)
        M=-(g['chern']/TAU)*(fm+fp+FTm-FTp)
        # A second window is a convergence check, not a fitted theory parameter.
        sel=qs>=480
        alt=np.polynomial.polynomial.polyfit(B[sel]/scale,y[sel]-y0,4)
        rows.append(dict(mu=mu,chi_ED=float(-2*cf[2]/scale**2),chi_closed=cc,
            M_ED=float(-cf[1]/scale),M_closed=float(M),
            chi_window_480_1200=float(-2*alt[2]/scale**2),
            fit_max_residual=float(np.max(np.abs(pred-y))),
            coefficients_in_B=(cf/scale**np.arange(5)).tolist()))
    np.savez_compressed(ROOT/'data/susceptibility_raw.npz',qs=qs,B=B,mus=CONFIG['chi_mu'],
        grand_potential=np.array([[grand(e,q,mu,T) for e,q in zip(ev,qs)] for mu in CONFIG['chi_mu']]),T=T)
    out={'I_Om2_1024':g['I'],'mu_scan_T0.1':rows,'T':T,'q':qs.tolist()}
    write_json(ROOT/'data/susceptibility.json',out)
    return out

def damping(qs,evs,g,m0):
    pin=m0==0
    geometry=g['pinned' if pin else 'inband']
    F=geometry['area']/TAU**2
    eff=TAU*(TAU/150)**2/geometry['coarea']
    mask=(qs>=130)&(qs<=170) if pin else (qs>=80)&(qs<=320)
    x=qs[mask].astype(float)
    def phasor(T):
        y=np.array([grand(e,q,-1+m0*TAU/q,T) for e,q in zip(evs,qs)])[mask]
        if pin:
            R,_=fq.detrend_rescaled(x,(x/150)**3*y*1e8)
            return fq.local_phasor(x,R,150,F,width=40)
        R,_=fq.scaled_residual(x,y)
        return fq.local_phasor(x,R,150,F,width=60)
    ab0=phasor(0)
    rows=[]
    for X in CONFIG['damping_X']:
        T=X*eff/(2*np.pi**2)
        D=fq.phase_coherent_ratio(phasor(T),ab0);pred=X/np.sinh(X)
        rows.append(dict(X=X,T=T,D_meas=D,D_pred=float(pred),relative_error=float(abs(D/pred-1))))
    return dict(m0=m0,mu_protocol='-1' if pin else '-1-B/2',q_center=150,
        spacing=eff,frequency_per_q=F,rows=rows,
        max_relative_error_pred_gt_005=max(r['relative_error'] for r in rows if r['D_pred']>.05))

def flat_analysis(g):
    qs=np.arange(CONFIG['flat_q_min'],CONFIG['flat_q_max']+1)
    evs=[spectrum(q) for q in qs]
    bwq=np.array(CONFIG['bandwidth_q']);bwB=TAU/bwq
    widths=np.array([np.ptp(spectrum(q)[spectrum(q)<0]) for q in bwq])
    tq=np.array(CONFIG['tail_q']);tail=[]
    area=g['pinned']['area'];lever=g['pinned']['coarea']
    for q in tq:
        e=spectrum(q);e=e[e<0];lo=e[e<-1][-1];hi=e[e>-1][0]
        nstar=np.sum(e>-1);frac=nstar-1+(hi+1)/(hi-lo)
        gam=(area*q/TAU**2-frac)%1
        tail.append(dict(q=int(q),gap=float(hi-lo),gamma=float(gam)))
    fqgap=bwq
    plotgaps=[]
    for q in fqgap:
        e=spectrum(q)
        plotgaps.append(float(e[e>-1][0]-e[e<-1][-1]))
    gaps=np.array([r['gap'] for r in tail])
    gam=np.array([r['gamma'] for r in tail])
    tailfits={}
    for qmin in [400,640,960]:
        s=tq>=qmin
        for deg in [1,2]:
            cf=np.polynomial.polynomial.polyfit(400/tq[s],gam[s],deg)
            tailfits[f'qmin{qmin}_degree{deg}']=dict(gamma_infinity=float(cf[0]),coefficients=cf.tolist())
    fit=np.linalg.lstsq(np.column_stack((bwB[-4:],bwB[-4:]**2)),widths[-4:],rcond=None)[0]
    F=area/TAU**2
    sel=(qs>=130)&(qs<=170);x=qs[sel].astype(float)
    y=np.array([grand(e,q,-1,CONFIG['waveform_T']) for e,q in zip(evs,qs)])[sel]
    comb=fq.joint_comb_fit(x,x*x*y,F)
    ed=(x*x*y-comb['background'])*x
    show=(x>=138)&(x<=162)
    def theory(q):
        B=TAU/q;X=2*np.pi**2*CONFIG['waveform_T']/(TAU*B**2/lever)
        return sum(4*np.pi/lever/p**2*(p*X/np.sinh(p*X))*np.cos(TAU*p*(F*q-g['pinned']['gamma'])) for p in range(1,5))
    xd=np.linspace(138,162,1200)
    np.savez_compressed(ROOT/'data/flat_observables.npz',bandwidth_q=bwq,bandwidth=widths,
        spacing_q=fqgap,spacing=np.array(plotgaps),moment_span=g['moment_max']-g['moment_min'],lever=lever,
        waveform_q=x[show],waveform_ed=ed[show],waveform_theory_at_ed=theory(x[show]),
        theory_q=xd,waveform_theory=theory(xd),gamma_q=tq,gamma_est=gam)
    out=dict(bandwidth_slope=float(fit[0]),bandwidth_slope_prediction=g['moment_max']-g['moment_min'],
        spacing_relative_error_at_q2560=float(gaps[-1]/(TAU*(TAU/2560)**2/lever)-1),
        spacing_log_slope=float(np.polyfit(np.log(TAU/tq[-4:]),np.log(gaps[-4:]),1)[0]),
        tail=tail,phase_fits=tailfits,
        waveform_rms_error_over_theory_rms=float(np.linalg.norm(ed[show]-theory(x[show]))/np.linalg.norm(theory(x[show]))),
        damping_pinned=damping(qs,evs,g,0.),damping_caption=damping(qs,evs,g,-.5))
    write_json(ROOT/'data/flat_analysis.json',out)
    return out

def crossover_analysis(g):
    d=np.load(ROOT/'data/sweep_crossover.npz');q=d['qs'];counts=d['Ncount'];ids=d['win_id']
    W=CONFIG['W'];rows=[];spectroscopy=[]
    for i,c in enumerate(d['centers']):
        sel=ids==i
        slope=np.polyfit(q[sel].astype(float),counts[sel],1)[0]
        qmean=float(np.mean(q[sel]));B=TAU/qmean
        geom=geo.scalars(geo.contour(B=B,W=W))
        rows.append(dict(q=qmean,B=B,window_points=int(sel.sum()),F_ED=float(TAU*(1-slope)),F_theory=geom['Fobs'],
                         A_over_2pi=geom['area']/TAU))
        original=sel&(np.abs(q-c)<=12)
        rows[-1]['F_ED_25_point_window']=float(TAU*(1-np.polyfit(q[original],counts[original],1)[0]))
        # Independent level positions: do not infer spacing or phase from the count slope.
        s=spc.near_mu(get_hops(),int(c),W)
        f=spc.near_mu(get_hops(),int(c),0.)
        s['gamma_est']=float((geom['area']*c/TAU**2-s['hole_fractional_index'])%1)
        f['gamma_est']=float((g['pinned']['area']*c/TAU**2-f['hole_fractional_index'])%1)
        s['gamma_pred']=geom['gamma'];f['gamma_pred']=g['pinned']['gamma']
        s['spacing_pred']=float(TAU*B/geom['coarea'])
        delta_meas=float(fq.wrap_unit(s['gamma_est']-f['gamma_est']))
        delta_pred=geom['gamma']-g['pinned']['gamma']
        spectroscopy.append(dict(q=int(c),B=B,cross=s,flat=f,
            delta_gamma_measured=delta_meas,delta_gamma_predicted=delta_pred,
            delta_gamma_error=float(fq.wrap_unit(delta_meas-delta_pred))))
        print(f'Crossover spectroscopy q={c}: spacing ratio={s["gap"]/s["spacing_pred"]:.5f}, phase difference error={spectroscopy[-1]["delta_gamma_error"]:.4g}',flush=True)
    out=dict(frequency=rows,spectroscopy=spectroscopy,
        median_frequency_absolute_error_per_q=float(np.median([abs(r['F_ED']-r['F_theory'])/TAU for r in rows])),
        max_frequency_absolute_error_per_q=float(np.max([abs(r['F_ED']-r['F_theory'])/TAU for r in rows])),
        max_differential_phase_error=float(max(abs(s['delta_gamma_error']) for s in spectroscopy)),
        max_expansion_parameter=float(TAU/q.min()*g['xi2_grid']),q_min_actual=int(q.min()),q_max_actual=int(q.max()))
    write_json(ROOT/'data/crossover_analysis.json',out)
    return out

def main():
    g=json.loads((ROOT/'data/geometry.json').read_text())
    chi=susceptibility(g);flat=flat_analysis(g);cross=crossover_analysis(g)
    from phase_audit import main as phase_run
    phase_run()
    print('Analysis complete. Midgap chi:',chi['mu_scan_T0.1'][-1],flush=True)
    print('Flat analysis:',{k:v for k,v in flat.items() if isinstance(v,float)},flush=True)
    print('Crossover:',{k:v for k,v in cross.items() if isinstance(v,float)},flush=True)

if __name__=='__main__':main()
