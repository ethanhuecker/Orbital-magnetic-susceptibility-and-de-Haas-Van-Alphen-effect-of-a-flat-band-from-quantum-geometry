#!/usr/bin/env python3
"""One-command regeneration of all five paper figures and validation results.

python reproduce.py --stage all --workers 4
Stages: geometry, spectra, counts, analyze, figures, verify, all.
Numerical outputs are regenerated from the Hamiltonian, never from paper targets.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy
from scipy.linalg import eigh
import flatqwz as fq
import geometry as geo
import sparse_spectrum as spc

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
_HOPS=None

def fingerprint():
    h=hashlib.sha256()
    for name in ['config.json','flatqwz.py','qwz_cross.py','geometry.py','sparse_spectrum.py','reproduce.py']:
        h.update(name.encode());h.update((ROOT/name).read_bytes())
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def get_hops():
    global _HOPS
    if _HOPS is None:
        _HOPS=fq.build_hoppings(N=CONFIG['fft_N'],r=CONFIG['hopping_radius'],u=CONFIG['u'])
    return _HOPS

def geometry_stage():
    g=geo.global_geometry(1024,CONFIG['u'])
    g['pinned']=geo.scalars(geo.contour(ntheta=CONFIG['contour_angles']))
    g['inband']=geo.scalars(geo.contour(m0=-.5,hole=False,ntheta=CONFIG['contour_angles']))
    g['hopping_tail_l1']=get_hops()['tail']
    g['source_fingerprint']=fingerprint()
    write_json(ROOT/'data/geometry.json',g)
    np.savez_compressed(ROOT/'data/theory_crossover.npz',**geo.crossover_theory(CONFIG['W']))
    print('Geometry:',json.dumps(g['pinned']),flush=True)

def spectrum_job(q):
    q=int(q)
    H=fq.harper_matrix(get_hops(),q,CONFIG['k1'],CONFIG['k2'])
    herm=float(np.max(np.abs(H-H.conj().T)))
    if herm>1e-11: raise ArithmeticError(f'Hermiticity {herm}')
    # evr and evd are independent LAPACK drivers; small-q validation compares both.
    ev=eigh(H,eigvals_only=True,driver='evr',overwrite_a=True,check_finite=False)
    if np.sum(ev<0)!=q-1: raise ArithmeticError('Unexpected magnetic cluster count')
    fn=ROOT/'data/spectra'/f'q{q:05d}.npz'
    np.savez_compressed(fn,evals=ev,q=q,hermiticity=herm,source_fingerprint=fingerprint())
    return q,herm

def count_job(item):
    q,window=item
    info=spc.count_levels(get_hops(),int(q),CONFIG['W'],CONFIG['k1'],CONFIG['k2'])
    info.update(q=int(q),window=int(window),source_fingerprint=fingerprint())
    write_json(ROOT/'data/count_cache'/f'q{q:05d}.json',info)
    return info

def run_jobs(func,jobs,workers,label):
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(func,j):j for j in jobs}
        for i,f in enumerate(as_completed(futures),1):
            result=f.result()
            if i%25==0 or i==len(jobs):
                print(f'{label}: {i}/{len(jobs)} completed in {time.time()-start:.1f}s',flush=True)

def spectra_stage(workers,force=False):
    directory=ROOT/'data/spectra';directory.mkdir(parents=True,exist_ok=True)
    qs=sorted(set(range(CONFIG['flat_q_min'],CONFIG['flat_q_max']+1))|
              set(CONFIG['bandwidth_q'])|set(CONFIG['tail_q'])|set(CONFIG['chi_q']))
    jobs=[]
    for q in reversed(qs):
        fn=directory/f'q{q:05d}.npz'
        if not force and fn.exists():
            with np.load(fn) as d:
                if str(d['source_fingerprint'])==fingerprint(): continue
        jobs.append(q)
    run_jobs(spectrum_job,jobs,workers,'Spectra')
    flatqs=np.arange(CONFIG['flat_q_min'],CONFIG['flat_q_max']+1)
    pad=np.full((len(flatqs),2*flatqs[-1]),np.nan)
    for i,q in enumerate(flatqs):
        with np.load(directory/f'q{q:05d}.npz') as d: pad[i,:2*q]=d['evals']
    np.savez_compressed(ROOT/'data/sweep_flat.npz',qs=flatqs,evals_pad=pad,u=CONFIG['u'],
                        source_fingerprint=fingerprint())

def counts_stage(workers,force=False):
    directory=ROOT/'data/count_cache';directory.mkdir(parents=True,exist_ok=True)
    centers=np.unique(np.geomspace(CONFIG['count_centers_min'],CONFIG['count_centers_max'],CONFIG['count_windows']).astype(int))
    # The supplied extension cache is absent. Reconstruct explicit odd windows:
    # at most 75 points, at most q_center/4, and at least the original 25.
    # This resolves count-staircase noise without substantially averaging the chirp.
    widths=[max(25,min(CONFIG['count_width'],int(c*CONFIG['count_relative_width_cap']))) for c in centers]
    widths=[w if w%2 else w-1 for w in widths]
    items=[(q,i) for i,(c,w) in enumerate(zip(centers,widths)) for q in range(c-w//2,c+w//2+1)]
    jobs=[]
    for q,i in items:
        p=directory/f'q{q:05d}.json'
        if not force and p.exists() and json.loads(p.read_text()).get('source_fingerprint')==fingerprint():continue
        jobs.append((q,i))
    run_jobs(count_job,list(reversed(jobs)),workers,'Inertia counts')
    rows=[json.loads((directory/f'q{q:05d}.json').read_text()) for q,i in items]
    np.savez_compressed(ROOT/'data/sweep_crossover.npz',qs=[r['q'] for r in rows],
        Ncount=[r['negative'] for r in rows],win_id=[r['window'] for r in rows],
        centers=centers,widths=widths,u=CONFIG['u'],delta=CONFIG['W'],dmu=3.5*CONFIG['W'],
        hermiticity=[r['hermiticity'] for r in rows],
        pivot_imag_relative=[r['pivot_imag_relative'] for r in rows],
        backward_error=[r['backward_error'] for r in rows],source_fingerprint=fingerprint())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','geometry','spectra','counts','analyze','figures','verify'],default='all')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--force',action='store_true',help='Recompute cached spectra/counts')
    a=p.parse_args()
    if not 1<=a.workers<=8: p.error('Use 1..8 workers; large spectra need memory per worker')
    for d in ['data','figures','validation']:(ROOT/d).mkdir(exist_ok=True)
    stages=['geometry','spectra','counts','analyze','figures','verify'] if a.stage=='all' else [a.stage]
    for stage in stages:
        print(f'Running {stage}',flush=True)
        if stage=='geometry':geometry_stage()
        elif stage=='spectra':spectra_stage(a.workers,a.force)
        # The doubled-real fallback is memory intensive at q=17000.
        elif stage=='counts':counts_stage(min(a.workers,2),a.force)
        elif stage=='analyze':
            from analyze import main as run
            run()
        elif stage=='figures':
            from plot_figures import main as run
            run()
        elif stage=='verify':
            from verify import main as run
            run()
    write_json(ROOT/'validation/environment.json',dict(python=os.sys.version,numpy=np.__version__,
        scipy=scipy.__version__,source_fingerprint=fingerprint(),configuration=CONFIG))

if __name__=='__main__':main()
