"""Generate the human-readable audit from the actual saved validation results."""
import json
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def write_report(validation=None):
    read=lambda name:json.loads((ROOT/name).read_text())
    v=validation or read('validation/checks.json')
    g=read('data/geometry.json');f=read('data/flat_analysis.json')
    c=read('data/crossover_analysis.json');chi=read('data/susceptibility.json')
    phase=read('data/phase_scale_analysis.json')
    th=np.load(ROOT/'data/theory_crossover.npz');counts=np.load(ROOT/'data/sweep_crossover.npz')
    passed=sum(r['passed'] for r in v['checks']);total=len(v['checks'])
    checkrows='\n'.join(f"| {r['name']} | {r['value']:.6g} | {r['tolerance']:.6g} | {'PASS' if r['passed'] else 'FAIL'} |" for r in v['checks'])
    chirel=max(abs(r['chi_ED']/r['chi_closed']-1) for r in chi['mu_scan_T0.1'])
    gammaed=f['phase_fits']['qmin400_degree1']['gamma_infinity']
    methods='; '.join(f'{k}: {n}' for k,n in v['count_methods'].items())
    rows=c['spectroscopy'];delta=[r['delta_gamma_predicted'] for r in rows]
    base25=np.median([abs(r['F_ED_25_point_window']-r['F_theory'])/(2*np.pi) for r in c['frequency']])
    crossings=phase['crossings']
    bc=[r['B_interpolated'] for r in crossings]
    bracket_min=min(r['B_bracket'][0] for r in crossings)
    bracket_max=max(r['B_bracket'][1] for r in crossings)
    text=f'''# Numerical reproduction and correctness audit

Paper: arXiv:2610.12193v1. Source: the supplied PDF and 18 Python files.

**{passed}/{total} numerical checks passed.** This establishes the accuracy of the
reconstructed two-band calculation at the tolerances below. It does not mean
every claim or caption in the paper is verified. The limitations are explicit.

## Reproduced numbers

| Quantity | Recomputed value | Comparison |
| --- | ---: | --- |
| Chern number, lower band | {g['chern']:.12g} | -1 |
| Integral I = average Omega squared | {g['I']:.12g} | Paper: 0.0732 |
| Moment minimum at Gamma | {g['moment_min']:.12g} | Paper: -0.78 |
| Moment maximum at M | {g['moment_max']:.12g} | Paper: +0.05 |
| Hole-pocket area A_M | {g['pinned']['area']:.12g} | Recomputed, not the hard-coded plot constant |
| Frequency A_M/(2*pi) | {g['pinned']['Fobs']:.12g} | Paper: 2.26 |
| Coarea integral | {g['pinned']['coarea']:.12g} | Paper: approximately 110.65; supplied plot: 110.644 |
| Berry phase contribution gamma_1 | {g['pinned']['gamma1']:.12g} | Paper: -0.078 |
| X contribution gamma_2 | {g['pinned']['gamma2']:.12g} | Paper: +0.452 |
| Total quantization phase | {g['pinned']['gamma']:.12g} | Paper: 0.874 |
| Phase with X omitted | {0.5+g['pinned']['gamma1']:.12g} | Paper: 0.422 |
| Magnitude ratio of X and Berry phase terms | {abs(g['pinned']['gamma2']/g['pinned']['gamma1']):.10g} | Paper: approximately 5.8 |
| ED phase extrapolation, q>=400, linear in 1/q | {gammaed:.12g} | Error {abs(gammaed-g['pinned']['gamma']):.3g} period |
| Geometric length squared, sampled maximum | {g['xi2_grid']:.10g} | Paper: approximately 1.6 |
| Zero-field narrow-band frequency | {float(th['F_nat'])*2*np.pi:.10g} | Low-field plateau |
| Chirp midpoint field | {float(th['Bmid']):.10g} | Paper labels approximately 0.013 |
| Phase-content scale B_gamma | {float(th['Bgamma']):.10g} | Outside Fig. 5; separate dense ED below |
| Strong-field contour coefficient Lambda_g | {g['pinned']['Lambda_g']:.12g} | Eq. (75), evaluated independently |

## Quantitative agreement

- Weak-field bandwidth coefficient: ED {f['bandwidth_slope']:.10g}, geometry
  {f['bandwidth_slope_prediction']:.10g}; relative difference
  {abs(f['bandwidth_slope']/f['bandwidth_slope_prediction']-1):.3g}.
- Pinned-mu level spacing at q=2560 differs by
  {100*abs(f['spacing_relative_error_at_q2560']):.4g}%; fitted small-field exponent
  {f['spacing_log_slope']:.8g} rather than exactly 2.
- Maximum susceptibility relative error across the eight chemical potentials:
  {chirel:.4g} ({100*chirel:.4g}%). Fits to a second field window were also checked.
- Maximum absolute error in the linear-coefficient magnetization comparison:
  {max(abs(r['M_ED']-r['M_closed']) for r in chi['mu_scan_T0.1']):.4g}.
- The unadjusted four-harmonic LK waveform has relative RMS error
  {100*f['waveform_rms_error_over_theory_rms']:.4g}% on the displayed points.
  This is a finite-field comparison, not an exact equality or an amplitude fit.
- Damping maximum relative error where the theoretical normalized amplitude
  exceeds 0.05: pinned protocol
  {100*f['damping_pinned']['max_relative_error_pred_gt_005']:.4g}%; caption protocol
  {100*f['damping_caption']['max_relative_error_pred_gt_005']:.4g}%.
- Crossover frequency median absolute error per unit q:
  {c['median_frequency_absolute_error_per_q']:.5g}; maximum
  {c['max_frequency_absolute_error_per_q']:.5g}. The original 25-point-only
  windows give median {base25:.5g}. Frequency per unit 1/B is larger by 2*pi.
- Differential crossover phase error is at most
  {c['max_differential_phase_error']:.5g} period. The predicted geometric phase
  changes by approximately {max(delta)-min(delta):.4g} period over the sampled range.
- The separate phase-scale test uses paired full spectra for every integer
  q={phase['configuration']['q_min']}--{phase['configuration']['q_max']}.
  Its differential phase error is at most
  {phase['max_differential_phase_error']:.5g} period. All {len(crossings)}
  crossings of the measured contamination with the predicted geometric content
  are retained: the discrete brackets together span B={bracket_min:.7g}--{bracket_max:.7g},
  with linearly interpolated crossings at {', '.join(f'{b:.9g}' for b in bc)}.
  Their maximum relative difference from B_gamma is
  {100*phase['max_crossing_relative_error']:.4g}%. There is no uniquely resolved
  crossing at this discrete level-interpolation precision.

The raw calculation contains 279 complete flat-model spectra through q=2560,
{len(counts['qs'])} narrow-band inertia counts, and paired near-mu eigensolutions
at 18 crossover centers through q=17000. Only the level counts and selected
near-mu eigenpairs, not full 34000-dimensional spectra, are computed at the
largest q. The additional phase-scale audit saves {2*len(phase['rows'])} complete
spectra in `data/phase_scale_spectra.npz`. These operations are clearly
distinguished in the data files.

Inertia methods used: {methods}. The real symmetric fallback checks doubled
eigenvalue multiplicity, symmetric permutations, a factor-congruence probe,
and solve backward error. Dense diagonalization checks both count methods
at small and intermediate q.

## Findings that prevent a claim of exact original-file recovery

1. **Wrong default model in supplied code.** `flatqwz.py` fixes u=-1.5, but the
   production paper uses -1.2. Several launchers set an environment variable
   that this file ignores. The related `flatqwz_raw.py` honors it, but is not
   the imported module. This package fixes the paper default explicitly.
2. **Missing generators/data.** The supplied final plotters require
   `fig_shared.npz`, `chi_ed_local.json`, a crossover theory file, and an
   extended crossover count file that were absent. Their contents have been
   recomputed. The exact original susceptibility q list, spacing sample list,
   and extension window list cannot be recovered from the provided material.
3. **Figure 4(b) source/caption mismatch.** `make_fig3.py` explicitly computes
   pinned mu=-1, whereas the PDF says mu=-1-B/2. The main reproduction follows
   the supplied final plotting source; the separate caption-protocol figure
   evaluates the condition actually printed in the caption. The two use
   different physical temperatures at the same plotted X.
4. **Magnetic cluster size.** The paper says q lower-band levels, but this
   implementation has q-1 for positive B and C=-1, confirmed by diagonalizing
   every saved spectrum. The phase-space state count 1+C/q gives the same
   result. Keeping q states would introduce a remote-band state.
5. **Truncation and crossover settings.** The core supplied default radius is
   24, while the paper specifies 30. This package uses 30 and checks 36.
   The older `run_crossover3.py` uses W=0.002, which is not the Fig. 5 value.
   This package uses the supplied count script's W=2.79e-4 (rounded to 2.8e-4
   in the caption). The physical bandwidth of g=2-cos(kx)-cos(ky) is 4W.
6. **Window reconstruction.** The final count widths are saved explicitly:
   {', '.join(map(str,counts['widths'].tolist()))}. They implement the caption's
   25--75-point description with width capped at one quarter of the center q.
   They are not represented as the unavailable original extension's exact grid.
7. **Expansion-window statement.** The first window extends to q={c['q_min_actual']},
   giving max B*xi^2={c['max_expansion_parameter']:.6g}, slightly beyond 0.10.
   At the q=100 center it is approximately 0.099. The count sweep ends at
   q={c['q_max_actual']} (the caption rounds this to 1.7e4).
8. **Separate phase-scale coverage.** B_gamma={float(th['Bgamma']):.6g} is larger than
   the largest sampled field, {2*np.pi/c['q_min_actual']:.6g}. Its formula is
   evaluated, and the phase migration inside the chirp is checked. The paper
   explicitly says that Fig. 5 does not test this phase scale and describes a
   separate phase measurement on page 14. This package reconstructs that test
   on the explicit grid in `phase_config.json`, yielding the crossing region
   above. Its maximum B*xi^2 is {phase['max_expansion_parameter']:.5g}; those
   fields are outside Fig. 5's x<=0.10 window. The unavailable original
   phase-measurement grid and its interpolation choices are not recovered.

These discrepancies are not silently repaired in the paper or hidden in the
figures. Both damping protocols and the original 25-point frequency estimates
are retained, and every reconstruction choice is documented in `README.md`.

## Independent checks

These include a second formula for X, gauge and energy-shift invariance,
independent Fourier/Harper assembly, two dense eigensolvers, sparse-versus-dense
counts and eigenpairs, hopping and momentum-grid convergence, Bloch-twist
dependence, and an independent Cartesian calculation of the moment contour.
The checks compare theory against ED; they do not adjust theory to fit ED.

| Check | Measured error/value | Allowed magnitude | Status |
| --- | ---: | ---: | --- |
{checkrows}

All errors, raw data, fit parameters, and environment versions are available
as machine-readable JSON/NPZ. SHA256 checksums identify the delivered files.
'''
    (ROOT/'VERIFICATION.md').write_text(text)

if __name__=='__main__':write_report()
