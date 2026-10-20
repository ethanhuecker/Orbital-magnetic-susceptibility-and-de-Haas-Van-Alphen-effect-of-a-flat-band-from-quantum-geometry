# Numerical reproduction and correctness audit

Paper: arXiv:2610.12193v1. Source: the supplied PDF and 18 Python files.

**47/47 numerical checks passed.** This establishes the accuracy of the
reconstructed two-band calculation at the tolerances below. It does not mean
every claim or caption in the paper is verified. The limitations are explicit.

## Reproduced numbers

| Quantity | Recomputed value | Comparison |
| --- | ---: | --- |
| Chern number, lower band | -1 | -1 |
| Integral I = average Omega squared | 0.0731586880597 | Paper: 0.0732 |
| Moment minimum at Gamma | -0.78125 | Paper: -0.78 |
| Moment maximum at M | 0.048828125 | Paper: +0.05 |
| Hole-pocket area A_M | 14.2031610877 | Recomputed, not the hard-coded plot constant |
| Frequency A_M/(2*pi) | 2.26050329463 | Paper: 2.26 |
| Coarea integral | 110.644357831 | Paper: approximately 110.65; supplied plot: 110.644 |
| Berry phase contribution gamma_1 | -0.0775342222491 | Paper: -0.078 |
| X contribution gamma_2 | 0.451594001971 | Paper: +0.452 |
| Total quantization phase | 0.874059779722 | Paper: 0.874 |
| Phase with X omitted | 0.422465777751 | Paper: 0.422 |
| Magnitude ratio of X and Berry phase terms | 5.824447436 | Paper: approximately 5.8 |
| ED phase extrapolation, q>=400, linear in 1/q | 0.874263627525 | Error 0.000204 period |
| Geometric length squared, sampled maximum | 1.577357314 | Paper: approximately 1.6 |
| Zero-field narrow-band frequency | 0.5350099464 | Low-field plateau |
| Chirp midpoint field | 0.01260557396 | Paper labels approximately 0.013 |
| Phase-content scale B_gamma | 0.1268015733 | Outside Fig. 5; separate dense ED below |
| Strong-field contour coefficient Lambda_g | 135.445931017 | Eq. (75), evaluated independently |

## Quantitative agreement

- Weak-field bandwidth coefficient: ED 0.8301142138, geometry
  0.830078125; relative difference
  4.35e-05.
- Pinned-mu level spacing at q=2560 differs by
  0.08588%; fitted small-field exponent
  1.9985146 rather than exactly 2.
- Maximum susceptibility relative error across the eight chemical potentials:
  3.539e-05 (0.003539%). Fits to a second field window were also checked.
- Maximum absolute error in the linear-coefficient magnetization comparison:
  4.833e-09.
- The unadjusted four-harmonic LK waveform has relative RMS error
  7.038% on the displayed points.
  This is a finite-field comparison, not an exact equality or an amplitude fit.
- Damping maximum relative error where the theoretical normalized amplitude
  exceeds 0.05: pinned protocol
  1.629%; caption protocol
  4.497%.
- Crossover frequency median absolute error per unit q:
  0.00056145; maximum
  0.0059035. The original 25-point-only
  windows give median 0.0049102. Frequency per unit 1/B is larger by 2*pi.
- Differential crossover phase error is at most
  0.0042843 period. The predicted geometric phase
  changes by approximately 0.3746 period over the sampled range.
- The separate phase-scale test uses paired full spectra for every integer
  q=40--100.
  Its differential phase error is at most
  0.012529 period. All 3
  crossings of the measured contamination with the predicted geometric content
  are retained: the discrete brackets together span B=0.1208305--0.1282283,
  with linearly interpolated crossings at 0.125760401, 0.124771367, 0.123086161.
  Their maximum relative difference from B_gamma is
  2.93%. There is no uniquely resolved
  crossing at this discrete level-interpolation precision.

The raw calculation contains 279 complete flat-model spectra through q=2560,
1214 narrow-band inertia counts, and paired near-mu eigensolutions
at 18 crossover centers through q=17000. Only the level counts and selected
near-mu eigenpairs, not full 34000-dimensional spectra, are computed at the
largest q. The additional phase-scale audit saves 122 complete
spectra in `data/phase_scale_spectra.npz`. These operations are clearly
distinguished in the data files.

Inertia methods used: complex_unpivoted_LDLH: 1021; real_symmetric_doubled_LDLT: 193. The real symmetric fallback checks doubled
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
   25, 33, 45, 61, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75. They implement the caption's
   25--75-point description with width capped at one quarter of the center q.
   They are not represented as the unavailable original extension's exact grid.
7. **Expansion-window statement.** The first window extends to q=88,
   giving max B*xi^2=0.112623, slightly beyond 0.10.
   At the q=100 center it is approximately 0.099. The count sweep ends at
   q=17037 (the caption rounds this to 1.7e4).
8. **Separate phase-scale coverage.** B_gamma=0.126802 is larger than
   the largest sampled field, 0.0713998. Its formula is
   evaluated, and the phase migration inside the chirp is checked. The paper
   explicitly says that Fig. 5 does not test this phase scale and describes a
   separate phase measurement on page 14. This package reconstructs that test
   on the explicit grid in `phase_config.json`, yielding the crossing region
   above. Its maximum B*xi^2 is 0.24777; those
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
| Chern number -1 | 1.11022e-16 | 1e-10 | PASS |
| BZ integral of U2 vanishes | -7.97973e-17 | 1e-10 | PASS |
| I convergence 512 to 1024 | -1.38778e-17 | 1e-11 | PASS |
| X Eq.35 vs independent eigenvector Eq.17 | 2.39551e-10 | 2e-07 | PASS |
| Eigenvector finite-difference step convergence | 2.23329e-10 | 2e-07 | PASS |
| Constant Hamiltonian energy-shift invariance | 3.03489e-11 | 2e-07 | PASS |
| Smooth eigenvector gauge invariance | 1.03259e-11 | 2e-07 | PASS |
| Berry curvature analytic vs projector derivatives | 1.11022e-16 | 1e-12 | PASS |
| FFT hopping convergence 512 to 1024 | 1.11025e-16 | 1e-13 | PASS |
| Largest discarded Fourier coefficient outside radius 30 | 3.06006e-13 | 1e-12 | PASS |
| Zero-field Fourier reconstruction | 1.54885e-12 | 1e-10 | PASS |
| Harper phases vs independent per-hop assembly | 8.88281e-16 | 1e-12 | PASS |
| Sparse vs dense Harper matrices | 1.11022e-16 | 1e-12 | PASS |
| Complex/real inertia counts vs dense ED | 0 | 0 | PASS |
| Independent dense LAPACK drivers | 2.88658e-15 | 1e-11 | PASS |
| Shift-invert spacing vs full diagonalization | 8.75944e-12 | 1e-07 | PASS |
| Magnetic Bloch dispersion near pinned mu / spacing | 6.5748e-12 | 1e-06 | PASS |
| Radius 30 to 36 convergence at q=2560 | -7.13394e-09 | 1e-05 | PASS |
| Pocket area polar vs independent Cartesian quadrature | -1.77636e-15 | 1e-08 | PASS |
| Coarea polar vs independent Cartesian quadrature | -2.17e-11 | 1e-07 | PASS |
| Contour quadrature 256 to 1024 angles | 1.42109e-14 | 1e-08 | PASS |
| Fixed moment contour area is independent of B | 6.40632e-16 | 1e-10 | PASS |
| Composite area derivative vs field finite difference | 9.37681e-09 | 1e-07 | PASS |
| No stale spectrum cache | 0 | 0 | PASS |
| Full sweep Hermiticity | 4.96507e-16 | 1e-11 | PASS |
| No stale inertia cache | 0 | 0 | PASS |
| All count solve backward errors | 3.50402e-13 | 1e-09 | PASS |
| Paper I=0.0732 (rounding interval) | -4.13119e-05 | 5e-05 | PASS |
| Paper gamma=0.874 (rounding interval) | 5.97797e-05 | 0.0005 | PASS |
| Paper frequency=2.26 (rounding interval) | 0.000503295 | 0.005 | PASS |
| Bandwidth weak-field slope relative error | 4.34764e-05 | 0.002 | PASS |
| Spacing relative error at q=2560 | 0.000858765 | 0.003 | PASS |
| Spacing B exponent | -0.00148541 | 0.01 | PASS |
| Phase extrapolation q>=400 linear vs geometry | 0.000203848 | 0.001 | PASS |
| Susceptibility relative errors | 3.53929e-05 | 0.0001 | PASS |
| Susceptibility independent fit window stability | 1.1194e-06 | 2e-06 | PASS |
| Smooth magnetization absolute error | 4.83251e-09 | 1e-07 | PASS |
| Pinned waveform relative RMS error | 0.0703765 | 0.1 | PASS |
| Pinned damping max relative error above 0.05 | 0.0162919 | 0.05 | PASS |
| Caption damping max relative error above 0.05 | 0.0449677 | 0.1 | PASS |
| Crossover count frequency median error per q | 0.00056145 | 0.015 | PASS |
| Crossover differential phase max absolute error | 0.00428428 | 0.02 | PASS |
| Crossover weak-field spacing exponent near 1 | -0.00211109 | 0.15 | PASS |
| Crossover spacing exponent rises toward 2 | 0 | 0 | PASS |
| No stale separate phase-scale data | 0 | 0 | PASS |
| Phase-scale differential phase absolute error | 0.0125289 | 0.03 | PASS |
| All measured phase-scale crossings vs Eq.77 | 0.029301 | 0.05 | PASS |

All errors, raw data, fit parameters, and environment versions are available
as machine-readable JSON/NPZ. SHA256 checksums identify the delivered files.
