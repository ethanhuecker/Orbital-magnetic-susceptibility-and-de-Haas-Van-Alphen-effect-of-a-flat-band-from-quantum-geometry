# Numerical Calculations arXiv:2610.12193v1

Numerical construction of the five figures and the accompanying two-band numerical checks in ArXiv:2610.12193. 

Contains the code, parameters, and computed data needed to rerun the paper's numerical results.

## Run

Tested with Python 3.12.14. On Windows, in this extracted directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py --stage all --workers 4
```

On Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --stage all --workers 4
```

Use `--workers 1` on a machine with limited RAM. Four workers were tested in an
8 GB environment; the memory-intensive count stage is capped at two concurrent
workers even when four are requested. BLAS thread counts are limited to avoid
oversubscription. Process entry points are guarded for Windows.

The archive includes the computed raw data. Cached spectra/counts are accepted
only when their source fingerprint matches. To recompute them, add `--force`.
The first full run takes several minutes on the tested machine; hardware and
LAPACK/BLAS implementation can substantially change this.

For a quick redraw, after activating the environment:

```bash
python reproduce.py --stage figures
```

For independent checks using the supplied data:

```bash
python reproduce.py --stage verify
```

Each stage can also be run separately, in order:
`geometry`, `spectra`, `counts`, `analyze`, `figures`, `verify`.
`validation/checks.json` contains every test, its value, tolerance, and status.
A failed check returns a nonzero exit code. `validation/reproduction_run.log`
records the final successful run. It reused computed spectra and counts after
checking their source fingerprints, then regenerated the analysis, separate
phase audit, figures, and independent checks. `validation/generation_run.log`
records the preceding run that completed the resumed raw-count calculation.
`validation/figure_update_run.log` records the current figure generation after
removing plots that are not figures in the paper.

## Figure and data map

| Paper result | Generated figure | Recomputed inputs |
| --- | --- | --- |
| Fig. 1: orbital moment and zero contour | `figures/figure_1_moment.png` | Analytic Bloch vector and Berry curvature |
| Fig. 2: bandwidth and spacing | `figures/figure_2_spectrum.png` | `data/spectra/q*.npz`, `data/flat_observables.npz` |
| Fig. 3: susceptibility | `figures/figure_3_susceptibility.png` | Full spectra at q=320--1200; `data/susceptibility_raw.npz` and `.json` |
| Fig. 4: pinned waveform and supplied plotting-script damping protocol | `figures/figure_4_dhva.png` | q=60--320 spectra; `data/flat_analysis.json` |
| Fig. 4 caption's different damping protocol | `figures/figure_4_caption_protocol.png` | Same spectra, mu=-1-B/2; independently computed contour geometry |
| Fig. 5: narrow-band crossover | `figures/figure_5_crossover.png` | `data/sweep_crossover.npz`, `data/theory_crossover.npz` |

Every figure is also saved as SVG. No PDF-reading software is needed to run the
calculation. The five raster images extracted from the provided paper are kept
in `reference/published_figures/` solely for visual comparison; none is an input
to the numerical computation or regenerated figures.

The figure generator produces the five numbered paper figures and the separately
named Figure 4 caption-protocol comparison described above. Phase extrapolation,
spacing, and phase-content checks remain in the numerical data and verification
suite; they do not produce additional figures. Their results are in
`data/flat_analysis.json`, `data/crossover_analysis.json`, and
`data/phase_scale_analysis.json`, with raw phase spectra in
`data/phase_scale_spectra.npz`.

## Conventions and parameters

Model and sweep parameters are explicit in `config.json` and `phase_config.json`.

- Units: e=hbar=a=1, positive B=2*pi/q, flat-band energies +/-1, u=-1.2.
- Fourier hoppings: 1024 x 1024 FFT, square cutoff max(|rx|,|ry|)<=30,
  straight-line Peierls phase, magnetic-cell twists k1=0.51, k2=1.13.
- With these conventions the separated lower magnetic cluster contains q-1
  levels, consistent with C=-1. Each level has weight 1/q per site. Selecting
  q lower levels would incorrectly include a remote-band level.
- The coefficient `flatqwz.c2_flat` is X/2: the energy is
  -1+B*M+B^2*c2. The new geometry module independently evaluates Eq. (35) with
  analytic derivatives and verifies it against the eigenvector evaluator.
- All contour areas, Berry fluxes, spacing coefficients, and phases are
  recomputed by polar quadrature. There are no fitted or hard-coded paper values
  in the theory curves. Cartesian quadrature and grid refinement check the
  contour calculations independently.
- Fig. 4(a): mu=-1, T=5e-6, background fit on q=130--170, display q=138--162.
  The joint background-plus-harmonic fit is the supplied script's extraction
  method. Theory uses four harmonics and no phase/amplitude adjustment.
- Fig. 4(b) has an unresolved source/caption inconsistency. The supplied final
  plotting script uses pinned mu=-1, whereas the PDF caption says mu=-1-B/2.
  Both are generated and clearly identified above. Temperatures are different
  even when their dimensionless X values coincide.
- Narrow band: H=normalized(d).sigma+W*(2-cos(kx)-cos(ky))*identity,
  W=2.79e-4 and mu=-1+3.5W. The caption rounds W to 2.8e-4. Here W is the
  coefficient of g; the actual zero-field bandwidth is **4W**.
- Crossover centers: 18 logarithmically spaced integers from 100 to 17000.
  The missing extension is reconstructed with odd window widths equal to the
  smaller of 75 and floor(q_center/4), with a minimum of 25. Thus the widths
  are 25, 33, 45, 61, then 75. Every raw q and window ID is saved. This is an
  explicit reconstruction of the caption's 25--75-point prescription, not a
  claim to know the missing original extension's exact grid.
- The original 25-point frequency estimate is also retained for every window
  in `data/crossover_analysis.json`. This exposes staircase-resolution effects.
- Susceptibility uses a free quartic fit in B to the full two-band grand
  potential at T=0.1. No theory coefficient is constrained in the fit. The
  q grid within the paper's stated 320--1200 interval is explicitly reconstructed
  because its original cache/generator was absent.
- The separate phase-content test uses paired full dense spectra at every
  integer q=40--100. The contamination in a constant-frequency intercept is
  the difference of fractional hole-level indices between W=0 and W>0, with
  no modulo choice. Every crossing bracket is retained, including the small
  oscillations caused by linear interpolation between levels. This additional
  reconstructed sweep extends beyond the Fig. 5 field window.

## Required source files

| File | Purpose |
| --- | --- |
| `reproduce.py` | Entry point, generation, caching, multiprocessing |
| `config.json`, `phase_config.json`, `requirements.txt` | Model/sweep settings and pinned dependencies |
| `flatqwz.py` | Supplied dense Harper and eigenvector-geometry implementation; paper defaults corrected |
| `qwz_cross.py` | Supplied scalar-dispersion extension; used in independent checks |
| `geometry.py` | Analytic derivatives, pocket quadrature, crossover theory |
| `sparse_spectrum.py` | Sparse Harper construction, checked inertia, near-mu eigenpairs |
| `analyze.py` | Thermodynamics, damping, phase and frequency extraction |
| `phase_audit.py` | Full dense spectra for the separate phase-content test |
| `plot_figures.py`, `paper3_style.py` | Figure assembly and supplied publication style |
| `verify.py`, `report.py` | Independent implementation/convergence/observable checks and the generated audit |

`reference/input_sha256.json` records all supplied input files by hash.
`reference/flatqwz_changes.patch` records the minimal changes to the supplied
core. The other old launchers and plotters are not required; they depended on
missing caches, obsolete parameters, or auxiliary unpublished figures.

## Limits of verification

This validates the stated noninteracting two-band lattice calculation. It does
not prove the paper's general multiband derivation. Finite-field ED and the
truncated weak-field theory are not exactly equal; deviations are reported.
One magnetic Bloch point is used as in the supplied code, with an explicit
twist-dependence check near the measured contour.

The paper's B_gamma scale is computed from zero-field integrals and lies
beyond the largest field in the Fig. 5 count sweep, as the paper acknowledges.
The separate phase-scale experiment described on page 14 is reconstructed at
q=40--100 and approximately confirms the predicted crossing. Its discrete
brackets and finite-field error are reported, and its larger expansion parameter
is not confused with the Fig. 5 weak-field window.
Likewise the x<=0.10 caption describes the centers approximately: the actual
first 25-point window reaches q=88, where x is approximately 0.113.

The sparse code does not assume that arbitrary LU pivot signs give inertia.
It checks permutations and factor accuracy, and switches to a doubled real
symmetric representation when complex pivots are numerically unreliable.
See the official SciPy documentation for the underlying solver contracts:
[splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)
and [SuperLU](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html).

See `NOTICE` for retained source attribution.
