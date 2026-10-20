# Numerical Calculations arXiv:2610.12193v1

Numerical construction of the five figures and the accompanying two-band numerical checks in ArXiv:2610.12193. 

Contains the code and parameters needed to rerun the paper's numerical results.

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

The archive does not include the raw data. To recompute them, add `--force`.
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

## Figure and data map

| Paper result | Generated figure |
| --- | --- |
| Fig. 1: orbital moment and zero contour | `figures/figure_1_moment.png` |
| Fig. 2: bandwidth and spacing | `figures/figure_2_spectrum.png` |
| Fig. 3: susceptibility | `figures/figure_3_susceptibility.png` |
| Fig. 4  damping protocol | `figures/figure_4_caption_protocol.png` | 
| Fig. 5: narrow-band crossover | `figures/figure_5_crossover.png` | 

Every figure is also saved as SVG. No PDF-reading software is needed to run the
calculation. The five raster images extracted from the provided paper are kept
in `reference/published_figures/` solely for visual comparison; none is an input
to the numerical computation or regenerated figures.

The figure generator produces the five numbered paper figures and the separately
named Figure 4 caption-protocol comparison described above. Phase extrapolation,
spacing, and phase-content checks remain in the numerical data and verification
suite. 

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


See `NOTICE` for retained source attribution.
