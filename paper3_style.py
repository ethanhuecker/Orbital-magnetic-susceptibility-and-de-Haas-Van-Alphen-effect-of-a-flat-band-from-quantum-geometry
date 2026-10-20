# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""paper3_style.py -- shared figure style for ALL paper3 (flat-band dHvA) figures.

Base: the figure-style skill's apply_figure_style() rc set (inlined below so the
scripts are standalone), then the paper pins:
  - ONE theory-line color C_THEORY for every 'no-fit theory' line/curve.
  - ONE ED marker: filled circle, color C_ED, mec white.  Open circles are
    reserved for extrapolation-tail subsets ONLY (fig 5 tail to q=2560).
  - Fit-to-ED lines are drawn in C_ED (thin), guides/limiting laws in C_GUIDE.
  - NO in-axes titles, NO grids, open spines, sizes (8, 7, 6) pt.
  - Panel labels: bold lowercase letters OUTSIDE the axes, upper-left.
Print sizes (REVTeX PRB): \\columnwidth = 3.404 in, \\textwidth = 7.056 in.
"""
import matplotlib as mpl

# ---------------------------------------------------------------- palette
C_THEORY = "#c44e52"   # no-fit theory lines everywhere
C_ED     = "#1f5fa8"   # ED marker fill and fit-to-ED lines
C_GUIDE  = "0.35"      # power-law guides / limiting laws
C_META   = "#888888"   # plateau labels, secondary annotations
# categorical hues (fig 6a field-contours; threaded to the dots in 6b)
C_FIELD  = ["#2a9d8f", "#1c1c1c", "#E0A100"]     # B->0, B=B*, B=5B*

COLW  = 3.404          # PRB \columnwidth in inches
TEXTW = 7.056          # PRB \textwidth in inches

ED_KW   = dict(marker="o", color=C_ED, mec="white", ls="none")   # base ED marker
TAIL_KW = dict(marker="o", mfc="white", mec=C_ED, ls="none")     # extrapolation tails ONLY


def apply_paper3_style():
    """figure-style skill base rc + paper pins.  Call before any plotting."""
    base, secondary, tick = 8, 7, 6
    rc = {
        # --- figure-style skill base (frame='open', sizes=(8,7,6), no grid) ---
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # --- paper pins ---
        "mathtext.fontset": "dejavusans",
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.35,
    }
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.16, dy=1.02):
    """Bold lowercase panel letter outside the axes box, upper-left."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=mpl.rcParams["font.size"] + 1, va="bottom", ha="left")


def ed_points(ax, x, y, ms=4.2, mew=0.7, zorder=3, **kw):
    """The single ED marker convention: filled circle, C_ED, white edge."""
    opts = dict(ED_KW); opts.update(ms=ms, mew=mew, zorder=zorder); opts.update(kw)
    return ax.plot(x, y, **opts)


def tail_points(ax, x, y, ms=4.2, mew=0.9, zorder=3, **kw):
    """Open circles: extrapolation-tail subsets ONLY."""
    opts = dict(TAIL_KW); opts.update(ms=ms, mew=mew, zorder=zorder); opts.update(kw)
    return ax.plot(x, y, **opts)


def clean_log_ticks(ax, axis="both"):
    from matplotlib.ticker import NullFormatter
    axes_ = ([ax.xaxis, ax.yaxis] if axis == "both"
             else [ax.xaxis] if axis == "x" else [ax.yaxis])
    for a in axes_:
        a.set_minor_formatter(NullFormatter())


def check_no_overlap(fig):
    """figure-style §9.1 geometric check: no visible text-text or
    text-foreign-spine overlaps; every text inside the figure bbox."""
    import matplotlib as _mpl
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    # Tick labels whose tick position lies outside the view interval are
    # kept as Text objects by matplotlib but never drawn -- exclude them.
    undrawn = set()
    for ax in fig.axes:
        for axis, idx, lim in ((ax.xaxis, 0, sorted(ax.get_xlim())),
                               (ax.yaxis, 1, sorted(ax.get_ylim()))):
            for t in axis.get_ticklabels(which="both"):
                pos = t.get_position()[idx]
                if not (lim[0] <= pos <= lim[1]):
                    undrawn.add(id(t))

    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(_mpl.text.Text)
             if t.get_text().strip() and t.get_visible()
             and id(t) not in undrawn]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    ticklabels = {ax: set(ax.get_xticklabels(which="both")
                          + ax.get_yticklabels(which="both")) for ax in fig.axes}
    bad = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                bad.append(("text-text", a.get_text()[:30], b.get_text()[:30]))
    for t, bt in texts:
        for s, bs in spines:
            if bt.overlaps(bs) and t not in ticklabels.get(s.axes, set()):
                bad.append(("text-spine", t.get_text()[:30], ""))
    return bad
