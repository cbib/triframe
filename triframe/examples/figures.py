#!/usr/bin/env python3
"""Figures for the validation examples: flag rate per group and method.

Fill = share of seeds flagged; text only where informative; filled/open dot =
designed truth; vermilion = disagreement with truth. Reads validation_out/*.csv.

  python -m triframe.examples.figures --in validation_out [--out DIR] [--tiff]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})

RAMP = LinearSegmentedColormap.from_list("flag", ["#eef3f7", "#a9c4da", "#5b88ad", "#1f4b6e"])
ACCENT, INK, MUTED, HAIR = "#d55e00", "#1d232b", "#68737f", "#dde2e7"

INFO = ["frechet", "cohen_d", "linear_probe", "gb_probe"]
USE = ["attention", "ablation", "patch_flip", "grad_x_input", "integrated_gradients"]
TWO = {"frechet": "Fréchet\ndistance", "cohen_d": "Cohen's d", "linear_probe": "Linear\nprobe",
       "gb_probe": "GB\nprobe", "attention": "Attention", "ablation": "Ablation",
       "patch_flip": "Activation\npatching", "grad_x_input": "Gradient\n×input",
       "integrated_gradients": "Integrated\ngradients", "rf_impurity": "Impurity\nimportance"}
FULL = {k: v.replace("\n", " ") for k, v in TWO.items()} | {"gb_probe": "Gradient-boosting probe"}
FULL2 = FULL | {"gb_probe": "GB probe"}
LAYER = {"frechet": "Layer 2", "cohen_d": "Layer 2", "linear_probe": "Layer 1", "gb_probe": "Layer 1",
         "attention": "Layer 1", "ablation": "Layer 1", "patch_flip": "Layer 3",
         "grad_x_input": "Attribution", "integrated_gradients": "Attribution", "rf_impurity": "Impurity"}

# (scenario label, [(row label, [(scenario, group), ...] or "noise"), ...])
SCEN = [
    ("Informative", [("S", [("informative", "S"), ("irrelevant", "S")])]),
    ("Suppressor", [("A", [("suppressor", "A")]), ("B", [("suppressor", "B")])]),
    ("Correlated, unused", [("A", [("correlated_unused", "A")]), ("B", [("correlated_unused", "B")])]),
    ("Covariance only", [("C", [("covariance_only", "C")])]),
    ("Redundant", [("Single (mean of 3)", [("redundant", g) for g in ("R1", "R2", "R3")]),
                   ("All three jointly", [("redundant", "R1+R2+R3")])]),
    ("XOR", [("Single (mean of 2)", [("xor", "X1"), ("xor", "X2")]),
             ("Pair jointly", [("xor", "X1+X2")])]),
    ("Noise groups", [("Reference", "noise"),
                      ("Held out (mean of 2)", [("irrelevant", "Z1"), ("irrelevant", "Z2")])]),
]


# ---------------------------------------------------------------- primitives
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def cell(ax, x, y, cw, ch, v, bad=False, annotate=True, fs=6.2, pad=0.02, r=0.04):
    if np.isnan(v):
        ax.text(x + cw / 2, y + ch / 2, "–", ha="center", va="center", color=MUTED, fontsize=fs)
        return
    box = dict(boxstyle=f"round,pad=0,rounding_size={r}")
    ax.add_patch(FancyBboxPatch((x + pad, y + pad), cw - 2 * pad, ch - 2 * pad, fc=RAMP(v), ec="none", **box))
    if bad:
        ax.add_patch(FancyBboxPatch((x + pad, y + pad), cw - 2 * pad, ch - 2 * pad, fc="none",
                                    ec=ACCENT, lw=1.4, zorder=5, **box))
    if annotate and (0 < v < 1 or bad):
        ax.text(x + cw / 2, y + ch / 2 + 0.004, f"{v:.0%}", ha="center", va="center", fontsize=fs,
                color="white" if v >= 0.55 else INK, fontweight="bold" if bad else "normal", zorder=6)


def bracket(ax, x0, x1, y, text, pad=0.04):
    ax.plot([x0 + pad, x1 - pad], [y, y], color=MUTED, lw=0.6, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - 0.045, text, ha="center", va="bottom", fontsize=6.0, color=MUTED)


def truth_dot(ax, x, y, yes, ms=3.6):
    ax.plot([x], [y], marker="o", ms=ms, mfc=INK if yes else "white", mec=INK, mew=0.8, ls="", clip_on=False)


def _width(ax, text):
    """Width of a text artist in data units (the canvas is drawn in inches)."""
    fig = ax.figure
    return text.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def legend_items(ax, x, y, items, gap=0.32):
    """Lay out key entries left to right with a constant gap. items: ("dot", filled, label),
    ("dotc", colour, label) or ("box", None, label). Returns the right edge."""
    for kind, arg, label in items:
        if kind == "box":
            ax.add_patch(FancyBboxPatch((x, y - 0.045), 0.16, 0.1, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc="none", ec=ACCENT, lw=1.4))
            x += 0.16 + 0.07
        elif kind == "dot":
            truth_dot(ax, x + 0.04, y, arg); x += 0.04 + 0.09
        else:
            ax.plot([x + 0.04], [y], marker="o", ms=3.2, mfc=arg, mec="white", ls="")
            x += 0.04 + 0.09
        t = ax.text(x, y, label, fontsize=6.0, color=MUTED, va="center")
        x += _width(ax, t) + gap
    return x - gap


def gradient_legend(ax, x, y, w=0.9, h=0.09, label="Share of seeds flagged", below=False):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.imshow(np.linspace(0, 1, 100)[None, :], extent=(x, x + w, y + h, y), cmap=RAMP, aspect="auto", zorder=2)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.text(x, y + h + 0.04, "0", fontsize=6.0, color=MUTED, ha="left", va="top")
    ax.text(x + w, y + h + 0.04, "100%", fontsize=6.0, color=MUTED, ha="right", va="top")
    if below:
        ax.text(x, y - 0.05, label, fontsize=6.0, color=MUTED, ha="left", va="bottom")
    else:
        t = ax.text(x + w + 0.08, y + h / 2, label, fontsize=6, color=MUTED, ha="left", va="center")
        return x + w + 0.08 + _width(ax, t)
    return x + w


TOP = 0.26


def header(ax, W, title):
    ax.text(0.05, 0.06, title, fontsize=8.6, fontweight="bold", color=INK, va="top")
    ax.plot([0.05, W - 0.05], [TOP - 0.04, TOP - 0.04], color=INK, lw=0.7)


TIFF = False          # set by render(tiff=True): 350 dpi LZW, for journal submission


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=240)
    if TIFF:
        fig.savefig(out / f"{name}.tiff", dpi=350, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# --------------------------------------------------------------- Fig. 1
def fig1(scores, out):
    d = scores.copy()
    d["agree"] = d.flagged == d.truth

    cw, ch, gap_s = 0.45, 0.25, 0.07
    x_scen, x_row, x_g1, x_c1, x_g2 = 0.05, 1.22, 2.18, 2.38, 4.45
    x_c2 = x_g2 + 0.22
    y0 = 0.98 + TOP
    nrows = sum(len(r) for _, r in SCEN)
    H = y0 + nrows * ch + (len(SCEN) - 1) * gap_s + 0.52
    W = x_c2 + 5 * cw + 0.08
    fig, ax = canvas(W, H)
    header(ax, W, "Controlled scenarios")

    # row geometry
    ys, y = [], y0
    for _, rows in SCEN:
        for _ in rows:
            ys.append(y); y += ch
        y += gap_s
    flat = [(si, rl, sel) for si, (_, rows) in enumerate(SCEN) for rl, sel in rows]

    panels = [("A", "Does the group contain label information?", "label_info", INFO, x_c1, x_g1),
              ("B", "Does the model use it?", "model_use", USE, x_c2, x_g2)]
    for tag, title, q, methods, xc, xg in panels:
        ax.text(xg - 0.12 if q == "model_use" else x_scen, 0.07 + TOP, tag, fontsize=8, fontweight="bold", va="top", color=INK)
        ax.text((xg - 0.12 if q == "model_use" else x_scen) + 0.16, 0.085 + TOP, title, fontsize=7, fontweight="bold", va="top", color=INK)
        # layer brackets
        groups = []
        for j, m in enumerate(methods):
            if groups and groups[-1][0] == LAYER[m]:
                groups[-1][2] = j
            else:
                groups.append([LAYER[m], j, j])
        for lab, j0, j1 in groups:
            bracket(ax, xc + j0 * cw, xc + (j1 + 1) * cw, y0 - 0.5, lab)
        for j, m in enumerate(methods):
            ax.text(xc + j * cw + cw / 2, y0 - 0.05, TWO[m], ha="center", va="bottom", fontsize=6.0,
                    color=INK, linespacing=1.05)
        ax.text(xg, y0 - 0.05, "Truth", ha="center", va="bottom", fontsize=6.0, color=MUTED)
        for (si, rl, sel), yy in zip(flat, ys):
            sub = d[d.question == q]
            if sel == "noise":
                sub = sub[sub.is_noise]
            else:
                key = set(sel)
                sub = sub[np.array([(s, g) in key for s, g in zip(sub.scenario, sub.group)]) & ~sub.is_noise.values]
            assert sub.truth.nunique() <= 1, (rl, q, "merged rows disagree on truth")
            truth_dot(ax, xg, yy + ch / 2, bool(sub.truth.iloc[0]))
            for j, m in enumerate(methods):
                s = sub[sub.method == m]
                v = s.flagged.mean() if len(s) else np.nan
                cell(ax, xc + j * cw, yy, cw, ch, v, bad=bool(len(s) and s.agree.mean() < 0.5))

    # row labels and scenario separators
    k = 0
    for si, (sname, rows) in enumerate(SCEN):
        y_first = ys[k]
        ax.text(x_scen, y_first + ch / 2, sname, ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
        for rl, _ in rows:
            ax.text(x_row, ys[k] + ch / 2, rl, ha="left", va="center", fontsize=6.3, color=INK if len(rows) > 1 or True else MUTED)
            k += 1
        if si < len(SCEN) - 1:
            yl = ys[k - 1] + ch + gap_s / 2
            ax.plot([x_scen, W - 0.05], [yl, yl], color=HAIR, lw=0.6)

    # legend
    ly = H - 0.40
    xe = gradient_legend(ax, x_scen, ly) + 0.32
    legend_items(ax, xe, ly + 0.045, [("dot", True, "Should be flagged"), ("dot", False, "Should not be flagged"),
                                      ("box", None, "Most seeds disagree with truth")])
    save(fig, out, "validation_agreement")


# --------------------------------------------------------------- Fig. 2
def _rates(df, q):
    d = df[(df.question == q) & ~df.joint]
    return d.pivot_table(index="group", columns="method", values="flagged", sort=False)


def fig2(mlp, rf, out):
    order = ["mean", "error", "worst", "worst_copy", "noise", "heldout"]
    names = {"mean": "Mean", "error": "Standard error", "worst": "Worst", "worst_copy": "Worst + 10% noise (copy)",
             "noise": "Noise, reference", "heldout": "Noise, held out (×2)"}
    cw, ch, gap_s = 0.30, 0.25, 0.09
    x_lab, x_g, x0, gapq, gapp = 0.42, 1.62, 1.82, 0.12, 0.34
    y0 = 1.42 + TOP
    H = y0 + 6 * ch + gap_s + 0.52
    pan = [("A", "MLP", mlp, []), ("B", "Random forest", rf, ["rf_impurity"])]
    widths = [4 * cw + gapq + (4 + len(e) * 0) * cw for *_, e in pan]
    W = 7.0
    fig, ax = canvas(W, H)
    header(ax, W, "Breast cancer")

    ys = [y0 + i * ch + (gap_s if i >= 3 else 0) for i in range(6)]
    x = x0
    for tag, name, df, extra in pan:
        info, use = _rates(df, "label_info"), _rates(df, "model_use")
        ucols = [m for m in ["ablation", "patch_flip", "grad_x_input", "integrated_gradients"] if m in use.columns]
        ucols += [m for m in extra if m in use.columns]
        ax.text(x - (0.0 if tag == "A" else 0.0), 0.07 + TOP, tag, fontsize=8, fontweight="bold", va="top", color=INK)
        ax.text(x + 0.17, 0.085 + TOP, name, fontsize=7, fontweight="bold", va="top", color=INK)
        bracket(ax, x, x + 4 * cw, 0.5 + TOP, "Label information")
        xu = x + 4 * cw + gapq
        bracket(ax, xu, xu + len(ucols) * cw, 0.5 + TOP, "Model use")
        cols = [(INFO[j], x + j * cw, "label_info") for j in range(4)] + \
               [(m, xu + j * cw, "model_use") for j, m in enumerate(ucols)]
        for m, xc, _ in cols:
            ax.text(xc + cw / 2, y0 - 0.06, FULL2[m], rotation=90, ha="center", va="bottom", fontsize=6.0, color=INK)
        for g, yy in zip(order, ys):
            for m, xc, q in cols:
                tab = info if q == "label_info" else use
                if m not in tab.columns:
                    v = np.nan
                elif g == "heldout":
                    v = tab.loc[["noise2", "noise3"], m].mean()
                else:
                    v = tab.loc[g, m]
                cell(ax, xc, yy, cw, ch, v)
        x = xu + len(ucols) * cw + gapp

    # row labels, gutter marker, bracket for injected
    for g, yy in zip(order, ys):
        ax.text(x_lab, yy + ch / 2, names.get(g, g), ha="left", va="center", fontsize=6.4, color=INK)
    for i in (4, 5):
        truth_dot(ax, x_g, ys[i] + ch / 2, False)
    ax.text(x_g, y0 - 0.06, "Truth", ha="center", va="bottom", fontsize=6.0, color=MUTED)
    for lab, a, b in (("Real groups", 0, 2), ("Injected", 3, 5)):
        ya, yb = ys[a] + 0.03, ys[b] + ch - 0.03
        ax.plot([0.3, 0.3], [ya, yb], color=MUTED, lw=0.6)
        ax.text(0.2, (ya + yb) / 2, lab, rotation=90, ha="center", va="center", fontsize=6.0, color=MUTED)
    ax.plot([0.05, W - 0.05], [ys[3] - gap_s / 2] * 2, color=HAIR, lw=0.6)

    ly = H - 0.40
    xe = gradient_legend(ax, 0.05, ly) + 0.32
    legend_items(ax, xe, ly + 0.045, [("dot", False, "Known to be unused")])
    save(fig, out, "real_breast_cancer")


# --------------------------------------------------------------- Fig. S (digits)
def figS_digits(dg, out):
    info, use = _rates(dg, "label_info"), _rates(dg, "model_use")
    cw = ch = 0.27
    px, gx = 4 * cw, 0.2
    x0, y0, row_gap = 0.45, 0.32 + TOP, 0.62
    W = x0 + 5 * px + 4 * gx + 0.05
    H = y0 + (px + 0.64) + px + 0.5
    fig, ax = canvas(W, H)
    header(ax, W, "Handwritten digits")

    plan = [(0, j, "label_info", m) for j, m in enumerate(INFO)] + [(1, j, "model_use", m) for j, m in enumerate(USE)]
    for r, c, q, m in plan:
        tab = info if q == "label_info" else use
        x, y = x0 + c * (px + gx), y0 + r * (px + 0.64)
        ax.text(x + px / 2, y - 0.07, FULL[m], ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=INK)
        for i in range(4):
            for j in range(4):
                cell(ax, x + j * cw, y + i * ch, cw, ch, tab.loc[f"p{i}{j}", m], fs=5.8)
        n_ref, n_held = tab.loc[["n1", "n2"], m].mean(), tab.loc[["n3", "n4"], m].mean()
        ax.text(x + px / 2, y + px + 0.07, f"Reference noise: {n_ref:.0%}", ha="center", va="top", fontsize=6.0, color=MUTED)
        ax.text(x + px / 2, y + px + 0.21, f"Held-out noise: {n_held:.0%}", ha="center", va="top", fontsize=6.0, color=INK)
    for r, lab in enumerate(["Label information", "Model use"]):
        y = y0 + r * (px + 0.64)
        ax.text(0.14, y + px / 2, lab, rotation=90, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
    x5 = x0 + 4 * (px + gx)
    gradient_legend(ax, x5, y0 + 0.3, w=0.9, label="Share of seeds flagged", below=True)
    save(fig, out, "real_digits_patches")


# --------------------------------------------------------------- Fig. S (noise validation)
NV_PANELS = [("Scenarios: held out", "scenarios", "held_out"),
             ("Scenarios: cross-fit", "scenarios", "cross_fit"),
             ("Breast cancer, MLP: held out", "breast_cancer_mlp", "held_out"),
             ("Breast cancer, forest: held out", "breast_cancer_rf", "held_out"),
             ("Digits: held out", "digits_vit", "held_out"),
             ("Digits: cross-fit", "digits_vit", "cross_fit")]
NV_METHODS = INFO + USE + ["rf_impurity"]


def figS_noise(nv, out, xmax=0.30):
    """False-positive rate with exact 95% interval for noise the rule was not calibrated on."""
    pw, rh = 1.2, 0.19
    x_lab, gx, gy = 1.22, 0.52, 0.42
    meths = [m for m in NV_METHODS if m in set(nv.method)]
    rows, y = {}, 0.0
    for m in meths:
        if m == USE[0]:
            y += 0.12
        rows[m] = y; y += rh
    ph = y + 0.05
    W = 7.0
    ncol = 3
    H = TOP + 2 * (ph + gy + 0.2) + 0.5
    fig, ax = canvas(W, H)
    header(ax, W, "Noise validation")
    for n, (title, ds, kind) in enumerate(NV_PANELS):
        r, c = divmod(n, ncol)
        x0 = x_lab + c * (pw + gx)
        y0 = TOP + 0.1 + r * (ph + gy + 0.2) + 0.22
        sub = nv[(nv.dataset == ds) & (nv.kind == kind)].set_index("method")
        ax.text(x0, y0 - 0.1, title, fontsize=6.6, fontweight="bold", color=INK, va="bottom")
        X = lambda v: x0 + min(v, xmax) / xmax * pw
        for t in (0.0, 0.1, 0.2, 0.3):
            ax.plot([X(t)] * 2, [y0, y0 + ph - 0.05], color=HAIR, lw=0.5, zorder=0)
            ax.text(X(t), y0 + ph - 0.01, f"{t:.0%}" if t else "0", fontsize=6.0, color=MUTED, ha="center", va="top")
        for m in meths:
            yy = y0 + rows[m] + rh / 2
            if c == 0:
                ax.text(x_lab - 0.08, yy, FULL[m] if m != "gb_probe" else "GB probe", fontsize=6.0, color=INK, ha="right", va="center")
            if m not in sub.index:
                ax.text(x0 + 0.02, yy, "n/a", fontsize=5.8, color=MUTED, va="center")
                continue
            q = sub.loc[m]
            q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
            bad = q.false_positives > 0
            colr = ACCENT if bad else RAMP(1.0)
            ax.plot([X(q.ci_low), X(q.ci_high)], [yy, yy], color=colr, lw=1.0, solid_capstyle="round")
            ax.plot([X(q.rate)], [yy], marker="o", ms=3.2, mfc=colr, mec="white", mew=0.4, ls="")
            ax.text(x0 + pw + 0.03, yy, f"{int(q.false_positives)}/{int(q.trials)}", fontsize=5.8, color=ACCENT if bad else MUTED, va="center")
    ly = H - 0.32
    legend_items(ax, 0.05, ly + 0.03, [("dotc", RAMP(1.0), "No false positive"),
                                       ("dotc", ACCENT, "At least one false positive")])
    save(fig, out, "noise_validation")



# ------------------------------------------------------------------ captions
CAPTIONS = {
    "validation_agreement": (
        "Agreement of triframe flags with designed ground truth in seven synthetic scenarios (10 seeds each). "
        "Cells show the share of seeds in which a method flags the group. (A) Does the group contain label "
        "information? (B) Does the model use it? Circles mark groups that should (filled) or should not (open) be "
        "flagged; orange outlines mark cells where most seeds disagree. Reference noise sets the threshold and cannot "
        "be flagged by construction; held-out noise estimates false positives. GB, gradient boosting."),
    "real_breast_cancer": (
        "triframe on the Breast Cancer Wisconsin (Diagnostic) data with injected groups: a copy of the worst group "
        "(plus 10% noise) and noise groups (one reference, two held out). MLP (A) and random forest (B), 10 random "
        "splits. Cells show the share of seeds in which a method flags the group. Open circles mark groups known to "
        "be unused. Reference noise cannot be flagged by construction; the held-out row estimates false positives. "
        "GB, gradient boosting."),
    "real_digits_patches": (
        "triframe on handwritten digits with a vision transformer that reads each 8x8 image as a 4x4 grid of "
        "2x2-pixel patches. Fill shows the share of 10 seeds in which a method flags the patch. Below each panel: "
        "mean flag rate of the two reference and the two held-out noise patches."),
    "noise_validation": (
        "False-positive rates on noise groups the decision rule was not calibrated on. Held out: never part of the "
        "noise reference. Cross-fit: each reference group judged against the others only (an upper bound on the "
        "production rate). Dots: rate; lines: exact 95% interval; numbers: false positives / trials (seeds x noise "
        "groups). Orange: at least one false positive. n/a: method not available for that model. GB, gradient "
        "boosting."),
}


def render(inp, out, which=("scenarios", "real", "noise"), tiff=False):
    """Write the figures whose input CSVs exist in `inp`; returns the names written."""
    global TIFF
    TIFF = tiff
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    if "scenarios" in which and (inp / "validation_scores.csv").exists():
        fig1(pd.read_csv(inp / "validation_scores.csv"), out)
        done.append("validation_agreement")
    real = [inp / f for f in ("real_breast_cancer_mlp.csv", "real_breast_cancer_rf.csv", "real_digits_vit.csv")]
    if "real" in which and all(f.exists() for f in real):
        fig2(pd.read_csv(real[0]), pd.read_csv(real[1]), out)
        figS_digits(pd.read_csv(real[2]), out)
        done += ["real_breast_cancer", "real_digits_patches"]
    if "noise" in which and (inp / "noise_validation.csv").exists():
        figS_noise(pd.read_csv(inp / "noise_validation.csv"), out)
        done.append("noise_validation")
    for n in done:
        (out / f"{n}_caption.txt").write_text(CAPTIONS[n] + "\n")
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="validation_out")
    p.add_argument("--out", default=None, help="default: same as --in")
    p.add_argument("--tiff", action="store_true", help="also write 350 dpi TIFF")
    a = p.parse_args()
    render(a.inp, a.out or a.inp, tiff=a.tiff)


if __name__ == "__main__":
    main()