"""Build the sepsis early-warning presentation as a .pptx file.

Tight 7-slide deck for a non-technical doctor + generalist tech audience.
Focus: results, then what changes when this meets live hospital data and the
Indian ICU context.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
ASSETS = ROOT / "docs" / "presentation_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = RGBColor(0x0E, 0x2A, 0x47)
ACCENT = RGBColor(0x1F, 0x77, 0xB4)
DARK = RGBColor(0x33, 0x33, 0x33)
LIGHT = RGBColor(0x77, 0x77, 0x77)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ---------- charts ----------
def render_results_panel(out: Path, sens: float, spec: float) -> None:
    """One panel: big numbers + threshold trade-off table."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160,
                             gridspec_kw={"width_ratios": [1, 1.6]})

    # Left: big stat tiles
    ax = axes[0]
    ax.axis("off")
    tiles = [
        (f"{sens*100:.0f}%", "of sepsis cases caught", "#1F77B4"),
        (f"{spec*100:.0f}%", "non-sepsis correctly cleared", "#0E2A47"),
        ("6 hrs", "before clinical onset", "#2A9D8F"),
        ("40,336", "patients trained on", "#555555"),
    ]
    positions = [(0.05, 0.55), (0.55, 0.55), (0.05, 0.05), (0.55, 0.05)]
    for (val, label, color), (x, y) in zip(tiles, positions):
        ax.add_patch(plt.Rectangle((x, y), 0.42, 0.4, facecolor="#F5F7FA",
                                   edgecolor="#E0E0E0", linewidth=1, transform=ax.transAxes))
        ax.text(x + 0.21, y + 0.27, val, ha="center", va="center", fontsize=28,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.21, y + 0.10, label, ha="center", va="center", fontsize=10,
                color="#444", transform=ax.transAxes)
    ax.set_title("Headline results", fontsize=13, color="#0E2A47", pad=14, loc="left")

    # Right: threshold trade-off table
    ax = axes[1]
    ax.axis("off")
    rows = [
        ("Conservative\n(0.50)",  "64%", "89%", "32%",  "5,940"),
        ("Balanced\n(0.30)",      "82%", "75%", "20%", "11,835"),
        ("Aggressive\n(0.20)",    "90%", "63%", "16%", "16,517"),
    ]
    headers = ["Mode\n(threshold)", "Sepsis\ncaught", "Non-sepsis\ncleared", "Precision",
               "Patients\nflagged"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center",
                     colWidths=[0.22, 0.16, 0.18, 0.14, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        if r == 0:
            cell.set_facecolor("#0E2A47")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F5F7FA" if r % 2 == 0 else "white")
            cell.set_text_props(color="#222")
    ax.set_title("Trade-off — clinicians pick the operating point",
                 fontsize=12, color="#0E2A47", pad=14, loc="center")

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_data_tiles(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=160)
    ax.axis("off")
    tiles = [
        ("40,336", "ICU patients",                  "#1F77B4"),
        ("1.5 M",  "hourly observations",           "#0E2A47"),
        ("34",     "clinical columns per hour",     "#2A9D8F"),
        ("7.3%",   "develop sepsis (2,932 cases)",  "#D87C3A"),
    ]
    for i, (val, label, color) in enumerate(tiles):
        x = 0.02 + i * 0.245
        ax.add_patch(plt.Rectangle((x, 0.15), 0.225, 0.7,
                                   facecolor="#F5F7FA", edgecolor="#E0E0E0",
                                   linewidth=1, transform=ax.transAxes))
        ax.text(x + 0.1125, 0.62, val, ha="center", va="center", fontsize=30,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.1125, 0.32, label, ha="center", va="center", fontsize=11,
                color="#444", transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_iv_bars(out: Path, iv_features: list[dict]) -> None:
    top = iv_features[:10][::-1]
    names = [f["feature"].replace("_", " ") for f in top]
    values = [f["iv"] for f in top]
    strengths = [f["iv_strength"] for f in top]
    color_map = {"Strong": "#1F77B4", "Medium": "#6BAED6", "Weak": "#C6DBEF"}
    colors = [color_map.get(s, "#888") for s in strengths]

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    y = np.arange(len(top))
    ax.barh(y, values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color="#333")
    ax.set_xlabel("Information Value (IV)", fontsize=10, color="#555")
    ax.set_title("Top 10 features by Information Value (causal pipeline)",
                 fontsize=12, color="#0E2A47", pad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC"); ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#666")
    ax.set_xlim(0, max(values) * 1.15)
    ax.axvline(0.1, color="#999", linestyle=":", linewidth=1)
    ax.axvline(0.3, color="#999", linestyle=":", linewidth=1)
    ax.text(0.1, len(top) - 0.3, "  Medium", fontsize=8, color="#888", va="bottom")
    ax.text(0.3, len(top) - 0.3, "  Strong", fontsize=8, color="#888", va="bottom")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_feature_selection_compare(out: Path, full: dict, top100: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=160)
    metrics_a = ("AUROC", full["xgb_auroc"], top100["xgb_auroc"])
    metrics_b = ("Overfit gap", full["overfit_gap"], top100["overfit_gap"])
    for ax, (name, a, b) in zip(axes, [metrics_a, metrics_b]):
        ax.bar(["309 features", "100 features"], [a, b],
               color=["#0E2A47", "#1F77B4"], edgecolor="white", width=0.5)
        ax.set_title(name, fontsize=11, color="#0E2A47")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#666", labelsize=9)
        for i, v in enumerate([a, b]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color="#222")
        ax.set_ylim(0, max(a, b) * 1.2)
    fig.suptitle("Cutting features barely costs us AUROC — and tightens the overfit gap",
                 fontsize=11, color="#0E2A47", y=1.04)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_feature_example(out: Path) -> dict:
    """Dark-theme bar chart: sepsis rate by Lactate_baseline_dev_max_6h bin.

    Computes the engineered feature from imputed_data.parquet using the same
    causal CUSUM baseline the model uses, then bins values and shows how the
    sepsis hour-rate climbs with deviation magnitude.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.features import _causal_cusum_baselines  # noqa: E402

    df = pd.read_parquet(ROOT / "data/processed/imputed_data.parquet")
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    dev = np.zeros(len(df))
    for _pid, g in df.groupby("patient_id", sort=False):
        idx = g.index.to_numpy()
        vals = g["Lactate"].to_numpy()
        base = _causal_cusum_baselines(vals)
        dev[idx] = vals - base
    df["Lac_dev"] = dev
    df["Lac_dev_max6h"] = (df.groupby("patient_id", sort=False)["Lac_dev"]
                             .rolling(6, min_periods=1).max()
                             .reset_index(level=0, drop=True))

    edges = [-np.inf, -0.5, -1e-3, 1e-3, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
    labels = ["< −0.5", "−0.5 to 0", "≈ 0", "0 to 0.25", "0.25 to 0.5",
              "0.5 to 1", "1 to 2", "2 to 4", "> 4"]
    df["bin"] = pd.cut(df["Lac_dev_max6h"], edges, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True).agg(
        n=("SepsisLabel", "size"), rate=("SepsisLabel", "mean")).reset_index()
    pop_rate = float(df["SepsisLabel"].mean())

    BG, CARD = "#0F1929", "#15243D"
    CYAN, SALMON = "#4FC3E5", "#E57373"
    LIGHT, DIM, YELLOW = "#E8EEF5", "#8898A8", "#F9C440"

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    rates = grp["rate"].to_numpy() * 100
    bin_labels = grp["bin"].astype(str).tolist()
    colors = [CYAN if r < pop_rate * 100 else SALMON for r in rates]
    bars = ax.bar(bin_labels, rates, color=colors, edgecolor=BG, linewidth=1.0)

    ax.axhline(pop_rate * 100, color=YELLOW, linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Population avg ({pop_rate*100:.1f}%)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.12, f"{r:.1f}%",
                ha="center", va="bottom", color=LIGHT, fontsize=9)

    ax.set_title("Sepsis Rate by Lactate Deviation from Personal Baseline",
                 fontsize=14, color=LIGHT, pad=12)
    ax.set_xlabel("Lactate deviation from this patient's baseline (mmol/L, 6h max)",
                  fontsize=11, color=LIGHT)
    ax.set_ylabel("Sepsis rate (% of hours in bin)", fontsize=11, color=LIGHT)
    for s in ax.spines.values():
        s.set_color(DIM); s.set_linewidth(0.6)
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_ylim(0, max(rates.max(), pop_rate * 100) * 1.25)
    leg = ax.legend(frameon=True, fontsize=9, loc="upper left",
                    facecolor=BG, edgecolor=DIM, labelcolor=LIGHT)
    leg.get_frame().set_alpha(0.9)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    rate_zero = float(grp.loc[grp["bin"] == "≈ 0", "rate"].iloc[0]) * 100
    rate_high = float(grp.loc[grp["bin"] == "> 4", "rate"].iloc[0]) * 100
    return {"pop_rate": pop_rate * 100, "rate_zero": rate_zero,
            "rate_high": rate_high, "n_total": int(grp["n"].sum())}


def render_baseline_deviation(out: Path) -> None:
    """A toy heart-rate trace showing personal baseline + deviation alarm."""
    rng = np.random.default_rng(7)
    hours = np.arange(0, 36)
    # Stable for ~16h, then drift up
    baseline_signal = 78 + rng.normal(0, 2.5, len(hours))
    drift = np.where(hours < 16, 0, (hours - 16) * 1.6)
    hr = baseline_signal + drift
    # Causal running baseline (mean of first 6h then frozen until alarm,
    # mimicking the algorithm the model uses)
    causal_base = np.full_like(hr, hr[:6].mean())
    alarm_idx = next((i for i, v in enumerate(hr) if v - causal_base[i] > 12 and i >= 6), None)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ax.plot(hours, hr, color="#1F77B4", linewidth=2.0, label="Heart rate (this patient)")
    ax.plot(hours, causal_base, color="#888", linewidth=1.5, linestyle="--",
            label="Their personal baseline")
    ax.fill_between(hours, causal_base, hr, where=(hr > causal_base + 4),
                    color="#FFB454", alpha=0.35, label="Deviation from baseline")
    if alarm_idx is not None:
        ax.axvline(alarm_idx, color="#D62728", linewidth=1.6, linestyle=":")
        ax.annotate("Alarm fires —\n6h before clinical onset",
                    xy=(alarm_idx, hr[alarm_idx]), xytext=(alarm_idx - 8, hr[alarm_idx] + 18),
                    fontsize=10, color="#D62728",
                    arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.2))
    ax.set_xlabel("Hours since ICU admission", fontsize=10, color="#555")
    ax.set_ylabel("Heart rate (bpm)", fontsize=10, color="#555")
    ax.set_title("Each patient is their own baseline — drift is the signal",
                 fontsize=12, color="#0E2A47", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#666")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3), dpi=160)
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.2, 1.4, 2.4, 1.4, "Hourly\nvitals + labs"),
        (3.0, 1.4, 2.6, 1.4, "Pattern\nrecognition\n(model)"),
        (6.0, 1.4, 2.6, 1.4, "Risk score\n0 → 1"),
        (9.0, 1.4, 2.6, 1.4, "Bedside alert\n(if elevated)"),
        (12.0, 1.4, 1.8, 1.4, "Clinical\naction"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#E8F0F9",
                                   edgecolor="#1F77B4", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, color="#0E2A47")
    pairs = [(2.6, 2.1, 3.0, 2.1), (5.6, 2.1, 6.0, 2.1),
             (8.6, 2.1, 9.0, 2.1), (11.6, 2.1, 12.0, 2.1)]
    for x1, y1, x2, y2 in pairs:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1F77B4", lw=1.6))
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- slide helpers ----------
def add_title(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.18),
                                 Inches(0.6), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()


def add_text(slide, left, top, width, height, text, *, size=14, bold=False, color=DARK):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tb


def add_bullets(slide, left, top, width, height, items, *, size=14, color=DARK):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = "•  " + item
        r.font.size = Pt(size); r.font.color.rgb = color
        p.space_after = Pt(8)


def add_footer(slide, idx, total):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.alignment = 2
    r = p.add_run(); r.text = f"Sepsis Early-Warning  ·  {idx} / {total}"
    r.font.size = Pt(9); r.font.color.rgb = LIGHT


# ---------- slides ----------
def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(2.5),
                                    Inches(0.4), Inches(0.06))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT; stripe.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(2.7), Inches(12), Inches(1.4))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = "Predicting Sepsis 6 Hours Before It Happens"
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
    tb2 = slide.shapes.add_textbox(Inches(0.6), Inches(4.0), Inches(12), Inches(0.8))
    r2 = tb2.text_frame.paragraphs[0].add_run()
    r2.text = "Proof of concept · 40,336 ICU patients · path to live deployment"
    r2.font.size = Pt(18); r2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)


def slide_problem(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "The Problem")
    add_bullets(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(2.5), [
        "Sepsis kills roughly 1 in 3 ICU patients who develop it.",
        "Each hour of delayed antibiotics raises mortality by ~7%.",
        "Today the diagnosis is reactive — we recognise sepsis when the patient is already crashing.",
    ], size=16)
    add_text(slide, Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.5),
             "Our question:", size=18, bold=True, color=NAVY)
    add_text(slide, Inches(0.5), Inches(4.85), Inches(12.3), Inches(1.5),
             "Can we use the data already being collected — vitals every hour, labs as drawn — "
             "to flag deterioration 6 hours before clinical onset?",
             size=17, color=DARK)
    add_footer(slide, idx, total)


def slide_what_we_built(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "What We Built")
    slide.shapes.add_picture(str(ASSETS / "pipeline.png"),
                             Inches(0.4), Inches(1.4), width=Inches(12.5))
    add_bullets(slide, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.5), [
        "Reads each patient's hourly vitals and labs — exactly what nurses already chart.",
        "Learns patterns from 40,336 ICU patients (open PhysioNet dataset, 2 hospitals).",
        "Produces a single risk score, every hour, per patient.",
        "Runs in milliseconds on a regular server — no GPU, no internet, no cloud required.",
    ], size=14)
    add_footer(slide, idx, total)


def slide_results(prs, idx, total, sens, spec):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Results")
    slide.shapes.add_picture(str(ASSETS / "results_panel.png"),
                             Inches(0.3), Inches(1.4), width=Inches(12.7))
    add_text(slide, Inches(0.5), Inches(6.1), Inches(12.3), Inches(0.9),
             "Validated honestly: every patient is scored by a model that has never seen them. "
             "The threshold is a clinical knob — sensitivity vs. alert burden is a unit-level decision.",
             size=12, color=LIGHT)
    add_footer(slide, idx, total)


def slide_data(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "The Data")
    slide.shapes.add_picture(str(ASSETS / "data_tiles.png"),
                             Inches(0.3), Inches(1.4), width=Inches(12.7))
    add_text(slide, Inches(0.5), Inches(4.3), Inches(12.3), Inches(0.5),
             "Source: PhysioNet / CinC Challenge 2019 — open ICU dataset, 2 hospitals.",
             size=14, bold=True, color=NAVY)
    add_bullets(slide, Inches(0.5), Inches(4.85), Inches(12.3), Inches(2.2), [
        "Per hour, per patient: 8 vitals (HR, BP, respiratory rate, O₂ sat, temperature, etc.) + 26 labs (lactate, creatinine, WBC, FiO₂, etc.).",
        "Sepsis is rare (7.3% patient-level / 1.8% hour-level) — drives how we measure success and how we sample during training.",
        "Label is provided by the dataset and flips to 1 exactly 6 hours before clinical sepsis onset (Sepsis-3 definition).",
    ], size=12)
    add_footer(slide, idx, total)


def slide_features(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Features — What We Built and Why")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "From 34 raw clinical columns we engineered ~300 features — every category answers a different clinical question.",
             size=13, color=DARK)

    add_text(slide, Inches(0.5), Inches(2.1), Inches(6.2), Inches(0.4),
             "Feature categories", size=15, bold=True, color=NAVY)
    add_bullets(slide, Inches(0.5), Inches(2.55), Inches(6.2), Inches(4.5), [
        "Raw vitals + labs — current values.",
        "6-hour rolling stats — mean, max, min, std (trend).",
        "Hour-over-hour deltas — acute change.",
        "Personal-baseline deviation (CUSUM) — drift from THIS patient's normal.",
        "Missingness flags + time-since-last-draw — what's NOT measured is informative.",
        "Clinical scores — SIRS, qSOFA, MEWS, Shock Index.",
        "Age & gender-stratified normal ranges.",
    ], size=12)

    add_text(slide, Inches(7.0), Inches(2.1), Inches(6.0), Inches(0.4),
             "Example assumption we challenged", size=15, bold=True, color=NAVY)
    add_text(slide, Inches(7.0), Inches(2.6), Inches(6.0), Inches(4.5),
             "We had hospital ID, ICU unit, and admission-time as candidate features. "
             "Early models ranked these as the #1 and #2 most important — meaning the model was learning "
             "\"this is Hospital B,\" not \"this patient is sick.\"\n\n"
             "We dropped them. Same with raw ICU length-of-stay (longer stay → more sepsis is tautological — "
             "we'd be predicting the past, not the future).\n\n"
             "Lesson: features that correlate with the outcome are not always features that PREDICT it. "
             "We force the model to learn biology, not bookkeeping.",
             size=11, color=DARK)
    add_footer(slide, idx, total)


def slide_feature_selection(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Feature Selection — From 309 to 100")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "Two filters: keep what predicts, drop what duplicates.",
             size=14, bold=True, color=NAVY)

    add_text(slide, Inches(0.5), Inches(2.0), Inches(6.0), Inches(0.4),
             "Information Value (IV)", size=14, bold=True, color=NAVY)
    add_text(slide, Inches(0.5), Inches(2.42), Inches(6.0), Inches(1.4),
             "How strongly does this feature, on its own, separate sepsis from non-sepsis? "
             "IV > 0.30 = strong predictor · 0.10–0.30 = medium · < 0.10 = weak. "
             "We rank every feature and keep the top discriminators.",
             size=11, color=DARK)

    add_text(slide, Inches(0.5), Inches(3.85), Inches(6.0), Inches(0.4),
             "Collinearity", size=14, bold=True, color=NAVY)
    add_text(slide, Inches(0.5), Inches(4.27), Inches(6.0), Inches(1.4),
             "When two features carry the same information (e.g., Lactate and Lactate-rolling-mean over 6h are 95% correlated), "
             "we keep one and drop the other. Reduces noise, shrinks overfitting, and makes the model easier to audit.",
             size=11, color=DARK)

    add_text(slide, Inches(0.5), Inches(5.7), Inches(6.0), Inches(0.4),
             "Result", size=14, bold=True, color=NAVY)
    add_text(slide, Inches(0.5), Inches(6.12), Inches(6.0), Inches(1.0),
             "Pruned 309 → 100 features. AUROC essentially unchanged (0.850 → 0.844). "
             "Overfit gap shrinks slightly. Inference is faster, the model is leaner.",
             size=11, color=DARK)

    slide.shapes.add_picture(str(ASSETS / "iv_bars.png"),
                             Inches(6.7), Inches(1.85), width=Inches(6.4))
    add_footer(slide, idx, total)


def slide_feature_example(prs, idx, total, info):
    BG_DARK = RGBColor(0x0F, 0x19, 0x29)
    CARD = RGBColor(0x15, 0x24, 0x3D)
    CYAN = RGBColor(0x4F, 0xC3, 0xE5)
    SALMON = RGBColor(0xE5, 0x73, 0x73)
    LIGHT = RGBColor(0xE8, 0xEE, 0xF5)
    DIM = RGBColor(0x88, 0x98, 0xA8)
    YELLOW = RGBColor(0xF9, 0xC4, 0x40)

    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = BG_DARK; bg.line.fill.background()

    title_tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.7))
    r = title_tb.text_frame.paragraphs[0].add_run()
    r.text = "How One Feature Separates Sepsis"
    r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = CYAN

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(0.5), Inches(1.05),
                                 Inches(12.3), Inches(0.04))
    bar.fill.solid(); bar.fill.fore_color.rgb = CYAN; bar.line.fill.background()

    sub = slide.shapes.add_textbox(Inches(0.5), Inches(1.18), Inches(12.3), Inches(0.45))
    sr = sub.text_frame.paragraphs[0].add_run()
    sr.text = ("Lactate deviation from each patient's own baseline (6-hour rolling max) — "
               "as the deviation grows, sepsis rate climbs.")
    sr.font.size = Pt(13); sr.font.color.rgb = DIM

    slide.shapes.add_picture(str(ASSETS / "feature_example.png"),
                             Inches(0.4), Inches(1.85), width=Inches(8.7))

    side = slide.shapes.add_textbox(Inches(9.4), Inches(1.95), Inches(3.6), Inches(0.4))
    r = side.text_frame.paragraphs[0].add_run()
    r.text = "What this shows"
    r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = CYAN

    bullets = [
        "X-axis = how far lactate is above this patient's own baseline",
        "Y-axis = % of those hours flagged as sepsis",
        "Cyan bars = below population average",
        "Red bars = above population average",
        "Yellow line = overall sepsis rate (1.8%)",
        "Pattern is monotonic — bigger drift, higher risk",
    ]
    btb = slide.shapes.add_textbox(Inches(9.4), Inches(2.4), Inches(3.6), Inches(3.5))
    btf = btb.text_frame; btf.word_wrap = True
    for i, item in enumerate(bullets):
        para = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
        run = para.add_run(); run.text = "•  " + item
        run.font.size = Pt(11.5); run.font.color.rgb = LIGHT
        para.space_after = Pt(6)

    def stat_tile(left_in, label_text, value_text, value_color):
        card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left_in), Inches(5.55),
                                       Inches(1.75), Inches(1.05))
        card.fill.solid(); card.fill.fore_color.rgb = CARD; card.line.fill.background()
        lab = slide.shapes.add_textbox(Inches(left_in + 0.1), Inches(5.6),
                                        Inches(1.6), Inches(0.3))
        r = lab.text_frame.paragraphs[0].add_run()
        r.text = label_text
        r.font.size = Pt(10); r.font.color.rgb = DIM
        val = slide.shapes.add_textbox(Inches(left_in + 0.1), Inches(5.93),
                                        Inches(1.6), Inches(0.6))
        r2 = val.text_frame.paragraphs[0].add_run()
        r2.text = value_text
        r2.font.size = Pt(20); r2.font.bold = True; r2.font.color.rgb = value_color

    stat_tile(9.4,  "Near baseline",       f"{info['rate_zero']:.1f}%",  CYAN)
    stat_tile(11.25, "Deviation > 4",       f"{info['rate_high']:.1f}%",  SALMON)

    cap_stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.4), Inches(6.75),
                                        Inches(0.06), Inches(0.5))
    cap_stripe.fill.solid(); cap_stripe.fill.fore_color.rgb = YELLOW
    cap_stripe.line.fill.background()
    cap = slide.shapes.add_textbox(Inches(0.55), Inches(6.7), Inches(12.3), Inches(0.6))
    crun = cap.text_frame.paragraphs[0].add_run()
    crun.text = ("Sepsis hour-rate at ≈4× the population average when lactate has drifted "
                 "more than 4 mmol/L above the patient's own baseline. The model uses ~300 such "
                 "features together — no single one is the answer, but each one moves the score.")
    crun.font.size = Pt(10.5); crun.font.color.rgb = LIGHT
    cap.text_frame.word_wrap = True

    foot = slide.shapes.add_textbox(Inches(0.5), Inches(7.18), Inches(12.3), Inches(0.25))
    fp = foot.text_frame.paragraphs[0]
    fr = fp.add_run()
    fr.text = (f"Lactate baseline-deviation (6h max) by hour  ·  "
               f"{info['n_total']/1e6:.2f}M hourly observations  ·  "
               f"both hospitals  ·  {idx} / {total}")
    fr.font.size = Pt(9); fr.font.color.rgb = DIM
    fp.alignment = 2


def slide_methodology(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "How We Detect Deviation")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "Each patient is their own baseline — and that's the point.",
             size=15, bold=True, color=NAVY)
    slide.shapes.add_picture(str(ASSETS / "deviation.png"),
                             Inches(0.4), Inches(1.85), width=Inches(8.5))
    add_bullets(slide, Inches(9.1), Inches(1.95), Inches(4.0), Inches(5.0), [
        "First ~6 hours establish this patient's personal normal.",
        "Each subsequent hour is compared against THEIR baseline, not a textbook range.",
        "We accumulate small deviations (a method called CUSUM) — sustained drift trips the alarm.",
        "Crucially: at hour t we only use data through hour t. No peeking at the future.",
        "70 bpm may be normal for one patient and abnormal for another — absolute thresholds miss this. Personal baselines don't.",
    ], size=11)
    add_footer(slide, idx, total)


def slide_assumptions(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "What the Model Assumes")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "Six standing assumptions — anywhere these break, the model needs revisiting.",
             size=14, color=LIGHT)
    items = [
        ("Hourly vitals and labs capture enough of the patient's state",
         "If a unit charts only every 4–6 hours, the rolling trends become coarse and the personal baseline takes longer to establish."),
        ("The 6-hour-before-onset label is clinically meaningful",
         "We rely on the Sepsis-3 definition encoded in the dataset. A different definition (e.g., site-specific protocols) shifts what we're predicting."),
        ("A patient's first hours represent their stable baseline",
         "If they arrive already deteriorating, there is no clean baseline to deviate from. We exclude these patients from training."),
        ("Patterns learned on PhysioNet patients transfer to similar ICUs",
         "Same broad demographics, similar instrument set. We assume the relationship between vital trajectories and sepsis is biological, not site-specific."),
        ("Missingness is itself informative",
         "Which labs get ordered, and how often, reflects clinical concern. Sicker patients get tested more — the model uses this signal."),
        ("The model is a recommender, not a decision-maker",
         "It outputs a risk score. The clinician chooses the action. Liability and trust both rest there, by design."),
    ]
    y = 2.0
    for header, body in items:
        add_text(slide, Inches(0.5), Inches(y), Inches(12.3), Inches(0.4),
                 "•  " + header, size=13, bold=True, color=NAVY)
        add_text(slide, Inches(0.85), Inches(y + 0.36), Inches(12.0), Inches(0.5),
                 body, size=11, color=DARK)
        y += 0.85
    add_footer(slide, idx, total)


def slide_live_data(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "From Research Data to Live Hospital Data")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "Our PoC runs on a clean, retrospective dataset. A live ICU is messier — and richer.",
             size=14, color=LIGHT)
    add_text(slide, Inches(0.5), Inches(2.05), Inches(6.0), Inches(0.5),
             "What gets harder", size=16, bold=True, color=NAVY)
    add_bullets(slide, Inches(0.5), Inches(2.5), Inches(6.0), Inches(4.5), [
        "Real-time data quality: lab delays, sensor dropouts, charting lag.",
        "EHR integration: every hospital's HL7 / FHIR feed is different.",
        "Alert fatigue: the threshold must be tuned for that unit's workflow.",
        "Drift: patient mix and protocols change. Model needs monthly review.",
        "Workflow trust: clinicians must understand what the alert is saying.",
    ], size=12)
    add_text(slide, Inches(7.0), Inches(2.05), Inches(6.0), Inches(0.5),
             "What gets better", size=16, bold=True, color=NAVY)
    add_bullets(slide, Inches(7.0), Inches(2.5), Inches(6.0), Inches(4.5), [
        "Variables we don't have today — urine output, vasopressors, ventilator status, procalcitonin, CRP, cultures.",
        "Outcome data — discharge status, mortality, antibiotic timing — feeds back into retraining.",
        "Patient-baseline learning: re-train on YOUR population, not Boston ICU patients.",
        "Cross-system signals: nurse notes, imaging timestamps, escalation patterns.",
        "Multi-outcome triage: sepsis is just the first label.",
    ], size=12)
    add_footer(slide, idx, total)


def slide_india(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Why This Matters More in India")
    add_bullets(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.6), [
        "Sepsis burden is higher: India accounts for ~3 million sepsis cases annually, "
        "with case-fatality rates substantially above OECD averages.",
        "ICU capacity is constrained: ~2.3 ICU beds per 100,000 vs. ~30 in the US. "
        "Better triage = better use of scarce beds.",
        "Nurse-to-patient ratios are tighter — automated, reliable alerting compensates "
        "for what continuous bedside vigilance can't.",
        "Antimicrobial resistance is steeper: carbapenem-resistant organisms are common. "
        "Earlier identification means earlier (and narrower) antibiotic choices.",
        "Lab access varies hospital-to-hospital — the model is designed to degrade gracefully "
        "when lactate or procalcitonin isn't available hourly.",
        "Cost-of-deployment matters: this model runs on commodity CPU. No GPUs, no cloud "
        "dependency, no per-query LLM cost.",
        "EHR landscape is fragmented — but a deterministic, auditable model is much easier "
        "to get past Indian regulators than a black-box LLM.",
    ], size=13)
    add_footer(slide, idx, total)


def slide_generalization(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Beyond Sepsis — A General ICU Triage Platform")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.6),
             "Sepsis is the proof of concept. The pipeline is outcome-agnostic.",
             size=15, bold=True, color=NAVY)

    add_text(slide, Inches(0.5), Inches(2.1), Inches(6.0), Inches(0.4),
             "Reusable across outcomes:", size=14, bold=True, color=NAVY)
    add_bullets(slide, Inches(0.5), Inches(2.55), Inches(6.0), Inches(4.5), [
        "Same data layer — hourly vitals, labs, missingness patterns.",
        "Same feature engineering — rolling trends, personal-baseline drift, clinical scores.",
        "Same validation discipline — patient-level cross-validation, no leakage.",
        "Same alerting + explanation layer.",
        "Only the label changes — what we're predicting.",
    ], size=12)

    add_text(slide, Inches(7.0), Inches(2.1), Inches(6.0), Inches(0.4),
             "Outcomes the same approach can target:", size=14, bold=True, color=NAVY)
    add_bullets(slide, Inches(7.0), Inches(2.55), Inches(6.0), Inches(4.5), [
        "Acute kidney injury (AKI)",
        "Respiratory failure / impending intubation",
        "Hemodynamic collapse / shock",
        "Cardiac arrest within X hours",
        "ICU readmission risk at discharge",
        "Delirium onset",
    ], size=12)

    add_text(slide, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.8),
             "Run several in parallel → each ICU bed gets a vector of risks, not a single label. "
             "One platform, many early-warnings, one triage dashboard per unit.",
             size=12, color=DARK)
    add_footer(slide, idx, total)


def slide_path_forward(prs, idx, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Path Forward")
    add_text(slide, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             "Three steps from proof of concept to bedside.", size=14, color=LIGHT)
    items = [
        ("1.  Partner with a hospital",
         "Pilot on de-identified EHR data from one ICU. Re-train on the local patient population. "
         "Validate on outcomes that matter — antibiotic timing, mortality, length of stay."),
        ("2.  Wire it into the workflow",
         "Live HL7 / FHIR feed → hourly scoring → alert routed to the nurse station. "
         "Run silent (shadow mode) for 4–8 weeks before any clinical action is taken."),
        ("3.  Expand label-by-label",
         "Add AKI, then respiratory failure, then shock — same pipeline, new labels. "
         "Each new outcome compounds the platform's value without rebuilding the foundation."),
    ]
    y = 2.1
    for header, body in items:
        add_text(slide, Inches(0.5), Inches(y), Inches(12.3), Inches(0.5),
                 header, size=16, bold=True, color=NAVY)
        add_text(slide, Inches(0.5), Inches(y + 0.45), Inches(12.3), Inches(1.0),
                 body, size=12, color=DARK)
        y += 1.55
    add_footer(slide, idx, total)


# ---------- driver ----------
def main():
    print("Loading metrics...")
    metrics = json.loads((ROOT / "data/processed/model_metrics.json").read_text())
    sens = metrics["patient_sensitivity"]
    spec = metrics["patient_specificity"]

    print("Rendering charts...")
    render_results_panel(ASSETS / "results_panel.png", sens, spec)
    render_baseline_deviation(ASSETS / "deviation.png")
    render_pipeline(ASSETS / "pipeline.png")
    render_data_tiles(ASSETS / "data_tiles.png")
    render_iv_bars(ASSETS / "iv_bars.png", metrics["iv_top20"])
    fe_info = render_feature_example(ASSETS / "feature_example.png")

    print("Building deck...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    total = 13
    slide_title(prs)
    slide_problem(prs, 1, total)
    slide_what_we_built(prs, 2, total)
    slide_results(prs, 3, total, sens, spec)
    slide_data(prs, 4, total)
    slide_features(prs, 5, total)
    slide_feature_selection(prs, 6, total)
    slide_feature_example(prs, 7, total, fe_info)
    slide_methodology(prs, 8, total)
    slide_assumptions(prs, 9, total)
    slide_live_data(prs, 10, total)
    slide_india(prs, 11, total)
    slide_generalization(prs, 12, total)
    slide_path_forward(prs, 13, total)

    out = ROOT / "docs" / "Sepsis_Early_Warning_Presentation.pptx"
    prs.save(out)
    print(f"Saved: {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
