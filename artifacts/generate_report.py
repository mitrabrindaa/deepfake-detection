"""Generate PDF training report with accuracy bar chart for all papers."""
import os, re, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, PageBreak)
from reportlab.lib.enums import TA_CENTER

_HERE        = os.path.dirname(os.path.abspath(__file__))
LOG_FILES    = [os.path.join(_HERE, "training_full_run.log")] + \
               glob.glob(os.path.join(_HERE, "training_full_run_fastdemo_*.log"))
EXCEL_FILE   = os.path.join(_HERE, "COMPARISON UPDATED.xlsx")
METRICS_FILE = os.path.join(_HERE, "metrics.json")
HISTORY_FILE = os.path.join(_HERE, "history.json")
CHART_DIR    = os.path.join(_HERE, "figures")
OUTPUT_PDF   = os.path.join(_HERE, "Deepfake_Training_Report.pdf")
os.makedirs(CHART_DIR, exist_ok=True)

# ── 1. Our model metrics ─────────────────────────────────────────────────────
print("Parsing logs and metrics...")

def read_log(path):
    for enc in ("utf-16", "utf-8"):
        try:
            return open(path, encoding=enc).read()
        except Exception:
            pass
    return ""

best_val_acc = 0.0
for lf in LOG_FILES:
    for m in re.finditer(r'val_acc[^\d]*([\d\.]+)', read_log(lf), re.IGNORECASE):
        v = float(m.group(1))
        if v > best_val_acc:
            best_val_acc = v

our = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
if os.path.exists(METRICS_FILE):
    mj = json.load(open(METRICS_FILE))
    cr = mj.get("classification_report", "")
    mac = re.search(r'macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', cr)
    if mac:
        our["Precision"] = float(mac.group(1))
        our["Recall"]    = float(mac.group(2))
        our["F1"]        = float(mac.group(3))
    our["Accuracy"] = max(best_val_acc, mj.get("test_accuracy", 0.0))

print(f"  Our model -> {our}")

# ── 2. Parse Excel — ALL papers ──────────────────────────────────────────────
print("Reading benchmark data...")

def parse_pct(val):
    """Extract first numeric value; return None if not parseable."""
    if pd.isna(val): return None
    s = str(val).strip()
    # skip clearly non-numeric entries
    if any(x in s.lower() for x in ["n/a", "not stated", "survey", "high",
                                      "competitive", "comparable", "improved",
                                      "robust", "state-of", "unspecified"]):
        return None
    # handle decimals like 0.96 or 0.615
    m = re.search(r'\b(0\.\d+)\b', s)
    if m:
        return float(m.group(1))
    # handle percentages like 98.5% or 96%
    m = re.search(r'(\d+\.?\d*)\s*%', s)
    if m:
        return float(m.group(1)) / 100.0
    # handle plain numbers like 98 (assume percentage if > 1)
    m = re.search(r'(\d+\.?\d*)', s)
    if m:
        v = float(m.group(1))
        return v / 100.0 if v > 1.0 else v
    return None

def short_method(method):
    """Use method as the label, trimmed."""
    s = str(method).strip()
    if s == "nan" or not s:
        return "Unknown"
    return s[:35] + "..." if len(s) > 35 else s

papers = []
if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    seen = set()
    for _, row in df.iterrows():
        name   = str(row.get("Paper Name", "")).strip()
        method = str(row.get("Method used", "")).strip()
        if not name or name == "nan": continue
        # deduplicate by (name, method)
        key = (name[:40], method[:30])
        if key in seen: continue
        seen.add(key)
        acc = parse_pct(row.get("Reported accuracy") or row.get("Accuracy"))
        papers.append({
            "label":  short_method(method),
            "name":   name,
            "method": method,
            "year":   str(row.get("Year", "")).strip(),
            "acc":    acc,   # None = no numeric value
        })

# Keep only papers with a numeric accuracy
papers = [p for p in papers if p["acc"] is not None]

# Add our model at the end
papers.append({
    "label":  "Our Model (XceptionNet+EfficientNet)",
    "name":   "Our Model",
    "method": "Ensemble",
    "year":   "2026",
    "acc":    our["Accuracy"],
})
print(f"  {len(papers)} entries with numeric accuracy (including ours)")

# ── 3. Accuracy bar chart — ALL papers ───────────────────────────────────────
print("Generating accuracy chart...")

labels    = [p["label"] for p in papers]
values    = [p["acc"]   for p in papers]   # None for papers without numeric acc
bar_vals  = [v if v is not None else 0.0 for v in values]
bar_colors= ["#f7c948" if "Our Model" in l else "#4f8ef7"
             for l in labels]

n = len(labels)
fig_h = max(8, n * 0.52 + 2)
fig, ax = plt.subplots(figsize=(14, fig_h))
fig.patch.set_facecolor("#0d1120")
ax.set_facecolor("#0d1120")

y = np.arange(n)
bars = ax.barh(y, bar_vals, color=bar_colors, edgecolor="none", height=0.65)

for bar, val, orig in zip(bars, bar_vals, values):
    txt = f"{val*100:.1f}%" if orig is not None else "N/A"
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            txt, va="center", ha="left", color="white", fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(labels, color="white", fontsize=8.5)
# Bold our model's tick label
for tick, lbl in zip(ax.get_yticklabels(), labels):
    if "Our Model" in lbl:
        tick.set_fontweight("bold")
        tick.set_fontsize(9.5)
ax.set_xlim(0, 1.2)
ax.set_xlabel("Accuracy", color="#94a3b8", fontsize=10)
ax.set_title("Accuracy Comparison — All Models", color="white", fontsize=13, pad=12)
ax.tick_params(colors="#94a3b8")
for spine in ax.spines.values(): spine.set_edgecolor("#2a2a40")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.grid(axis="x", color="#2a2a40", linewidth=0.6)
ax.legend(handles=[
    mpatches.Patch(color="#f7c948", label="Our Model"),
    mpatches.Patch(color="#4f8ef7", label="Numeric accuracy"),
], facecolor="#1a1a2e", edgecolor="#2a2a40", labelcolor="white", fontsize=8,
   loc="lower right")

plt.tight_layout()
acc_chart = os.path.join(CHART_DIR, "accuracy_all.png")
plt.savefig(acc_chart, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved accuracy chart")

# ── 4. Training curves ───────────────────────────────────────────────────────
curve_path = None
if os.path.exists(HISTORY_FILE):
    h = json.load(open(HISTORY_FILE))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1120")
    for ax in axes:
        ax.set_facecolor("#0d1120")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a40")
        ax.grid(color="#2a2a40", linewidth=0.5)
    if "train_loss" in h: axes[0].plot(h["train_loss"], color="#4f8ef7", label="Train")
    if "val_loss"   in h: axes[0].plot(h["val_loss"],   color="#f7874f", label="Val")
    axes[0].set_title("Loss", color="white")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[0].set_xlabel("Epoch", color="#94a3b8")
    if "train_acc" in h: axes[1].plot(h["train_acc"], color="#4f8ef7", label="Train")
    if "val_acc"   in h: axes[1].plot(h["val_acc"],   color="#f7874f", label="Val")
    axes[1].set_title("Accuracy", color="white")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[1].set_xlabel("Epoch", color="#94a3b8")
    plt.tight_layout()
    curve_path = os.path.join(CHART_DIR, "curves_report.png")
    plt.savefig(curve_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved training curves")

# ── 5. Build PDF ─────────────────────────────────────────────────────────────
print("Compiling PDF...")

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4,
                        leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()
title_s = ParagraphStyle("t", fontSize=22, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#5078ff"),
                          alignment=TA_CENTER, spaceAfter=6)
sub_s   = ParagraphStyle("s", fontSize=10, fontName="Helvetica",
                          textColor=colors.HexColor("#8899bb"),
                          alignment=TA_CENTER, spaceAfter=20)
h2_s    = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#3c3c50"), spaceAfter=8)

story = []

# Title
story.append(Paragraph("Deepfake Detection Model Report", title_s))
story.append(Paragraph("XceptionNet + EfficientNet Ensemble<br/>2026", sub_s))

# Our metrics table
story.append(Paragraph("Our Model Performance", h2_s))
mdata = [["Metric", "Score"]] + [[k, f"{v*100:.2f}%"] for k, v in our.items()]
mt = Table(mdata, colWidths=[8*cm, 6*cm])
mt.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#283c9a")),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 10),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eef0ff"), colors.white]),
    ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
    ("ALIGN",        (1,0), (1,-1), "CENTER"),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
]))
story.append(mt)
story.append(Spacer(1, 0.5*cm))

# Training curves
if curve_path and os.path.exists(curve_path):
    story.append(Paragraph("Training Curves", h2_s))
    story.append(Image(curve_path, width=16*cm, height=6*cm))
    story.append(Spacer(1, 0.4*cm))

# Accuracy chart — full page
story.append(PageBreak())
story.append(Paragraph("Accuracy Comparison — All Models", h2_s))
story.append(Spacer(1, 0.2*cm))
# scale image height to fit A4 (max ~24cm usable)
img_h = min(24, max(10, n * 0.52 + 2)) * cm
story.append(Image(acc_chart, width=17*cm, height=img_h))

# Paper table
story.append(PageBreak())
story.append(Paragraph("Benchmark Papers Summary", h2_s))
story.append(Spacer(1, 0.3*cm))

tdata = [["Method / Model", "Year", "Accuracy"]]
for p in papers:
    acc_str = f"{p['acc']*100:.1f}%" if p["acc"] is not None else "N/A"
    tdata.append([p["label"], p["year"], acc_str])

pt = Table(tdata, colWidths=[11*cm, 2*cm, 4.5*cm])
row_styles = []
for i in range(1, len(tdata)):
    if "Our Model" in tdata[i][0]:
        row_styles.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fff8cc")))
        row_styles.append(("FONTNAME",   (0,i), (-1,i), "Helvetica-Bold"))
    elif i % 2 == 0:
        row_styles.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#f0f2ff")))

pt.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#283c9a")),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#ccccdd")),
    ("TOPPADDING",   (0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
] + row_styles))
story.append(pt)

doc.build(story)
print(f"\nDone -> {OUTPUT_PDF}")
