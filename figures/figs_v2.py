"""Figuras 2 y 3 (v2, 24 sep 2026). Solo cifras agregadas (conteos por semana y AUC con IC)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE = '#2a7ad6'; ORANGE = '#e8673a'; GREY = '#9a9a92'; INK = '#262624'; GRID = '#ecebe6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GREY, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'text.color': INK})

# ---------- Fig 2 ----------
bins = np.arange(4, 34, 1)
elig = [1, 6, 7, 24, 109, 356, 232, 54, 28, 12, 9, 8, 6, 4, 3, 1, 4, 4, 6, 30, 79, 64, 43, 7, 4, 2, 2, 0, 0]
nonelig = [1, 1, 10, 10, 54, 159, 105, 40, 16, 10, 9, 4, 2, 3, 1, 2, 4, 24, 50, 250, 708, 511, 230, 49, 21, 10, 5, 2, 1]
fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0), dpi=300, sharex=True)
for ax, counts, col, title in [
        (axes[0], elig, BLUE, "Eligible for early testing (BMI ≥30, family history, previous GDM or high-risk ethnicity)\nn = 1163; first O'Sullivan before 20 weeks in 74%"),
        (axes[1], nonelig, GREY, "Not eligible\nn = 2389; first O'Sullivan before 20 weeks in 18%")]:
    ax.bar(bins[:-1], counts, width=0.95, align='edge', color=col, edgecolor='white', linewidth=0.6)
    ax.axvline(20, color=INK, ls=(0, (3, 2)), lw=1)
    ax.set_title(title, loc='left', fontsize=10.5)
    ax.set_ylabel('Women')
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
axes[1].set_xlabel("Gestational week of first O'Sullivan test")
axes[1].set_xlim(3.5, 33.5); axes[1].set_xticks([5, 10, 15, 20, 25, 30])
fig.tight_layout()
fig.savefig('Fig2_timing.png', facecolor='white')

# ---------- Fig 3 ----------
rows = [
    ('(a) Multicentre cohort (10 hospitals)', None, None, None),
    ('All women, untested counted as not early', (0.788, 0.718, 0.849), (0.627, 0.563, 0.692), '0.16 (0.06, 0.25)'),
    ('Only women eligible for early testing', (0.594, 0.492, 0.696), (0.595, 0.488, 0.696), '0.00 (−0.15, 0.14)'),
    ('(b) Single-centre cohort (individual test dates)', None, None, None),
    ('All women, untested counted as not early', (0.817, 0.762, 0.871), (0.642, 0.590, 0.693), '0.18 (0.10, 0.25)'),
    ('Only women actually tested early', (0.618, 0.522, 0.718), (0.579, 0.503, 0.654), '0.04 (−0.09, 0.16)'),
    ('Corrected: multiple imputation (primary)', (0.715, 0.627, 0.800), (0.659, 0.607, 0.716), '0.06 (−0.06, 0.16)'),
    ('Corrected: inverse probability weighting', (0.770, 0.705, 0.837), (0.620, 0.546, 0.707), '0.15 (0.04, 0.25)'),
]
fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=200)
y = 0; ys = []; labels = []
for lab, e, l, dtxt in rows:
    if e is None:
        y -= 0.3 if y < 0 else 0
        ax.text(-0.012, y, lab, ha='right', va='center', fontweight='bold', fontsize=12, transform=ax.get_yaxis_transform())
        y -= 0.75; continue
    ax.errorbar(e[0], y + 0.15, xerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt='o', color=BLUE, ms=8, mec='white', lw=2, capsize=0,
                label='Early-onset GDM' if not ys else None)
    ax.errorbar(l[0], y - 0.15, xerr=[[l[0] - l[1]], [l[2] - l[0]]], fmt='s', color=ORANGE, ms=8, mec='white', lw=2, capsize=0,
                label='Late-onset GDM' if not ys else None)
    ax.text(1.02, y, dtxt, ha='left', va='center', fontsize=11.5, transform=ax.get_yaxis_transform())
    ys.append(y); labels.append(lab); y -= 1
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=12)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0.3, 0.9); ax.set_ylim(y + 0.4, 0.5)
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.set_xlabel('AUC of the clinical model (95% CI)', fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.09), frameon=False, fontsize=11, ncol=2)
ax.text(1.02, 0.5, 'Difference,\nearly − late (95% CI)', ha='left', va='bottom', fontsize=11, fontweight='bold', transform=ax.get_yaxis_transform())
fig.tight_layout()
fig.savefig('Fig3_auc.png', facecolor='white')
print('ok')
