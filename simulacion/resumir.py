"""Summarise the simulation (ESM Table 3) and draw Fig. 4 from res_*.csv (output of correr_todo.py)."""
import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = pd.concat([pd.read_csv(f'res_{i}.csv') for i in range(10)], ignore_index=True)
R.loc[R.usar_glu == False, 'politica'] = R.politica + ' (MNAR)'
rows = []
for (esc, pol), g in R.groupby(['escenario', 'politica'], sort=False):
    r = dict(esc=esc, politica=pol, testadas=g.pct_testadas.mean())
    for k in ['verdad', 'ingenuo', 'restr', 'ipw', 'im']:
        d = g[k + '_precoz'] - g[k + '_tardia']
        r[k + '_dif'] = d.mean(); r[k + '_lo'] = d.quantile(.025); r[k + '_hi'] = d.quantile(.975)
        r[k + '_p'] = g[k + '_precoz'].mean(); r[k + '_t'] = g[k + '_tardia'].mean()
    r.update(pct_v=g.pct_precoz_verdad.mean(), pct_o=g.pct_precoz_obs.mean(), imc_v=g.dif_imc_verdad.mean(),
             imc_o=g.dif_imc_obs.mean(), af_v=g.dif_af_verdad.mean(), af_o=g.dif_af_obs.mean())
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv('resumen_simulacion_repro.csv', index=False)
print(S[['esc', 'politica', 'verdad_dif', 'ingenuo_dif', 'restr_dif', 'ipw_dif', 'im_dif', 'pct_o', 'imc_o', 'af_o']].round(3).to_string())

# Fig. 4
BLUE, ORANGE, GREY, INK, GRID = '#2a7ad6', '#e8673a', '#6b6b66', '#262624', '#ecebe6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': '#9a9a92', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
labs = [('verdad', 'Truth (everyone tested early)', GREY), ('ingenuo', 'Naive: untested counted as not early', ORANGE),
        ('restr', 'Restricted to women tested early', BLUE), ('ipw', 'Inverse probability weighting', BLUE),
        ('im', 'Multiple imputation', BLUE)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), dpi=258, sharey=True)
for ax, esc, title in [(axes[0], 'S1', 'Scenario 1: no true difference'), (axes[1], 'S2', 'Scenario 2: true difference ≈ 0.10')]:
    s = S[(S.esc == esc) & (S.politica == 'PAI')].iloc[0]
    for i, (k, lab, col) in enumerate(labs):
        y = -i
        ax.errorbar(s[k + '_dif'], y, xerr=[[s[k + '_dif'] - s[k + '_lo']], [s[k + '_hi'] - s[k + '_dif']]], fmt='o', color=col,
                    ms=7, mec='white', lw=2, capsize=0)
    ax.axvline(s['verdad_dif'], color=GREY, ls='--', lw=1.2)
    ax.set_title(title, fontsize=11); ax.set_xlim(-0.22, 0.26)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('AUC early − AUC late (mean, 95% range)')
axes[0].set_yticks([-i for i in range(len(labs))]); axes[0].set_yticklabels([l[1] for l in labs])
fig.tight_layout(); fig.savefig('Fig4_simulation.png', facecolor='white')
