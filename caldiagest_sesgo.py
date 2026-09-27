"""Multicentre cohort (CALDIAGEST): verification-bias analysis.
Uncorrected AUCs of the fixed early-onset and late-onset models (each subtype vs women without GDM, excluding the
other subtype and GDM without timing), AUCs restricted to women eligible for early testing under the Andalusian
protocol (BMI >=30, previous GDM or first-degree family history of diabetes), AUC of eligibility alone, proportion of
GDM diagnosed early by eligibility and early-late differences in BMI, family history and age.
Percentile bootstrap CIs (500 resamples, fixed seed). Sensitivity analyses: excluding the Puerto Real and Madrid
sites, and assigning GDM without timing to early-onset or to late-onset GDM."""
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cohorte import cargar

SITE_PUERTO_REAL = 'hospital_universitc'; SITE_MADRID = 'hospital_universit'

def preparar(df, unclassified=None):
    d = df.copy()
    d['prev_gdm'] = (d.ant_diab_gest == 1).astype(int)
    d['eligible'] = ((d.IMC >= 30) | (d.prev_gdm == 1) | (d.ant_fam_diabetes == 1)).astype(int)
    if unclassified == 'early': d.loc[d.subtipo == 'no_clasif', 'subtipo'] = 'precoz'
    if unclassified == 'late': d.loc[d.subtipo == 'no_clasif', 'subtipo'] = 'tardia'
    return d.reset_index(drop=True)

def auc(d, sub, score):
    s = d[d.subtipo.isin(['control', sub])]
    return roc_auc_score((s.subtipo == sub).astype(int), s[score])

def estimar(d):
    r = {}
    r['naive_early'] = auc(d, 'precoz', 'lp_precoz'); r['naive_late'] = auc(d, 'tardia', 'lp_tardia')
    e = d[d.eligible == 1]
    r['restr_early'] = auc(e, 'precoz', 'lp_precoz'); r['restr_late'] = auc(e, 'tardia', 'lp_tardia')
    r['flag_early'] = auc(d, 'precoz', 'eligible')
    r['overall'] = roc_auc_score(d.diabetes, d.lp_total)
    r['naive_dif'] = r['naive_early'] - r['naive_late']; r['restr_dif'] = r['restr_early'] - r['restr_late']
    gdm = d[d.subtipo != 'control']
    for k, g in [('elig', gdm[gdm.eligible == 1]), ('nonelig', gdm[gdm.eligible == 0])]:
        r['pct_early_' + k] = (g.subtipo == 'precoz').mean()
    for lab, dd in [('naive', d), ('restr', e)]:
        for v, col in [('bmi', 'IMC'), ('fh', 'ant_fam_diabetes'), ('age', 'edad')]:
            r[f'{lab}_diff_{v}'] = dd[dd.subtipo == 'precoz'][col].mean() - dd[dd.subtipo == 'tardia'][col].mean()
    r['n'] = len(d); r['n_early'] = int((d.subtipo == 'precoz').sum()); r['n_late'] = int((d.subtipo == 'tardia').sum())
    return r

def con_ic(d, B=500, seed=2026):
    pt = estimar(d); rng = np.random.default_rng(seed); res = []
    for b in range(B):
        bs = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        try: res.append(estimar(bs))
        except ValueError: pass
    R = pd.DataFrame(res)
    out = {k: [float(pt[k]), float(R[k].quantile(.025)), float(R[k].quantile(.975))] for k in pt if not k.startswith('n')}
    out.update({k: pt[k] for k in ['n', 'n_early', 'n_late']}); out['B'] = len(R)
    return out

if __name__ == '__main__':
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    df = cargar(False)
    out = {'primary': con_ic(preparar(df), B)}
    out['excluding_PuertoReal_Madrid'] = con_ic(preparar(df[~df.redcap_data_access_group.isin([SITE_PUERTO_REAL, SITE_MADRID])]), B)
    out['unclassified_as_early'] = con_ic(preparar(df, 'early'), B)
    out['unclassified_as_late'] = con_ic(preparar(df, 'late'), B)
    json.dump(out, open('caldiagest_sesgo.json', 'w'), indent=1)
    for k, v in out.items():
        print(k, {kk: [round(x, 3) for x in vv] if isinstance(vv, list) else vv for kk, vv in v.items() if kk in ['naive_dif', 'restr_dif', 'naive_early', 'naive_late', 'restr_early', 'restr_late', 'flag_early', 'overall', 'n', 'n_early', 'n_late']})
