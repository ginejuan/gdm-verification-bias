"""Aggregated counts for Fig. 2 (gestational week of the first GCT by eligibility) and AUC of the verification model."""
import json
from sensibilidad_hupr import *
d = preparar(20); out = {}; bins = np.arange(4, 34, 1)
for k, m in [('elig', d.PAI == 1), ('nonelig', d.PAI == 0)]:
    x = d[m].sg_prim_osull.dropna(); h, _ = np.histogram(x, bins=bins)
    out[k] = dict(n=int(m.sum()), pct_lt20=round(100 * (x < 20).sum() / m.sum(), 1), counts=h.tolist())
out['bins'] = bins.tolist()
XX = pd.concat([d[X0], pd.get_dummies(d.anio, prefix='a', drop_first=True).astype(float)], axis=1)
fv = fitp(d.test_precoz, XX); pi = fv.predict(sm.add_constant(XX[fv.model.exog_names[1:]], has_constant='add'))
out['auc_verif'] = round(A(d.test_precoz, pi), 3)
json.dump(out, open('fig2_counts_v2.json', 'w')); print(json.dumps(out))
