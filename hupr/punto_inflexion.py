"""Tipping-point analysis: difference after multiple imputation for decreasing values of delta (lower odds of early-onset GDM in untested women)."""
import json
from sensibilidad_hupr import *
d = preparar(20); out = {}
for dl in [0, -1, -2, -3, -4, -5, -6]:
    r = estimar(d, delta=dl, M=50); out[dl] = dict(mi=float(r['mi']), pct_mi=float(r['pct_mi'])); print(dl, round(r['mi'], 3), round(r['pct_mi'], 3))
json.dump(out, open('punto_inflexion.json', 'w'), indent=1)
