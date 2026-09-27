"""Combine the primary point estimate (final_point.json) with the bootstrap chunks (final_boot_*.csv) into percentile CIs."""
import json, glob, pandas as pd
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('final_boot_*.csv'))])
pt = json.load(open('final_point.json'))
out = {k: [pt[k], float(R[k].quantile(.025)), float(R[k].quantile(.975))] for k in pt}; out['B'] = len(R)
json.dump(out, open('final_hupr_ci.json', 'w'), indent=1)
for k, v in out.items(): print(k, [round(x, 3) for x in v] if isinstance(v, list) else v)
