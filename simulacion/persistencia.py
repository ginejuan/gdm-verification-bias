import pandas as pd, numpy as np, json
from sim_verificacion import correr
out={}
for esc in ['S1','S2']:
    for p in [0.5,0.75,1.0]:
        R=correr(esc,'PAI',reps=300,N=4000,seed=int(p*100)+(0 if esc=='S1' else 7),fuera=0.12,persistencia=p)
        dif=lambda a: (R[a+'_precoz']-R[a+'_tardia'])
        out[f'{esc}_p{p}']={k:round(float(dif(k).mean()),3) for k in ['verdad','ingenuo','restr','ipw','im']}
        out[f'{esc}_p{p}']['pct_obs']=round(float(R.pct_precoz_obs.mean()),3)
        print(esc,p,out[f'{esc}_p{p}'],flush=True)
json.dump(out,open('persistencia.json','w'),indent=1)
