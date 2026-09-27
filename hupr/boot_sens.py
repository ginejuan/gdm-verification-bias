import sys, json, zlib
from sensibilidad_hupr import *
esc=sys.argv[1]; B=int(sys.argv[2])
base={'corte':20,'imc':False,'sub':None,'kw':{},'crit':'nddg','bp':False}
conf={'base':{},'p90':{'kw':{'persist':0.9}},'p75':{'kw':{'persist':0.75}},'p50':{'kw':{'persist':0.5}},
      'delta_m1':{'kw':{'delta':-1}},'delta_p1':{'kw':{'delta':1}},'flags':{'kw':{'flags':True}},'trim99':{'kw':{'trim':0.99}},
      'corte24':{'corte':24},'imc_imp':{'imc':True},'analitica1T':{'sub':'glu'},'sin2020':{'sub':'2021'},'cc':{'crit':'cc'},'orig':{'crit':'orig'},'nddg_prest':{'bp':True},
      'delta_m2':{'kw':{'delta':-2}},'delta_m3':{'kw':{'delta':-3}},'delta_m4':{'kw':{'delta':-4}}}[esc]
c={**base,**conf}
rng=np.random.default_rng(zlib.crc32(esc.encode()))  # semilla reproducible
def datos(r):
    d=preparar(c['corte'],imc_imputado=c['imc'],rng=r,criterio=c['crit'],basal_prestada=c['bp'])
    if c['sub']=='glu': d=d[d.glu_mis==0].reset_index(drop=True)
    if c['sub']=='2021': d=d[d.anio>=2021].reset_index(drop=True)
    return d
d=datos(np.random.default_rng(0))
pt=estimar(d,**c['kw'],M=20)
res=[]
for b in range(B):
    bs=d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))).reset_index(drop=True)
    try: res.append(estimar(bs,**c['kw'],M=10,rng=rng))
    except Exception as e: pass
R=pd.DataFrame(res)
out={k:[float(pt[k]),float(R[k].quantile(.025)),float(R[k].quantile(.975))] for k in ['naive','restr','ipw','mi','pct_naive','pct_mi']}
out['n']=pt['n']; out['n_early']=pt['n_early']; out['n_late']=pt['n_late']; out['B']=len(R)
json.dump(out,open(f"sens_{esc}.json","w"),indent=1); print(esc,json.dumps({k:[round(x,3) for x in v] if isinstance(v,list) else v for k,v in out.items()}))
