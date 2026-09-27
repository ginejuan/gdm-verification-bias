"""Tablas 1 y 2 del manuscrito (v2, 24 sep 2026): edad >=35; cohorte HUPR con DG por NDDG (valores propios de la SOG)."""
import sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from cohorte import cargar
sys.path.insert(0,'hupr')
def ms(x): return f"{x.mean():.1f} ({x.std():.1f})"
def mi(x,dec=1): q=x.quantile([.5,.25,.75]); return f"{q.iloc[0]:.{dec}f} ({q.iloc[1]:.{dec}f}–{q.iloc[2]:.{dec}f})"
def np_(b,n): return f"{int(b.sum())} ({100*b.sum()/n:.1f})"
def col(d, sub, glu=False):
    n=len(sub); r=dict(n=n, age=ms(sub.edad), age35=np_(sub.edad>=35,n), bmi=mi(sub.IMC), obese=np_(sub.IMC>=30,n),
      fh=np_(sub.ant_fam_diabetes==1,n), multip=np_(sub.multipara==1,n), prevgdm=np_(sub.prev==1,n),
      gdm=np_(sub.subtipo.isin(['precoz','tardia','no_clasif']),n), early=int((sub.subtipo=='precoz').sum()),
      late=int((sub.subtipo=='tardia').sum()), uncl=int((sub.subtipo=='no_clasif').sum()))
    if glu:
        g=sub.glucosa_basal_1t.dropna()/18.016; r['glu_mmol']=mi(g,2); r['glu_n']=len(g)
    return r
def grupos(sub, tested=None):
    out={}
    for k,m in [('elig',sub.elig==1),('age_only',(sub.elig==0)&(sub.edad>=35)),('none',(sub.elig==0)&(sub.edad<35))]:
        s=sub[m]; g=s.subtipo.isin(['precoz','tardia','no_clasif'])
        out[k]=dict(n=len(s), tested_pct=(round(100*s.test_precoz.mean(),1) if tested else None), gdm=np_(g,len(s)),
                    early=int((s.subtipo=='precoz').sum()), late=int((s.subtipo=='tardia').sum()), uncl=int((s.subtipo=='no_clasif').sum()),
                    pct_early=round(100*(s.subtipo=='precoz').sum()/g.sum(),1))
    return out
R={}
# CALDIAGEST
c=cargar(False); c['prev']=(c.ant_diab_gest==1).astype(int)
c['elig']=((c.IMC>=30)|(c.prev==1)|(c.ant_fam_diabetes==1)).astype(int)
R['CAL_all']=col(c,c); R['CAL_elig']=col(c,c[c.elig==1]); R['CAL_nonelig']=col(c,c[c.elig==0]); R['CAL_grupos']=grupos(c)
# HUPR
from sensibilidad_hupr import preparar
import os; os.chdir('hupr'); d=preparar(20); os.chdir('..')
d['prev']=d.DG_previa_lab; d['elig']=d.PAI
d['subtipo']=np.select([d.early==1,d.late==1],['precoz','tardia'],'control')
R['HUPR_all']=col(d,d,True); R['HUPR_tested']=col(d,d[d.test_precoz==1],True); R['HUPR_untested']=col(d,d[d.test_precoz==0],True)
R['HUPR_elig_n']=int(d.elig.sum()); R['HUPR_elig_tested']=int(((d.elig==1)&(d.test_precoz==1)).sum()); R['HUPR_untested_elig']=int(((d.elig==1)&(d.test_precoz==0)).sum())
R['HUPR_grupos']=grupos(d,True)
ne=d[d.elig==0]
R['glu_nonelig_tested_mgdl']=round(ne[ne.test_precoz==1].glucosa_basal_1t.mean(),1); R['glu_nonelig_untested_mgdl']=round(ne[ne.test_precoz==0].glucosa_basal_1t.mean(),1)
R['nonelig_tested_pct']=round(100*ne.test_precoz.mean(),1)
json.dump(R,open('tablas_v2.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(R,indent=0,ensure_ascii=False))
