"""Análisis de sensibilidad del sesgo de verificación en la cohorte HUPR."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from cohorte_hupr import construir
from analisis_hupr import A_PRECOZ, A_TARDIA, lp
X0=['edad','IMC','ant_fam_diabetes','multipara','DG_previa_lab','glu','glu_mis']
def preparar(corte=20, imc_imputado=False, rng=None, criterio='nddg', basal_prestada=False):
    m=construir(corte, verbose=False, criterio=criterio, basal_prestada=basal_prestada)
    m=m[m.subtipo!='no_clasif'].copy()
    m['glu_mis']=m.glucosa_basal_1t.isna().astype(int); m['glu']=m.glucosa_basal_1t.fillna(m.glucosa_basal_1t.median())
    m['anio']=m.fecha_fin.dt.year
    m['early']=(m.subtipo=='precoz').astype(int); m['late']=(m.subtipo=='tardia').astype(int)
    d=m.dropna(subset=['edad','ant_fam_diabetes','para']).copy()
    if imc_imputado:
        rng=rng or np.random.default_rng(0)
        Z=['edad','multipara','ant_fam_diabetes','glu','glu_mis','test_precoz','early','late']
        obs=d[d.IMC.notna()]; f=sm.OLS(np.log(obs.IMC),sm.add_constant(obs[Z])).fit()
        mis=d.IMC.isna()
        d.loc[mis,'IMC']=np.exp(f.predict(sm.add_constant(d.loc[mis,Z],has_constant='add'))+rng.normal(0,np.sqrt(f.scale),mis.sum()))
    else:
        d=d.dropna(subset=['IMC'])
    d['obesa']=(d.IMC>=30).astype(int)
    d['PAI']=((d.obesa==1)|(d.ant_fam_diabetes==1)|(d.DG_previa_lab==1)|(d.etnia_riesgo==1)).astype(int)
    d['edad35']=(d.edad>=35).astype(int)
    d['lp_p']=lp(d,A_PRECOZ); d['lp_t']=lp(d,A_TARDIA)
    return d.reset_index(drop=True)
def A(y,s,w=None): return roc_auc_score(y,s,sample_weight=w)
def fitp(y,X):
    X=X.loc[:,X.std()>0]
    return sm.GLM(y,sm.add_constant(X,has_constant='add'),family=sm.families.Binomial()).fit()
def estimar(d, persist=1.0, delta=0.0, trim=None, flags=False, M=10, rng=None):
    rng=rng or np.random.default_rng(1); r={}
    e=d[d.late==0]; l=d[d.early==0]
    r['naive']=A(e.early,e.lp_p)-A(l.late,l.lp_t)
    v=d[d.test_precoz==1]; ve=v[v.late==0]; vl=v[v.early==0]
    r['restr']=A(ve.early,ve.lp_p)-A(vl.late,vl.lp_t)
    Xv=X0+(['PAI','edad35'] if flags else [])
    XX=pd.concat([d[Xv],pd.get_dummies(d.anio,prefix='a',drop_first=True).astype(float)],axis=1)
    fv=fitp(d.test_precoz,XX); pi=fv.predict(sm.add_constant(XX[fv.model.exog_names[1:]],has_constant='add'))
    w=1/pi[d.test_precoz==1]
    if trim: w=np.minimum(w,np.quantile(w,trim))
    v=v.assign(w=w.values); ve=v[v.late==0]; vl=v[v.early==0]
    r['ipw']=A(ve.early,ve.lp_p,ve.w)-A(vl.late,vl.lp_t,vl.w)
    r['ipw_early']=A(ve.early,ve.lp_p,ve.w)
    # IM con persistencia p y desplazamiento delta
    fE=fitp(v.early,v[X0]); vn=v[v.early==0]; fL=fitp(vn.late,vn[X0])
    u=d[d.test_precoz==0]
    pE=fE.predict(sm.add_constant(u[fE.model.exog_names[1:]],has_constant='add')).values
    pL=fL.predict(sm.add_constant(u[fL.model.exog_names[1:]],has_constant='add')).values
    lateu=u.late.values==1
    pr=np.where(lateu, pE*persist/(pE*persist+(1-pE)*pL), pE*(1-persist)/(pE*(1-persist)+(1-pE)*(1-pL)))
    pr=np.clip(pr,1e-9,1-1e-9); pr=1/(1+np.exp(-(np.log(pr/(1-pr))+delta)))
    if persist>=1: pr=np.where(lateu,pr,0.0)
    dif=[];pct=[];ae=[]
    for k in range(M):
        imp=rng.random(len(u))<pr
        yE=d.early.values.copy(); yL=d.late.values.copy()
        idx=u.index.values[imp]; yE[idx]=1; yL[idx]=0
        mE=yL==0; mL=yE==0
        ae.append(A(yE[mE],d.lp_p.values[mE])); dif.append(ae[-1]-A(yL[mL],d.lp_t.values[mL]))
        pct.append(yE.sum()/(yE.sum()+yL.sum()))
    r['mi']=np.mean(dif); r['mi_early']=np.mean(ae); r['pct_mi']=np.mean(pct)
    r['pct_naive']=d.early.sum()/(d.early.sum()+d.late.sum())
    r['n']=len(d); r['n_early']=int(d.early.sum()); r['n_late']=int(d.late.sum())
    return r
