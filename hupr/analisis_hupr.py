import pandas as pd, numpy as np, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
A_PRECOZ=dict(c=-10.3436,edad=0.0809,IMC=0.1117,af=1.0900,mp=-0.1035)
A_TARDIA=dict(c=-6.9871,edad=0.0728,IMC=0.0508,af=0.5058,mp=0.1045)
A_TOTAL=dict(c=-7.3336,edad=0.0751,IMC=0.0690,af=0.6610,mp=0.0610)
def lp(d,m): return m['c']+m['edad']*d.edad+m['IMC']*d.IMC+m['af']*d.ant_fam_diabetes+m['mp']*d.multipara
def cargar():
    m=pd.read_pickle("cohorte_hupr.pkl")
    d=m.dropna(subset=['edad','IMC','ant_fam_diabetes','para']).copy()
    d=d[d.subtipo!='no_clasif'].copy()
    d['lp_p']=lp(d,A_PRECOZ); d['lp_t']=lp(d,A_TARDIA); d['lp_tot']=lp(d,A_TOTAL)
    d['early']=(d.subtipo=='precoz').astype(int); d['late']=(d.subtipo=='tardia').astype(int)
    d['glu_mis']=d.glucosa_basal_1t.isna().astype(int)
    d['glu']=d.glucosa_basal_1t.fillna(d.glucosa_basal_1t.median())
    d['anio']=d.fecha_fin.dt.year
    return d
VX=['edad','IMC','ant_fam_diabetes','multipara','DG_previa_lab','etnia_riesgo','glu','glu_mis']
def pi_verif(d):
    X=sm.add_constant(pd.concat([d[VX],pd.get_dummies(d.anio,prefix='a',drop_first=True).astype(float)],axis=1))
    f=sm.GLM(d.test_precoz,X,family=sm.families.Binomial()).fit()
    return f.predict(X), f
def auc_w(y,s,w=None): return roc_auc_score(y,s,sample_weight=w)
MX=['edad','IMC','ant_fam_diabetes','multipara','glu','glu_mis']
def estimar(d, M=20, rng=None, trim=None):
    """Devuelve dict de AUC: ingenuo, restringido, IPW, IM para DG precoz y tardía (armonizados) y marcas."""
    rng = rng or np.random.default_rng(1)
    out={}
    pi,_=pi_verif(d); d=d.assign(pi=pi)
    w=1/d.pi
    if trim: w=np.minimum(w, np.quantile(w[d.test_precoz==1], trim))
    d=d.assign(w=w)
    # ingenuo (toda la cohorte; no testadas = no precoz)
    e=d[d.late==0]; l=d[d.early==0]
    out['naive_early']=auc_w(e.early,e.lp_p); out['naive_late']=auc_w(l.late,l.lp_t)
    out['PAI_flag_early']=auc_w(e.early,e.PAI)
    # restringido a testadas precozmente
    v=d[d.test_precoz==1]; ve=v[v.late==0]; vl=v[v.early==0]
    out['restr_early']=auc_w(ve.early,ve.lp_p); out['restr_late']=auc_w(vl.late,vl.lp_t)
    out['ipw_early']=auc_w(ve.early,ve.lp_p,ve.w); out['ipw_late']=auc_w(vl.late,vl.lp_t,vl.w)
    # IM: imputar precoz/tardía en DG de no testadas
    g=v[(v.early==1)|(v.late==1)]
    X=sm.add_constant(g[MX]); f=sm.GLM(g.early,X,family=sm.families.Binomial()).fit()
    u=d[(d.test_precoz==0)&(d.late==1)]
    beta=f.params.values; cov=f.cov_params().values
    ae=[];al=[];nimp=[];ph=[]
    for k in range(M):
        b=rng.multivariate_normal(beta,cov)
        pu=1/(1+np.exp(-(sm.add_constant(u[MX],has_constant='add').values@b)))
        imp=rng.random(len(u))<pu
        dd=d.copy(); dd.loc[u.index[imp],'early']=1; dd.loc[u.index[imp],'late']=0
        e2=dd[dd.late==0]; l2=dd[dd.early==0]
        ae.append(auc_w(e2.early,e2.lp_p)); al.append(auc_w(l2.late,l2.lp_t)); nimp.append(imp.sum())
        ph.append({v_: dd[dd.early==1][v_].mean()-dd[dd.late==1][v_].mean() for v_ in ['edad','IMC','ant_fam_diabetes']})
    out['mi_early']=np.mean(ae); out['mi_late']=np.mean(al); out['mi_n_reclasif']=np.mean(nimp)
    out['n_late_no_testadas']=len(u)
    for v_ in ['edad','IMC','ant_fam_diabetes']:
        out['naive_diff_'+v_]=d[d.early==1][v_].mean()-d[d.late==1][v_].mean()
        out['mi_diff_'+v_]=np.mean([p[v_] for p in ph])
        # IPW phenotype among verified GDM
        gg=v[(v.early==1)|(v.late==1)]
        out['ipw_diff_'+v_]=np.average(gg[gg.early==1][v_],weights=gg[gg.early==1].w)-np.average(gg[gg.late==1][v_],weights=gg[gg.late==1].w)
    out['pct_early_naive']=d.early.sum()/(d.early.sum()+d.late.sum())
    out['pct_early_mi']=(d.early.sum()+np.mean(nimp))/(d.early.sum()+d.late.sum())
    out['pct_early_ipw']=np.average(v.early[(v.early+v.late)>0],weights=v.w[(v.early+v.late)>0])
    out['total_auc']=auc_w(d.DG,d.lp_tot)
    return out
