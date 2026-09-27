"""Diagnósticos de la IPW y comparación con la imputación múltiple (respuesta a revisores, 24 sep 2026)."""
import json, warnings; warnings.filterwarnings('ignore')
from sensibilidad_hupr import *
def diag(criterio='nddg'):
    d=preparar(20, criterio=criterio)
    d['grp']=np.select([d.PAI==1,d.edad35==1],['elig','edad'],'ninguno')
    XX=pd.concat([d[X0],pd.get_dummies(d.anio,prefix='a',drop_first=True).astype(float)],axis=1)
    fv=fitp(d.test_precoz,XX); pi=fv.predict(sm.add_constant(XX[fv.model.exog_names[1:]],has_constant='add'))
    d['pi']=pi; d['w']=1/pi
    v=d[d.test_precoz==1]; w=v.w
    R={}
    R['w_min'],R['w_p50'],R['w_p90'],R['w_p99'],R['w_max']=[float(x) for x in np.quantile(w,[0,.5,.9,.99,1])]
    R['w_gt10']=int((w>10).sum()); R['w_gt20']=int((w>20).sum()); R['n_tested']=int(len(v))
    R['ess_tested']=float(w.sum()**2/(w**2).sum())
    R['pi_min']=float(d.pi.min()); R['pi_lt005']=int((d.pi<0.05).sum())
    e=v[v.early==1]
    R['ess_early']=float(e.w.sum()**2/(e.w**2).sum()); R['n_early']=int(len(e))
    # implied hidden early cases among untested, by group, vs untested late cases observed
    for g in ['elig','edad','ninguno']:
        eg=e[e.grp==g]
        R[f'ipw_early_obs_{g}']=int(len(eg)); R[f'ipw_early_implied_{g}']=float(eg.w.sum())
        R[f'untested_late_{g}']=int(((d.grp==g)&(d.test_precoz==0)&(d.late==1)).sum())
        R[f'untested_n_{g}']=int(((d.grp==g)&(d.test_precoz==0)).sum())
    R['ipw_hidden_total']=float(e.w.sum()-len(e)); R['untested_late_total']=int(((d.test_precoz==0)&(d.late==1)).sum())
    # influence: drop each non-eligible early case
    base=estimar(d,M=20)['ipw']; R['ipw_base']=float(base)
    loo=[]
    for i in e[e.grp!='elig'].index:
        loo.append(float(estimar(d.drop(i).reset_index(drop=True),M=5)['ipw']))
    R['ipw_loo_lowrisk']=loo
    # MI implied reclassified by group (one run M=50)
    return R
out={'nddg':diag('nddg')}
json.dump(out,open('diagnosticos_ipw.json','w'),indent=1)
print(json.dumps(out,indent=1))
