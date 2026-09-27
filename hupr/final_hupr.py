"""Análisis principal consolidado para el manuscrito (imputación con marco de persistencia, p=1)."""
import sys, json
from sensibilidad_hupr import *
def est_full(d, persist=1.0, M=10, rng=None):
    rng=rng or np.random.default_rng(1); r={}
    e=d[d.late==0]; l=d[d.early==0]
    r['naive_early']=A(e.early,e.lp_p); r['naive_late']=A(l.late,l.lp_t)
    v=d[d.test_precoz==1]; ve=v[v.late==0]; vl=v[v.early==0]
    r['restr_early']=A(ve.early,ve.lp_p); r['restr_late']=A(vl.late,vl.lp_t)
    XX=pd.concat([d[X0],pd.get_dummies(d.anio,prefix='a',drop_first=True).astype(float)],axis=1)
    fv=fitp(d.test_precoz,XX); pi=fv.predict(sm.add_constant(XX[fv.model.exog_names[1:]],has_constant='add'))
    v=v.assign(w=(1/pi[d.test_precoz==1]).values); ve=v[v.late==0]; vl=v[v.early==0]
    r['ipw_early']=A(ve.early,ve.lp_p,ve.w); r['ipw_late']=A(vl.late,vl.lp_t,vl.w)
    fE=fitp(v.early,v[X0]); vn=v[v.early==0]; fL=fitp(vn.late,vn[X0]); u=d[d.test_precoz==0]
    pE=fE.predict(sm.add_constant(u[fE.model.exog_names[1:]],has_constant='add')).values
    pL=fL.predict(sm.add_constant(u[fL.model.exog_names[1:]],has_constant='add')).values
    lateu=u.late.values==1
    pr=np.where(lateu, pE*persist/(pE*persist+(1-pE)*pL), pE*(1-persist)/(pE*(1-persist)+(1-pE)*(1-pL)))
    if persist>=1: pr=np.where(lateu,pr,0.0)
    ae=[];al=[];pc=[];nre=[];ph={k:[] for k in ['IMC','ant_fam_diabetes','edad']}
    for k in range(M):
        imp=rng.random(len(u))<pr; yE=d.early.values.copy(); yL=d.late.values.copy()
        idx=u.index.values[imp]; yE[idx]=1; yL[idx]=0
        ae.append(A(yE[yL==0],d.lp_p.values[yL==0])); al.append(A(yL[yE==0],d.lp_t.values[yE==0]))
        pc.append(yE.sum()/(yE.sum()+yL.sum())); nre.append(imp.sum())
        for v_ in ph: ph[v_].append(d[v_].values[yE==1].mean()-d[v_].values[yL==1].mean())
    r['mi_early']=np.mean(ae); r['mi_late']=np.mean(al); r['pct_mi']=np.mean(pc); r['n_reclas']=np.mean(nre)
    r['pct_naive']=d.early.sum()/(d.early.sum()+d.late.sum())
    for v_ in ph:
        r['naive_diff_'+v_]=d[d.early==1][v_].mean()-d[d.late==1][v_].mean(); r['mi_diff_'+v_]=np.mean(ph[v_])
    for a in ['naive','restr','ipw','mi']: r[a+'_dif']=r[a+'_early']-r[a+'_late']
    r['flag_early']=A(e.early,e.PAI)
    return r
if __name__=="__main__":
    seed=int(sys.argv[1]); B=int(sys.argv[2]); d=preparar(20); rng=np.random.default_rng(seed)
    if B==0:
        r=est_full(d,M=50); json.dump({k:float(v) for k,v in r.items()},open("final_point.json","w"),indent=1); print({k:round(float(v),3) for k,v in r.items()})
    else:
        res=[]
        for b in range(B):
            bs=d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))).reset_index(drop=True)
            try: res.append(est_full(bs,M=10,rng=rng))
            except Exception: pass
        pd.DataFrame(res).to_csv(f"final_boot_{seed}.csv",index=False); print("done",len(res))
