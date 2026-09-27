import warnings; warnings.filterwarnings('ignore')
from sim_verificacion import *
import sys
def estim_flags(d, rng):
    d=d.copy(); d['eleg']=((d.imc>=30)|(d.af==1)|(d.dg_prev==1)).astype(int); d['glu90']=(d.glu>90).astype(int)
    X=['edad','imc','af','dg_prev','glu','eleg','glu90']
    v=d[d.test==1]
    lr=LogisticRegression(C=1e6,max_iter=2000).fit(d[X],d.test); pi=lr.predict_proba(d[X])[:,1]
    w=1/pi[d.test.values==1]
    return auc_arm(v.obs_precoz.values,v.obs_tardia.values,v.score.values,w), auc_arm(v.obs_tardia.values,v.obs_precoz.values,v.score.values,w)
def modelo_aprendido(d, rng):
    # desarrollar modelo con etiquetas observadas (ingenuas) en la mitad; evaluar en la otra mitad
    idx=rng.permutation(len(d)); a=d.iloc[idx[:len(d)//2]]; b=d.iloc[idx[len(d)//2:]]
    X=['edad','imc','af']
    ae=a[a.obs_tardia==0]; lr=LogisticRegression(C=1e6,max_iter=2000).fit(ae[X],ae.obs_precoz)
    at=a[a.tardia==0]; lrT=LogisticRegression(C=1e6,max_iter=2000).fit(at[X],at.precoz)
    s=lr.decision_function(b[X])
    return dict(coef_af_ing=lr.coef_[0][2],coef_af_ver=lrT.coef_[0][2],coef_imc_ing=lr.coef_[0][1],coef_imc_ver=lrT.coef_[0][1],
                auc_ing=auc_arm(b.obs_precoz.values,b.obs_tardia.values,s),auc_ver=auc_arm(b.precoz.values,b.tardia.values,s),
                auc_ver_score=auc_arm(b.precoz.values,b.tardia.values,b.score.values))
esc=sys.argv[1]; rng=np.random.default_rng(7 if esc=='S1' else 8); out=[]
for i in range(200):
    d=aplicar_politica(simular_poblacion(8000,esc,rng),'PAI',rng,fuera=0.12)
    p,t=estim_flags(d,rng); r=modelo_aprendido(d,rng); r.update(ipwf_p=p,ipwf_t=t); out.append(r)
R=pd.DataFrame(out); R.to_csv(f"extra_{esc}.csv",index=False); print(esc); print(R.mean().round(3).to_string()); print("ipwf sd",R.ipwf_p.std().round(3))
