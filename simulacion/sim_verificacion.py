"""Simulación del sesgo de verificación por test precoz selectivo de diabetes gestacional (DG).

Verdad conocida: cada gestante tiene (o no) DG, y si la tiene, un momento de inicio (precoz/tardía).
Se aplican distintas políticas de test precoz y se comparan estimadores del AUC de una
puntuación clínica (edad, IMC, antecedente familiar) para la DG precoz y la tardía.

Datos 100 % sintéticos; parámetros calibrados de forma aproximada a las cifras agregadas
de CALDIAGEST y del HUPR (prevalencias, proporción testada por grupo).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def simular_poblacion(N, escenario, rng):
    edad = rng.normal(32, 5, N).clip(16, 46)
    imc = np.exp(rng.normal(np.log(25.5), 0.19, N)).clip(16, 55)
    af = (rng.random(N) < 0.22).astype(int)
    multip = (rng.random(N) < 0.5).astype(int)
    dg_prev = ((rng.random(N) < 0.06) & (multip == 1)).astype(int)
    # riesgo de DG total (igual para precoz y tardía en S1)
    lp = -3.35 + 0.07*(edad-32) + 0.09*(imc-25.5) + 0.60*af + 1.60*dg_prev
    dg = rng.random(N) < 1/(1+np.exp(-lp))
    # momento de inicio
    if escenario == 'S1':      # sin diferencia real: P(precoz | DG) constante
        lp_s = np.full(N, np.log(0.30/0.70))
    else:                      # S2: la DG precoz depende más del IMC (diferencia real moderada)
        lp_s = np.log(0.30/0.70) + 0.10*(imc-25.5)
    precoz = dg & (rng.random(N) < 1/(1+np.exp(-lp_s)))
    tardia = dg & ~precoz
    # glucemia del primer trimestre (auxiliar, dispara test fuera de protocolo)
    glu = rng.normal(78 + 0.5*(imc-25.5), 7, N) + 9*precoz + 2*tardia
    score = 0.07*edad + 0.09*imc + 0.60*af          # puntuación clínica evaluada (sin DG previa)
    return pd.DataFrame(dict(edad=edad, imc=imc, af=af, multip=multip, dg_prev=dg_prev, glu=glu,
                             precoz=precoz.astype(int), tardia=tardia.astype(int), score=score))

def aplicar_politica(d, politica, rng, cumplimiento=0.75, fuera=0.08, persistencia=0.75):
    N = len(d); u = rng.random(N)
    if politica == 'universal':
        test = np.ones(N, bool)
    else:
        if politica == 'PAI':
            eleg = (d.imc >= 30) | (d.af == 1) | (d.dg_prev == 1)
        elif politica == 'PAI+edad':
            eleg = (d.imc >= 30) | (d.af == 1) | (d.dg_prev == 1) | (d.edad >= 35)
        elif politica == 'NICE':
            eleg = (d.dg_prev == 1)
        # test fuera de protocolo: aleatorio + desencadenado por glucemia alta
        fuera_p = fuera + 0.35*(d.glu > 90)
        test = np.where(eleg, u < cumplimiento, u < fuera_p)
    d = d.copy(); d['test'] = test.astype(int)
    # etiquetas observadas
    d['obs_precoz'] = ((d.precoz == 1) & d.test.astype(bool)).astype(int)
    persiste = rng.random(N) < persistencia
    no_detect = (d.precoz == 1) & ~d.test.astype(bool)
    d['obs_tardia'] = ((d.tardia == 1) | (no_detect & persiste)).astype(int)   # precoz no testada y persistente -> "tardía"
    return d

def auc_arm(y_pos, y_excl, s, w=None):
    m = y_excl == 0
    return roc_auc_score(y_pos[m], s[m], sample_weight=None if w is None else w[m])

def estimadores(d, rng, usar_glu=True, M=10):
    X = ['edad', 'imc', 'af', 'dg_prev'] + (['glu'] if usar_glu else [])
    r = {}
    s = d.score.values
    # verdad (todas testadas, detección perfecta)
    r['verdad_precoz'] = auc_arm(d.precoz.values, d.tardia.values, s)
    r['verdad_tardia'] = auc_arm(d.tardia.values, d.precoz.values, s)
    # ingenuo
    r['ingenuo_precoz'] = auc_arm(d.obs_precoz.values, d.obs_tardia.values, s)
    r['ingenuo_tardia'] = auc_arm(d.obs_tardia.values, d.obs_precoz.values, s)
    v = d[d.test == 1]
    r['restr_precoz'] = auc_arm(v.obs_precoz.values, v.obs_tardia.values, v.score.values)
    r['restr_tardia'] = auc_arm(v.obs_tardia.values, v.obs_precoz.values, v.score.values)
    # IPW
    if d.test.mean() < 1:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(d[X], d.test)
        pi = lr.predict_proba(d[X])[:, 1]
        w = 1/pi[d.test.values == 1]
        r['ipw_precoz'] = auc_arm(v.obs_precoz.values, v.obs_tardia.values, v.score.values, w)
        r['ipw_tardia'] = auc_arm(v.obs_tardia.values, v.obs_precoz.values, v.score.values, w)
        # IM: imputar precoz/tardía en DG "tardías" de no testadas
        g = v[(v.obs_precoz == 1) | (v.obs_tardia == 1)]
        u = d[(d.test == 0) & (d.obs_tardia == 1)]
        ap, at = [], []
        if g.obs_precoz.nunique() == 2 and len(u) > 0:
            lr2 = LogisticRegression(C=1e6, max_iter=1000).fit(g[X], g.obs_precoz)
            pu = lr2.predict_proba(u[X])[:, 1]
            for _ in range(M):
                imp = rng.random(len(u)) < pu
                yp = d.obs_precoz.values.copy(); yt = d.obs_tardia.values.copy()
                idx = d.index.get_indexer(u.index[imp])
                yp[idx] = 1; yt[idx] = 0
                ap.append(auc_arm(yp, yt, s)); at.append(auc_arm(yt, yp, s))
            r['im_precoz'] = np.mean(ap); r['im_tardia'] = np.mean(at)
        else:
            r['im_precoz'] = np.nan; r['im_tardia'] = np.nan
    else:
        for k in ['ipw', 'im']:
            r[k+'_precoz'] = r['ingenuo_precoz']; r[k+'_tardia'] = r['ingenuo_tardia']
    # proporción de DG precoz y fenotipo (IMC)
    r['pct_precoz_verdad'] = d.precoz.sum()/(d.precoz.sum()+d.tardia.sum())
    r['pct_precoz_obs'] = d.obs_precoz.sum()/(d.obs_precoz.sum()+d.obs_tardia.sum())
    r['dif_imc_verdad'] = d[d.precoz == 1].imc.mean() - d[d.tardia == 1].imc.mean()
    r['dif_imc_obs'] = d[d.obs_precoz == 1].imc.mean() - d[d.obs_tardia == 1].imc.mean()
    r['dif_af_verdad'] = d[d.precoz == 1].af.mean() - d[d.tardia == 1].af.mean()
    r['dif_af_obs'] = d[d.obs_precoz == 1].af.mean() - d[d.obs_tardia == 1].af.mean()
    r['pct_testadas'] = d.test.mean()
    return r

def correr(escenario, politica, reps=500, N=4000, usar_glu=True, seed=0, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(reps):
        d = aplicar_politica(simular_poblacion(N, escenario, rng), politica, rng, **kw)
        if d.obs_precoz.sum() < 3:
            continue
        r = estimadores(d, rng, usar_glu=usar_glu); r['rep'] = i
        out.append(r)
    R = pd.DataFrame(out)
    R['escenario'] = escenario; R['politica'] = politica; R['usar_glu'] = usar_glu
    return R
