"""Cohorte HUPR 2020-2025 enlazada laboratorio-clínica: verificación precoz observada individualmente.
Edad gestacional por FUR clínica (fecha_fin - semanas_gestacion*7). Test precoz = O'Sullivan o SOG < 20 semanas.
DG precoz = SOG diagnóstica < 20 semanas; DG tardía = >= 20 semanas (distribución bimodal, sin tests entre 20-21).
v2 (24 sep 2026): DG por umbrales NDDG aplicados a los valores de la propia SOG (criterio='nddg'); criterio='orig' reproduce la v1 (Carpenter-Coustan con basal prestada)."""
import pandas as pd, numpy as np
CORTE = 20
def construir(corte=CORTE, verbose=True, criterio='nddg', basal_prestada=False):
    ep=pd.read_pickle("episodios.pkl"); pe=pd.read_pickle("peticiones.pkl")
    if criterio!='orig':
        from etiquetas import curvas_etiquetadas
        cv=curvas_etiquetadas(criterio, basal_prestada)
        pos=cv[cv.dg==1].sort_values('fecha_peticion').groupby('episodio_id').fecha_peticion.first()
        ep=ep.copy(); ep['fenotipo_DG_episodio']=np.where(ep.episodio_id.isin(pos.index),'DG_confirmada','no_DG')
        ep['fecha_curva']=pd.to_datetime(ep.episodio_id.map(pos)).where(ep.episodio_id.isin(pos.index), ep['fecha_curva'])
    m=ep[(ep.parto_casado==True)&ep.fur_clinica_calc.notna()].copy()
    p=pe[pe.tipo_peticion.isin(['osullivan','curva_3h_100g'])].merge(m[['episodio_id','fur_clinica_calc']],on='episodio_id')
    p['sg']=(pd.to_datetime(p.fecha_peticion)-p.fur_clinica_calc).dt.days/7
    p=p[(p.sg>=4)&(p.sg<=42)]
    early=p[p.sg<corte].groupby('episodio_id').size()
    m['test_precoz']=m.episodio_id.isin(early.index).astype(int)
    m['sg_prim_osull']=m.episodio_id.map(p[p.tipo_peticion=='osullivan'].groupby('episodio_id').sg.min())
    m['sg_dx']=(m.fecha_curva-m.fur_clinica_calc).dt.days/7
    m['DG']=(m.fenotipo_DG_episodio=='DG_confirmada').astype(int)
    m['subtipo']=np.select([m.DG==0,(m.DG==1)&(m.sg_dx<corte),(m.DG==1)&(m.sg_dx>=corte)],['control','precoz','tardia'],'no_clasif')
    # DG previa (proxy): episodio previo de la misma mujer en el laboratorio con DG confirmada
    allp=ep[['NTS','episodio_id','fur_lab','fenotipo_DG_episodio']].copy()
    prev=m[['episodio_id','NTS','fur_lab']].merge(allp,on='NTS',suffixes=('','_o'))
    prev=prev[(prev.fur_lab_o<prev.fur_lab-pd.Timedelta(days=200))&(prev.fenotipo_DG_episodio=='DG_confirmada')]
    m['DG_previa_lab']=m.episodio_id.isin(prev.episodio_id).astype(int)
    et=m.GRUPO_ETNICO.astype(str).str.upper()
    m['etnia_riesgo']=et.str.contains('LATINO|ARABE|ÁRABE|ASIAT|INDI|PAKIST|NEGR|AFRIC|MAGREB',regex=True).astype(int)
    m['multipara']=(m.para>=1).astype(float).where(m.para.notna())
    m['obesa']=(m.IMC>=30).astype(float).where(m.IMC.notna())
    m['PAI']=((m.obesa==1)|(m.ant_fam_diabetes==1)|(m.DG_previa_lab==1)|(m.etnia_riesgo==1)).astype(int)
    if verbose:
        print("episodios con parto y FUR clínica:",len(m))
        print("test precoz <%d s:"%corte, m.test_precoz.sum(), "| subtipos:", m.subtipo.value_counts().to_dict())
        print("faltan: edad %d IMC %d AF %d para %d"%(m.edad.isna().sum(),m.IMC.isna().sum(),m.ant_fam_diabetes.isna().sum(),m.para.isna().sum()))
        print("DG previa (proxy lab):",m.DG_previa_lab.sum()," etnia riesgo:",m.etnia_riesgo.sum())
    return m
if __name__=="__main__":
    m=construir(); m.to_pickle("cohorte_hupr.pkl")
