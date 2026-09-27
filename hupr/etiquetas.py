"""Re-etiquetado de la DG en la cohorte HUPR a partir de los valores de la propia SOG de 100 g.
Umbrales NDDG (105/190/165/145 mg/dl) o Carpenter-Coustan (95/180/155/140).
La glucemia basal solo se usa si procede de la misma petición o del mismo día (dias_basal_curva==0);
si no, se exigen 2 valores alterados entre los tres postcarga."""
import pandas as pd, numpy as np
BASAL_REAL=True
UMB={'nddg':[105,190,165,145],'cc':[95,180,155,140]}
def curvas_etiquetadas(criterio='nddg', basal_prestada=False):
    c=pd.read_pickle('curvas.pkl').copy()
    pe=pd.read_pickle('peticiones.pkl')[['num_peticion','episodio_id']]
    b=c.glucosa_basal.where((c.glucosa_basal_origen=='misma_peticion')|(c.dias_basal_curva==0)) if not basal_prestada else c.glucosa_basal
    # v3 (27 sep 2026): glucemia basal real de la SOG (0 min, laboratorio HUPR, J.D. Santotoribio), enlazada por petición+NTS+fecha.
    # Si la curva no está en el fichero del laboratorio se mantiene la regla anterior.
    if BASAL_REAL and not basal_prestada:
        r=pd.read_pickle('curvas_basal_real.pkl')[['num_peticion','NTS','Fecha Petición','glucosa_basal']].rename(columns={'Fecha Petición':'fecha_peticion','glucosa_basal':'basal_real'})
        c=c.merge(r,on=['num_peticion','NTS','fecha_peticion'],how='left')
        c['basal_en_lab']=c.basal_real.notna()
        b=c.basal_real.where(c.basal_en_lab, b.values)
    g=pd.concat([b,c.glucosa_60,c.glucosa_120,c.glucosa_180],axis=1)
    c['n_alt']=(g.values>=np.array(UMB[criterio])).sum(axis=1)
    c['n_val']=g.notna().sum(axis=1)
    c['dg']=(c.n_alt>=2).astype(int)
    c['evaluable']=(c.n_val>=3)|(c.dg==1)
    c=c.merge(pe,on='num_peticion',how='inner')
    return c
def etiquetar_episodios(m, criterio='nddg', basal_prestada=False):
    c=curvas_etiquetadas(criterio, basal_prestada)
    c=c[c.episodio_id.isin(m.episodio_id)]
    pos=c[c.dg==1].sort_values('fecha_peticion').groupby('episodio_id').fecha_peticion.first()
    m=m.copy()
    m['DG']=m.episodio_id.isin(pos.index).astype(int)
    m['fecha_dx']=pd.to_datetime(m.episodio_id.map(pos))
    return m
