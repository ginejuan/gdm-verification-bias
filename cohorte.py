"""Pipeline de cohorte CALDIAGEST verificado (ver memo cifras_validacion_verificadas, ago 2026)."""
import pyreadstat, numpy as np, pandas as pd
import os
SAV = os.environ.get("CALDIAGEST_SAV", "../DATOS CON DECIMALES A 12 DE MAYO DE 2022.sav")
A_PRECOZ = dict(intercept=-10.3436, edad=0.0809, IMC=0.1117, ant_fam_diabetes=1.0900, multipara=-0.1035)
A_TOTAL  = dict(intercept=-7.3336,  edad=0.0751, IMC=0.0690, ant_fam_diabetes=0.6610, multipara=0.0610)
A_TARDIA = dict(intercept=-6.9871,  edad=0.0728, IMC=0.0508, ant_fam_diabetes=0.5058, multipara=0.1045)

def lp(d, m):
    return (m['intercept'] + m['edad']*d.edad + m['IMC']*d.IMC
            + m['ant_fam_diabetes']*d.ant_fam_diabetes + m['multipara']*d.multipara)

def cargar(verbose=True):
    df, _ = pyreadstat.read_sav(SAV)
    n0 = len(df)
    df = df[df.redcap_data_access_group != 'hospital_universitb']        # Puerta del Mar
    n1 = len(df)
    df = df.drop_duplicates(subset=['edad','peso','talla','redcap_data_access_group','diabetes'], keep='first')
    n2 = len(df)
    df = df.dropna(subset=['para','diabetes'])
    n3 = len(df)
    df = df.copy()
    df['IMC'] = df.peso / (df.talla/100)**2
    df['multipara'] = (df.para >= 1).astype(int)
    df['subtipo'] = np.select([df.diabetes==0, df.early_onset_gdm==1, df.late_onset_gdm==1],
                              ['control','precoz','tardia'], 'no_clasif')
    df['lp_precoz'] = lp(df, A_PRECOZ); df['lp_total'] = lp(df, A_TOTAL); df['lp_tardia'] = lp(df, A_TARDIA)
    for k in ['precoz','total','tardia']:
        df['p_'+k] = 1/(1+np.exp(-df['lp_'+k]))
    if verbose:
        print(f"{n0} -> sin Puerta del Mar {n1} (-{n0-n1}) -> sin duplicados {n2} (-{n1-n2}) -> sin NA {n3} (-{n2-n3})")
        print(df.subtipo.value_counts().to_dict(), "DMG:", int(df.diabetes.sum()))
    return df

def cohorte_precoz(df):
    """Controles armonizados: se excluyen DG tardía y no clasificadas."""
    d = df[df.subtipo.isin(['control','precoz'])].copy()
    d['y'] = (d.subtipo=='precoz').astype(int)
    return d
