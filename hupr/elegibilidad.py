"""Eligibility misclassification check: pattern of early testing and early diagnosis with eligibility defined without
ethnicity (as recorded in the multicentre cohort) or by BMI and family history only; availability of first-trimester glucose."""
import json, warnings; warnings.filterwarnings('ignore')
from sensibilidad_hupr import *
d = preparar(20); R = {}
R['etnia_riesgo_n'] = int(d.etnia_riesgo.sum())
R['eleg_solo_por_etnia'] = int(((d.PAI == 1) & (d.obesa == 0) & (d.ant_fam_diabetes == 0) & (d.DG_previa_lab == 0) & (d.etnia_riesgo == 1)).sum())
g = d[d.early + d.late > 0]
for nombre, e in [('sin_etnia', (d.obesa == 1) | (d.ant_fam_diabetes == 1) | (d.DG_previa_lab == 1)), ('solo_IMC_AF', (d.obesa == 1) | (d.ant_fam_diabetes == 1))]:
    ee = e[g.index]
    R[nombre] = dict(n_elig=int(e.sum()), tested_elig=round(100 * d[e].test_precoz.mean(), 1), tested_nonelig=round(100 * d[~e].test_precoz.mean(), 1),
                     pct_early_elig=round(100 * g[ee].early.mean(), 1), pct_early_nonelig=round(100 * g[~ee].early.mean(), 1),
                     auc_flag=round(A(d[d.late == 0].early, e[d.late == 0].astype(int)), 3))
R['glu_disp_testadas'] = round(100 * (1 - d[d.test_precoz == 1].glu_mis.mean()), 1); R['glu_disp_no_testadas'] = round(100 * (1 - d[d.test_precoz == 0].glu_mis.mean()), 1)
json.dump(R, open('elegibilidad_misclas.json', 'w'), indent=1); print(json.dumps(R, indent=1))
