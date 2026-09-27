# Selective first-trimester testing and early-onset gestational diabetes — analysis and simulation code

Code for the article *"Selective first-trimester testing inflates the apparent predictability of early-onset
gestational diabetes: two cohort studies and a simulation"* (Fernández Alba JJ et al., submitted to *Diabetologia*).

The study shows that when first-trimester testing for gestational diabetes (GDM) is offered only to women with risk
factors, early-onset GDM can only be diagnosed in those women. This verification bias inflates the apparent
discrimination of prediction models for early-onset GDM and the phenotypic contrast between early-onset and
late-onset GDM. The code evaluates fixed clinical models in two cohorts without correction, restricted to women
eligible for or tested early, and corrected by multiple imputation and inverse probability weighting (IPW), and runs
a simulation study with known truth (ADEMP structure).

**No individual-level data are included.** The cohort data cannot be shared because of data protection regulations.
The simulation is fully reproducible from the code alone.

## Contents

| Path | Purpose |
|---|---|
| `cohorte.py` | Multicentre cohort (CALDIAGEST): data loading, exclusions and fixed linear predictors of the clinical models |
| `caldiagest_sesgo.py` | Multicentre cohort: uncorrected and eligibility-restricted AUCs, eligibility AUC, proportion diagnosed early, phenotype differences, site and timing sensitivity analyses (bootstrap, 500 resamples) |
| `tablas_v2.py` | Tables 1 and 2 (both cohorts) |
| `hupr/cohorte_hupr.py` | Single-centre cohort: early testing (GCT or OGTT < 20 weeks), GDM and timing, eligibility |
| `hupr/etiquetas.py` | GDM definition from the four OGTT values (NDDG or Carpenter–Coustan thresholds; fasting value of the OGTT itself or, in sensitivity analyses, from other blood tests) |
| `hupr/sensibilidad_hupr.py` | Analysis set and estimators: uncorrected, restricted, IPW, multiple imputation with persistence and δ-shift |
| `hupr/final_hupr.py`, `hupr/combinar_ic.py` | Primary single-centre analysis (point estimates with 50 imputations; bootstrap in two chunks of 500) and CIs |
| `hupr/boot_sens.py` | Sensitivity analyses (200 bootstrap resamples each) |
| `hupr/punto_inflexion.py` | Tipping-point analysis for δ |
| `hupr/diagnosticos_ipw.py` | IPW weight diagnostics and implied undiagnosed early-onset cases (ESM Table 4) |
| `hupr/elegibilidad.py` | Eligibility misclassification check and availability of first-trimester glucose |
| `hupr/fig2_datos.py` | Aggregated counts for Fig. 2 and AUC of the verification model |
| `simulacion/` | Simulation: data-generating mechanism and estimators (`sim_verificacion.py`), main runs (`correr_todo.py`), model-learning and protocol-flag experiments (`extra.py`), persistence runs (`persistencia.py`), summary and Fig. 4 (`resumir.py`) |
| `figures/figs_v2.py` | Figs 2 and 3 from aggregated results |

Code comments are partly in Spanish.

## Input data (not distributed)

* **Multicentre cohort**: CALDIAGEST REDCap export in SPSS format; set its path with the environment variable
  `CALDIAGEST_SAV`.
* **Single-centre cohort**: three pseudonymised tables derived from the laboratory information system and the
  delivery registry of Hospital Universitario Puerto Real, stored as pandas pickles in `hupr/`:
  * `episodios.pkl` — one row per pregnancy episode (episode and woman identifiers, linked delivery flag, clinical
    dating, delivery date, age, BMI, parity, family history of diabetes, ethnic group, first-trimester fasting glucose);
  * `peticiones.pkl` — laboratory requests assigned to episodes (request number, episode, test type, date);
  * `curvas.pkl` — 100 g OGTT values (60, 120 and 180 min, and fasting glucose from other blood tests) by request;
  * `curvas_basal_real.pkl` — fasting (0 min) value of each OGTT, extracted separately from the laboratory information
    system and linked to `curvas.pkl` by request number, patient identifier and request date (request numbers are
    reused by the laboratory, so request number alone is not a valid key).

  Grouping of laboratory requests into pregnancy episodes and record linkage with the delivery registry used
  identifiable data, as did the linkage of the fasting OGTT values; the procedure is described in the article and its
  electronic supplementary material, and the code for these steps is not distributed.

## How to run

```bash
pip install -r requirements.txt

# Multicentre cohort
export CALDIAGEST_SAV=/path/to/caldiagest.sav
python caldiagest_sesgo.py 500
python tablas_v2.py

# Single-centre cohort (run inside hupr/)
cd hupr
python final_hupr.py 0 0          # point estimates
for s in 101 102 103 104 201 202 203 204; do
  python final_hupr.py $s 125     # bootstrap in 8 chunks of 125 (1000 resamples)
done
python combinar_ic.py
for s in base p90 p75 p50 delta_m1 delta_p1 flags trim99 corte24 imc_imp analitica1T sin2020 cc orig nddg_prest; do
  python boot_sens.py $s 200
done
python punto_inflexion.py
python diagnosticos_ipw.py
python elegibilidad.py
python fig2_datos.py
cd ..

# Simulation (no data needed)
cd simulacion
for i in 0 1 2 3 4 5 6 7 8 9; do python correr_todo.py $i; done
python extra.py S1; python extra.py S2
python persistencia.py
python resumir.py
```

All random seeds are fixed. Bootstrap resamples, imputations and simulated cohorts are reproducible with the
package versions in `requirements.txt` (Python 3.10).

## Licence and citation

MIT licence. If you use this code, please cite the article and this repository (see `CITATION.cff`).
