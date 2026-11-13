# Fuzzy clustering and membership-gated autoencoders for GHG inventory compositions

Code for the manuscript "Fuzzy Clustering and Membership-Gated Autoencoders for Sectoral
Compositions of European Greenhouse Gas Inventories: A Country-Grouped Evaluation".

## Data
`env_air_gge_2023_parts.csv` (supplied separately) is derived from Eurostat table `env_air_gge`
(dissemination API, filter `time=2023`, `unit=THS_T`, table update 2 June 2026). One row per
country x gas; columns are the 20 non-overlapping CRF parts listed in Table 1 of the paper,
the official total `TOTX4_MEMO`, and the number of missing cells.

## Running
```
pip install -r requirements.txt
python run_experiments.py   env_air_gge_2023_parts.csv out   # main results (Tables 2-3, 6)
python run_sensitivity_L.py env_air_gge_2023_parts.csv out   # bottleneck width L = 2, 8
python run_robustness.py     env_air_gge_2023_parts.csv out   # NMI/purity, C stability, fuzzifier (Tables 4-5)
python run_loco_bootstrap.py env_air_gge_2023_parts.csv out  # leave-one-country-out + bootstrap (Table 7)
python run_ablation.py     env_air_gge_2023_parts.csv out   # constant-gate ablation, Wilcoxon, within-gas FCM
python make_figures.py      env_air_gge_2023_parts.csv out figs
```
Scripts must be run in this order (later scripts read earlier outputs). All random seeds are fixed;
total run time is roughly 30 minutes on two CPU cores.

## Files
| File | Purpose |
|---|---|
| `ghg_core.py` | data loading, zero replacement, ILR, FCM and validity indices, NumPy autoencoders |
| `run_experiments.py` | data audit, clustering, 5x5 country-grouped cross-validation |
| `run_sensitivity_L.py` | bottleneck-width sensitivity |
| `run_robustness.py` | external cluster metrics, choice of C, fuzzifier sensitivity, CIs |
| `run_loco_bootstrap.py` | leave-one-country-out validation with country bootstrap |
| `run_ablation.py` | constant-gate ablation, non-parametric tests, Ridge/PCA check, within-gas FCM |
| `make_figures.py` | all figures |

## Licence
MIT
