
# Simulation study

One can generate synthetic data for simulation study.

## Simulation Data Generation

Use `data_generate.py` to generate `data.csv`, `data_seed`, and `data_test`.

`data.csv` suffers from positivity issue, while `data_seed` and `data_test` are IID RCT.


## Sythetic Data

We use CTGAN and GReaT to generate synthetic data based on `data_seed`. Place them under `./gan_data` and `./llm_data.csv` as `syn_full.csv`. 


## Hybrid Generation

Run `syn_hybrid.py` to generate A|W and Y|A, W by random forest. The `syn_hybrid.csv` will be seen under ./gan_data and ./llm_data. 


## Positivity
Use `pair.py` to match the sample in `data.csv` and `syn_hybrid.csv` and obtain `pair.csv` that is the final dataset we do inference. Use `iptw.py` and `aipw.py` to see synthetic data helps.


## Other metrics
Use `dcr.py`, `tstr.py` to see the synthetic data quality.


## Synthetic helps with simulaation
Use `simulate_gen.py` to see synthetic data helps to benchmark different estimator.