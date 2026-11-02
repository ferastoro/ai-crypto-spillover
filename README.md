# Do Global AI Stock Volatility Spillovers Improve BTC/IDR Volatility Forecasts?

**Authors:**  
Ferrari Meilani and Supri Bin Amir  
Department of Mathematics, Universitas Hasanuddin, Makassar  

**Repository Contact:** GitHub `@ferastoro`  
**Manuscript Status:** Manuscript in preparation; not published, not peer reviewed.

In this study, we evaluate whether cross-market volatility spillovers from five major US artificial intelligence equities and two global cryptocurrencies improve out-of-sample volatility forecasts for the Indonesian domestic Bitcoin market (BTC/IDR) from 2020 to 2026. Using a two-stage empirical design combining time-varying parameter vector autoregression (TVP-VAR) with deep learning architectures (Temporal Fusion Transformer and N-BEATSx), we identify dynamic connectedness across markets, though domestic net directional connectedness depends on the volatility proxy. Out of sample, econometric benchmarks match or outperform deep models, and exogenous spillover features provide no statistically detectable forecasting advantage.

---

## 2. Research Questions and Empirical Design

### Research Questions
We address three core empirical research questions:
- **RQ1 (Connectedness & Proxy Stability):** Do dynamic volatility spillovers exist between US AI equities, global cryptocurrencies, and the Indonesian domestic Bitcoin market (BTC/IDR), and are net directional spillover signs invariant across alternative volatility proxies?
- **RQ2 (Model Hierarchy):** Do deep learning architectures (Temporal Fusion Transformer, N-BEATSx) statistically outperform established econometric benchmarks (HAR, GARCH, EGARCH, Historical Expectation, Rolling 20-day mean) in out-of-sample volatility forecasting?
- **RQ3 (Feature Value):** Does augmenting forecasting models with dynamic TVP-VAR volatility spillover features improve forecast accuracy relative to purely autoregressive target volatility histories?

### Empirical Design
The framework consists of two sequential stages. Stage 1 estimates dynamic volatility connectedness across eight assets using a TVP-VAR model with lag order $p=1$ and forgetting factors $\lambda=0.99$ and $\kappa=0.96$. Stage 2 implements rolling-origin out-of-sample forecasting of BTC/IDR squared returns across 335 daily cutoffs using linear HAR, GARCH(1,1), EGARCH(1,1), Roll-20, Hist-Exp, TFT, and N-BEATSx models evaluated with and without TVP-VAR spillover features.

---

## 3. Main Findings
We summarize our empirical findings in five key results:
- **System Connectedness:** The dynamic Total Spillover Index (TSI) averages 38.74%, fluctuating between 21.89% and 88.96% across major market dislocations such as the COVID-19 onset, the Terra-Luna collapse, and the ChatGPT release (Source: `data/clean/tvp_var_dynamic_features.csv`, column `tvp_tsi`; `results/RQ1_Average_Spillover_Table_vol.csv`, row 8, column 8).
- **Proxy Sensitivity of Net Spillovers:** Net directional connectedness depends on the chosen volatility proxy. For BTC/IDR, it acts as a net transmitter under compressed proxies (+5.45% under baseline $\ln(r^2)$ and +3.29% under $|r|$), but reverses to a net receiver under uncompressed squared returns (-4.98%) and extreme range (-4.10% under $\ln(\text{Parkinson})$) (Source: `results/RQ1_Average_Spillover_Table_vol.csv`, row `CRYPTO_BTC_IDR_INDODAX`; `results/RQ1_proxy_sensitivity_table.csv`, row `CRYPTO_BTC_IDR_INDODAX`).
- **Contemporaneous vs. Dynamic Interaction:** Reciprocal pairwise connectedness between BTC-USD and BTC/IDR is nearly symmetric: BTC-USD accounts for 30.78% of BTC/IDR forecast error variance, while BTC/IDR accounts for 29.35% of BTC-USD variance (Source: `audit_outputs/followup/followup1_pairwise_spillovers.csv`, rows 0 and 1). Connectedness is predominantly contemporaneous: a static $p=0$ TVP-VAR yields a mean TSI of 35.17% compared to 38.74% for $p=1$ (Source: `audit_outputs/task2_p0_comparison.csv`, rows 0 and 1), and a 200-iteration circular shuffle produces a mean TSI of only 16.83% ($p < 0.005$; Source: `audit_outputs/task2_shuffle_distribution.csv`, row 0).
- **Forecasting.** HAR has the lowest MSE (2.235e-06 at h=1, 2.440e-06 at h=10). After Bonferroni correction we find no evidence of a difference between HAR and any deep learning configuration: the smallest adjusted p is 0.098 (m=12; v6 and v7 Experiment A models against HAR) and it is 1.0 for the four v6 ensembles alone. Under QLIKE, GARCH and EGARCH are ahead of HAR at h=1 (QLIKE 1.654 and 1.641 vs 1.702; adjusted p 0.025 and 0.022, m=12 candidate-vs-HAR comparisons across two horizons). Sources: `audit_outputs/v7ALL/followup/followup1_baseline_reproduction_sse_mse.csv`; `audit_outputs/v7ALL/followup/followup4_dm_vs_har_m12_vs_m16.csv`; `audit_outputs/task5_qlike_evaluation.csv`; `audit_outputs/followup/followup3_mcs_qlike_7models.csv`.
- **Spillover features.** We find no evidence that they help. For the four v6 ensembles the adjusted p of SPILL vs NOSPILL is at least 0.339 (m=4); across all configurations in the audit, on the full sample, it is at least 0.066 (m=4 per model family). Sources: `audit/v6/audit_v6_dm_ensemble.csv`; `audit_outputs/v7ALL/followup/followup3_spill_vs_nospill_bonferroni.csv`.

These empirical results describe statistical connectedness and predictive association, not causal risk transmission.

---

## 4. Data

### Asset System
The dataset spans eight financial instruments from January 3, 2020 to September 4, 2026:

| Ticker / Symbol | Asset Name | Asset Class | Primary Market | Data Source |
| :--- | :--- | :--- | :--- | :--- |
| `NVDA` | NVIDIA Corporation | US Equity (AI Hardware) | Nasdaq | Yahoo Finance (`yfinance`) |
| `MSFT` | Microsoft Corporation | US Equity (AI Platform) | Nasdaq | Yahoo Finance (`yfinance`) |
| `TSLA` | Tesla, Inc. | US Equity (Autonomous AI) | Nasdaq | Yahoo Finance (`yfinance`) |
| `GOOGL` | Alphabet Inc. | US Equity (AI Research) | Nasdaq | Yahoo Finance (`yfinance`) |
| `META` | Meta Platforms, Inc. | US Equity (Open AI / Social) | Nasdaq | Yahoo Finance (`yfinance`) |
| `BTC-USD` | Bitcoin / US Dollar | Global Cryptocurrency | Global Exchanges | Yahoo Finance (`yfinance`) |
| `ETH-USD` | Ethereum / US Dollar | Global Cryptocurrency | Global Exchanges | Yahoo Finance (`yfinance`) |
| `BTC/IDR` | Bitcoin / Indonesian Rupiah | Domestic Cryptocurrency | Indodax | Indodax Public REST API |

### Calendar Harmonization Protocols
Because equity markets trade five days a week while cryptocurrency exchanges operate continuously 24/7, we implement two distinct protocols:
1. **Stage 1 Protocol (Connectedness Estimation):** Retains US equity trading days ($T = 1,556$ calendar days, yielding 1,555 effective log returns). Cryptocurrency returns over weekends and holidays are accumulated into the subsequent equity trading session.
2. **Stage 2 Protocol (Out-of-Sample Forecasting):** Models $T = 1,677$ trading days (1,676 effective observations: 1,165 training, 167 validation, 344 test across 335 daily cutoffs). Calendar gaps are captured via an explicit weekend-return feature (`crypto_btc_usd_weekend_return`) and indicator variables (`gap_days`, `is_monday`, `is_ihsg_holiday`).

### Data Availability & Re-downloading
The data collection end date is hard-coded as `END_DATE = "2026-09-05"` in `src/collect_data.py` on line 50. Both raw and processed data directories are included in the repository (`data/raw`, 1.93 MB; `data/clean`, 1.26 MB). Data comes from third-party sources; check their terms before redistributing. src/collect_data.py re-downloads it.

---

## 5. Repository Layout
```
├── audit/                  # Independent audit scripts (author's record; some paths point to local scratch folders)
├── audit_outputs/          # Audit verification tables, CSV logs, and traces
├── data/
│   ├── clean/              # Processed time-series and TVP-VAR dynamic features (1.26 MB)
│   └── raw/                # Raw OHLCV price histories from Yahoo Finance & Indodax (1.93 MB)
├── inputs/
│   └── v5/                 # Fixed dataset files and baseline predictions for model inputs
├── notebooks/
│   ├── 01_eda.ipynb                         # Main pipeline: data exploration and stationarity
│   ├── 02_tvp_var_spillover.ipynb           # Main pipeline: Stage 1 TVP-VAR estimation
│   ├── 06_dl_v6_multiseed.ipynb             # Main pipeline: Stage 2 deep learning forecasting
│   ├── 07_summary_results_visuals.ipynb     # Main pipeline: visualization and summary tables
│   ├── 08-dl-v7-audit-loss-refit.ipynb      # Supplementary: loss tuning and 60-day refit audit
│   ├── 03_dl_v4_multivariate_baseline.ipynb # Earlier iteration: initial deep learning setup
│   ├── 04_dl_v5_benchmark.ipynb            # Earlier iteration: preliminary metric evaluation
│   ├── 05_dl_v5_sensitivity.ipynb          # Earlier iteration: initial sensitivity runs
│   └── archive/                             # Archive notebooks kept for provenance
├── references/             # Reference list only; PDFs are not included
├── results/
│   ├── v6/                 # Primary experimental results and predictions used in the paper
│   ├── v7-tuned/           # Supplementary loss-matched and periodic refit results
│   ├── v5/                 # Earlier model iterations kept for provenance
│   ├── v4/                 # Earlier model iterations kept for provenance
│   ├── outlier_check/      # Residual outlier analysis files
│   ├── RQ1_Average_Spillover_Table_vol.csv  # Baseline 8x8 GFEVD table
│   └── RQ1_proxy_sensitivity_table.csv     # 8-asset net spillovers across 4 volatility proxies
└── src/
    ├── collect_data.py     # Data extraction from Yahoo Finance and Indodax
    ├── preprocess_data.py  # Time alignment, returns calculation, and feature engineering
    ├── v5_helpers.py       # Helper functions for GARCH and baseline metric estimation
    └── verify_stock_splits.py # Stock split verification checks
```

### Pipeline Organization & File Roles
- **Main Pipeline Notebooks:** Notebooks `notebooks/01_eda.ipynb`, `notebooks/02_tvp_var_spillover.ipynb`, `notebooks/06_dl_v6_multiseed.ipynb`, and `notebooks/07_summary_results_visuals.ipynb` form the core empirical pipeline.
- **Supplementary Experiments:** Notebook `notebooks/08-dl-v7-audit-loss-refit.ipynb` evaluates loss-matched retraining and periodic expanding-window refitting.
- **Historical Iterations:** Notebooks `03`, `04`, `05`, and the `notebooks/archive/` folder are earlier iterations kept for provenance.
- **Results Structure:** Folder `results/v6/` contains the primary multi-seed predictions and metrics reported in the paper; `results/v7-tuned/` contains supplementary experiments; earlier `results/` folders document model evolution.
- **Audit Verification:** The `audit/` and `audit_outputs/` directories hold independent verification scripts written after the main analysis. These scripts are the author's record and some paths point to local scratch folders. Four key audit summaries are:
  - `audit_outputs/v7ALL/followup/followup4_dm_vs_har_m12_vs_m16.csv`: Unified DM tests vs. HAR under $m=12$ and $m=16$ families across all candidates.
  - `audit_outputs/v7ALL/followup/followup3_spill_vs_nospill_bonferroni.csv`: Full ablation test comparing SPILL vs. NOSPILL across all architectures.
  - `audit_outputs/followup/followup3_mcs_qlike_7models.csv`: Model Confidence Set (MCS) inclusion and QLIKE loss metrics.
  - `audit_outputs/task2_p0_comparison.csv`: Instantaneous ($p=0$) vs. dynamic ($p=1$) connectedness decomposition.

---

## 6. Paper-to-File Map
The table below maps all tables and cited empirical quantities in the paper to their generating files:

| Paper Element | Description | Result File | Generating Script / Notebook |
| :--- | :--- | :--- | :--- |
| Table I (`tab:rq1_connectedness`) | 8-asset average net spillover across 4 volatility proxies | `results/RQ1_Average_Spillover_Table_vol.csv`, `results/RQ1_proxy_sensitivity_table.csv` | `notebooks/02_tvp_var_spillover.ipynb`, `audit/run_checkpoint3a.py` |
| Table II (`tab:forecast_performance`) | Out-of-sample MSE, MAE, and $R^2_{\text{OOS}}$ ($h=1, 10$) | `results/v6/v6_metrics_ensemble.csv` | `notebooks/06_dl_v6_multiseed.ipynb` |
| Table III (`tab:dm_tests`) | Diebold-Mariano tests vs. HAR and Hist-Exp ($m=8$) | `audit/v6/audit_v6_dm_ensemble.csv` | `notebooks/06_dl_v6_multiseed.ipynb` |
| Text: Pairwise Connectedness | Reciprocal BTC-USD and BTC/IDR shares (30.78% / 29.35%) | `audit_outputs/followup/followup1_pairwise_spillovers.csv` | `audit/run_task1_stage1_sample.py` |
| Text: Contemporaneous Dominance | Static $p=0$ TSI (35.17%) vs. $p=1$ (38.74%) | `audit_outputs/task2_p0_comparison.csv` | `audit/run_task2_lag0_shuffle.py` |
| Text: Circular Shuffle Test | 200-iteration shuffle mean TSI (16.83%, $p < 0.005$) | `audit_outputs/task2_shuffle_distribution.csv` | `audit/run_task2_lag0_shuffle.py` |
| Text: QLIKE & MCS Evaluation | QLIKE at $h=1$ ranking EGARCH and GARCH above HAR | `audit_outputs/followup/followup3_mcs_qlike_7models.csv` | `audit/run_task5_qlike_evaluation.py` |
| Text: TVP-VAR Grid Sensitivity | 9-combination $\lambda \times \kappa$ sensitivity grid | `audit_outputs/task8_tvp_var_sensitivity.csv` | `audit/run_task8_tvp_var_sensitivity.py` |
| Text: Loss & Refit Re-check | Loss-matched retraining and 60-day expanding refitting | `audit_outputs/v7ALL/followup/followup4_dm_vs_har_m12_vs_m16.csv` | `notebooks/08-dl-v7-audit-loss-refit.ipynb` |
| Text: Regulatory Transition | Chow and Sup-Wald tests for structural stability | `audit_outputs/regime/structural_break_tests.csv` | `audit/run_task10_regimes.py` |

---

## 7. How to Reproduce

### Ordered Pipeline Execution
To reproduce the empirical results, execute the pipeline in the following sequence:

1. **Data Collection:**  
   - Command: `python3 src/collect_data.py`  
   - Environment: Local CPU (runtime: ~1 minute).  
   - Output: Raw OHLCV price series in `data/raw/*.csv`.

2. **Data Preprocessing & Calendar Alignment:**  
   - Command: `python3 src/preprocess_data.py`  
   - Environment: Local CPU (runtime: ~15 seconds).  
   - Output: Aligned and engineered dataset in `data/clean/dataset_deep_learning.csv`.

3. **Exploratory Data Analysis:**  
   - Execution: Run `notebooks/01_eda.ipynb` sequentially.  
   - Environment: Local CPU (runtime: ~2 minutes).  
   - Output: Stationarity tests, correlation structures, and descriptive statistics.

4. **Stage 1 TVP-VAR Connectedness Estimation:**  
   - Execution: Run `notebooks/02_tvp_var_spillover.ipynb` sequentially.  
   - Environment: Local CPU (runtime: ~12 minutes).  
   - Output: `results/RQ1_Average_Spillover_Table_vol.csv` and dynamic spillover series.

5. **Stage 2 Deep Learning Multi-Seed Forecasting:**  
   - Execution: Run `notebooks/06_dl_v6_multiseed.ipynb` on Kaggle.  
   - Environment: Kaggle Tesla T4 GPU (runtime: ~3.5 hours).  
   - Output: `results/v6/v6_metrics_ensemble.csv` and out-of-sample predictions.

6. **Results Summary and Visualization:**  
   - Execution: Run `notebooks/07_summary_results_visuals.ipynb` sequentially.  
   - Environment: Local CPU (runtime: ~1 minute).  
   - Output: Formatted summary tables and high-resolution visual figures.

7. **Supplementary Loss-Matched and Periodic Refit Audit (Optional):**  
   - Execution: Run `notebooks/08-dl-v7-audit-loss-refit.ipynb` on Kaggle.  
   - Environment: Kaggle Tesla T4 GPU (runtime: ~6.5 hours).  
   - Output: `results/v7-tuned/run_log.csv` and expanding-refit predictions.

8. **Proxy Sensitivity & Provenance Audit:**  
   - Command: `python3 audit/run_checkpoint3a.py`  
   - Environment: Local CPU (runtime: ~1 minute 57 seconds).  
   - Output: Re-estimates TVP-VAR connectedness across 4 volatility proxies into `audit_outputs/checkpoint3a/` (`Average_Spillover_Table_vol.csv`, `tvp_var_dynamic_features_rerun.csv`, `checkpoint3a_A.txt`, `checkpoint3a_B.txt`, and `checkpoint3a_C.txt`), replicating `results/RQ1_proxy_sensitivity_table.csv` exactly (max absolute difference: 0.000000).

### Environment and Hardware Requirements
- **Python Version:** Python 3.10+ is supported (the reference Kaggle environment executed Python 3.13.15).
- **Core Dependencies:** Specified in `requirements.txt`. The deep learning pipeline pins `neuralforecast==3.3.0`, `torch==2.14.0`, `pytorch-lightning==2.6.6`, `arch==8.0.0`, `statsmodels==0.15.0`, `pandas==2.3.3`, `scipy==1.18.1`, and `scikit-learn==1.9.1`. Plotting packages (`matplotlib`, `seaborn`) and utility libraries (`yfinance`, `jupyter`) are unpinned.
- **Fixed Seeds:** Multi-seed deep learning models are evaluated across five fixed random seeds: `[42, 7, 13, 21, 99]`.
- **Sample Splits:** Fixed chronological splits of 1,165 training observations, 167 validation observations, and 344 test observations across 335 daily rolling-origin cutoffs.
- **Kaggle Setup Notes:** Notebooks `notebooks/06_dl_v6_multiseed.ipynb` and `notebooks/08-dl-v7-audit-loss-refit.ipynb` reference Kaggle dataset mount paths (`/kaggle/input/spillover-v5-inputs/`). An outside user reproducing these steps should upload the local folder `inputs/v5/` to Kaggle as a private dataset named `spillover-v5-inputs`.
- **GPU Determinism Notice:** Due to non-deterministic CUDA floating-point atomic operations across different GPU hardware architectures, re-running deep learning models on a new GPU will not produce bit-for-bit identical predictions. The predictions stored in `results/v6/` serve as the verified reference benchmark.

---

## 8. Limitations
We note five methodological and data limitations of this study:
- **Single Target Asset and Split:** Analysis is limited to a single domestic trading pair (BTC/IDR) and a single fixed chronological split; evaluation across multiple emerging exchange pairs and rolling walk-forward splits is left for future research.
- **Proxy Sensitivity:** The sign of net directional connectedness depends on the volatility proxy; BTC/IDR is a net transmitter under log-squared returns but reverses to a net receiver under raw squared returns.
- **Lag Order Selection:** The TVP-VAR model employs lag order $p=1$ for dynamic propagation, although static information criteria (AIC and BIC) select $p=0$, indicating that connectedness is predominantly contemporaneous.
- **Omission of Broad Market Benchmarks:** The asset system omits broad equity market indices (e.g., S&P 500 or NASDAQ 100), meaning market-wide equity risk is not separately controlled for.
- **Regulatory Transition and Unconstrained Forecasts:** The sample spans an Indonesian crypto regulatory transition (from CoFTRA / Bappebti to OJK under Law No. 4/2023), and N-BEATSx produces unconstrained negative variance predictions requiring zero-clipping post-processing.

---

## 9. How to Cite
If you use this codebase or find our findings useful, please cite the manuscript in preparation:
```bibtex
@misc{meilani2026global,
  author = {Meilani, Ferrari and Bin Amir, Supri},
  title = {Do Global AI Stock Volatility Spillovers Improve BTC/IDR Volatility Forecasts?},
  year = {2026},
  note = {Manuscript in preparation},
  url = {<repo-url>}
}
```
This entry will be updated if the paper is published.

---

## 10. License and Data Terms
No license file has been added yet; all rights are reserved by default.

This repository and its contents are for academic research purposes only and do not constitute financial, investment, or trading advice.

Contact: GitHub issues or `@ferastoro` only.
