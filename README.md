# Do Global AI Stock Volatility Spillovers Improve BTC/IDR Volatility Forecasts?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-ee4c2c.svg)](https://pytorch.org/)
[![License: Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](#10-license-and-data-terms)
[![Status: Manuscript in Preparation](https://img.shields.io/badge/Status-Manuscript%20in%20Preparation-yellow.svg)](#9-how-to-cite)

Official open-source implementation and evaluation manifests for the empirical study:
> **"Do Global AI Stock Volatility Spillovers Improve BTC/IDR Volatility Forecasts?"**  
> *Ferrari Meilani and Supri Bin Amir* (2026).  
> Department of Mathematics, Universitas Hasanuddin, Makassar.  
> **Repository Contact:** GitHub `@ferastoro` · **Manuscript Status:** In preparation; not published, not peer reviewed.

---

## 🔬 Overview & Key Contributions

We evaluate whether cross-market volatility spillovers from five major US artificial intelligence equities and two global cryptocurrencies improve out-of-sample volatility forecasts for the Indonesian domestic Bitcoin market (BTC/IDR) from 2020 to 2026.

The framework is a **two-stage empirical design**:

$$\underbrace{\text{TVP-VAR}}_{\text{Stage 1: Connectedness}} \;\longrightarrow\; \underbrace{\text{TFT, N-BEATSx, HAR, GARCH, EGARCH}}_{\text{Stage 2: Out-of-Sample Forecasting}}$$

Stage 1 estimates the dynamic Total Spillover Index (TVP-VAR, $p=1$, $\lambda=0.99$, $\kappa=0.96$) and produces daily net directional connectedness features. Stage 2 feeds those features into deep learning models and econometric benchmarks across 335 rolling-origin cutoffs at horizons $h=1$ and $h=10$ days.

### Key Results
1. **System Connectedness:** The dynamic TSI averages **38.74%**, ranging from 21.89% to 88.96% across major market dislocations (COVID-19 onset, Terra-Luna collapse, ChatGPT release).
2. **Proxy Sensitivity:** BTC/IDR net directional connectedness flips sign with the volatility proxy — net transmitter under $\ln(r^2)$ (+5.45%) and $|r|$ (+3.29%), but net receiver under $r^2$ (−4.98%) and $\ln(\text{Parkinson})$ (−4.10%).
3. **Forecasting Hierarchy:** HAR has the lowest out-of-sample MSE at both horizons (2.235×10⁻⁶ at $h=1$, 2.440×10⁻⁶ at $h=10$). After Bonferroni correction ($m=12$) we find no evidence of a difference between HAR and any deep learning configuration.
4. **QLIKE Exception:** GARCH(1,1) and EGARCH(1,1) rank above HAR under QLIKE at $h=1$ (QLIKE 1.654 and 1.641 vs 1.702; adjusted $p$ = 0.025 and 0.022, $m=12$).
5. **Spillover Features:** We find no statistically detectable forecasting advantage from TVP-VAR spillover features. SPILL vs. NOSPILL adjusted $p \geq 0.339$ for v6 ensembles ($m=4$).

These results describe statistical association, not causal risk transmission.

---

## 📁 Repository Structure

```text
ai-crypto-spillover/
├── README.md
├── requirements.txt              # Pinned dependencies
├── pyrefly.toml
│
├── src/                          # Core data pipeline scripts
│   ├── collect_data.py           # Data extraction: Yahoo Finance & Indodax REST API
│   ├── preprocess_data.py        # Calendar alignment, returns, feature engineering
│   ├── v5_helpers.py             # GARCH and baseline metric helpers
│   └── verify_stock_splits.py    # Stock split verification
│
├── notebooks/                    # Main and supplementary pipelines
│   ├── 01_eda.ipynb              # Main pipeline: EDA and stationarity
│   ├── 02_tvp_var_spillover.ipynb# Main pipeline: Stage 1 TVP-VAR estimation
│   ├── 06_dl_v6_multiseed.ipynb  # Main pipeline: Stage 2 deep learning (5 seeds, Kaggle GPU)
│   ├── 07_summary_results_visuals.ipynb  # Main pipeline: tables and figures
│   ├── 08-dl-v7-audit-loss-refit.ipynb   # Supplementary: loss-matched retraining & 60-day refit
│   ├── 03_dl_v4_multivariate_baseline.ipynb  # Earlier iteration (provenance)
│   ├── 04_dl_v5_benchmark.ipynb              # Earlier iteration (provenance)
│   ├── 05_dl_v5_sensitivity.ipynb            # Earlier iteration (provenance)
│   └── archive/                              # Archived notebooks (provenance)
│
├── data/
│   ├── clean/                    # Processed time-series and TVP-VAR features (1.26 MB)
│   └── raw/                      # Raw OHLCV histories: Yahoo Finance & Indodax (1.93 MB)
│
├── inputs/
│   └── v5/                       # Fixed dataset snapshots and baseline predictions
│
├── results/
│   ├── v6/                       # Primary multi-seed results reported in the paper
│   ├── v7-tuned/                 # Supplementary: loss-matched and periodic refit results
│   ├── v5/, v4/                  # Earlier iterations (provenance)
│   ├── outlier_check*/           # Residual outlier analyses
│   ├── RQ1_Average_Spillover_Table_vol.csv     # Baseline 8×8 GFEVD table
│   └── RQ1_proxy_sensitivity_table.csv         # Net spillovers across 4 volatility proxies
│
├── audit/                        # Independent verification scripts (author's record)
│   └── v6/                       # V6-specific audit files and DM tables
│
├── audit_outputs/                # Audit verification tables and CSV logs
│   ├── v7ALL/followup/           # Unified DM, SPILL vs NOSPILL, and refit tests
│   ├── followup/                 # MCS, QLIKE, pairwise spillover tables
│   ├── regime/                   # Structural break and regime analyses
│   └── checkpoint3a/             # TVP-VAR proxy sensitivity replication outputs
│
└── references/                   # Reference list only; PDFs are not included
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/ferastoro/ai-crypto-spillover.git
cd ai-crypto-spillover
pip install -r requirements.txt
```

### 2. Smoke-Test the Data Pipeline
```bash
python3 src/collect_data.py          # ~1 minute; writes data/raw/*.csv
python3 src/preprocess_data.py       # ~15 seconds; writes data/clean/*.csv
```

### 3. Verify the Proxy Sensitivity Audit (No GPU Required)
```bash
python3 audit/run_checkpoint3a.py
```
Expected: re-estimates TVP-VAR connectedness across 4 volatility proxies and writes results to `audit_outputs/checkpoint3a/`. Max absolute difference vs `results/RQ1_proxy_sensitivity_table.csv`: **0.000000**. Runtime: ~1 minute 57 seconds on a local CPU.

---

## 📊 Research Questions

| RQ | Question |
| :--- | :--- |
| **RQ1** | Do dynamic volatility spillovers exist between US AI equities, global cryptocurrencies, and BTC/IDR — and are net directional signs invariant across volatility proxies? |
| **RQ2** | Do TFT and N-BEATSx statistically outperform HAR, GARCH, EGARCH, Roll-20, and Hist-Exp? |
| **RQ3** | Does augmenting models with TVP-VAR spillover features improve forecast accuracy vs. purely autoregressive histories? |

---

## 📊 Reproducing Paper Tables

All primary tables are read directly from pre-computed CSV files.

### Table I — Net Volatility Spillover & Proxy Sensitivity

| Asset | Baseline $\ln(r^2)$ | $\|r\|$ | $r^2$ | $\ln(\text{Parkinson})$ |
| :--- | :---: | :---: | :---: | :---: |
| NVDA | −2.32% | +16.03% | +7.52% | +6.69% |
| MSFT | −2.34% | −14.94% | −3.07% | −0.14% |
| TSLA | −4.76% | +26.29% | −2.08% | −5.25% |
| GOOGL | −3.42% | −12.82% | −4.80% | −0.72% |
| META | −4.01% | −14.77% | −8.72% | −1.63% |
| BTC-USD | +11.80% | +0.32% | +7.54% | +4.60% |
| ETH-USD | −0.40% | +4.87% | +0.33% | +0.54% |
| **BTC/IDR** | **+5.45%** | **−4.98%** | **+3.29%** | **−4.10%** |

Source: `results/RQ1_proxy_sensitivity_table.csv` · Generating script: `audit/run_checkpoint3a.py`

### Table II — Out-of-Sample Forecasting Performance

Source: `results/v6/v6_metrics_ensemble.csv` · Generating notebook: `notebooks/06_dl_v6_multiseed.ipynb`

| Model | MSE ×10⁻⁶ ($h=1$) | $R^2_\text{OOS}$ ($h=1$) | MSE ×10⁻⁶ ($h=10$) | $R^2_\text{OOS}$ ($h=10$) |
| :--- | :---: | :---: | :---: | :---: |
| **HAR** | **2.235** | **+16.91%** | **2.440** | **+9.17%** |
| GARCH(1,1) | 2.313 | +13.99% | 2.478 | +7.74% |
| EGARCH(1,1) | 2.257 | +16.07% | 2.496 | +7.06% |
| TFT SPILL | 2.388 | +11.20% | 2.447 | +8.90% |
| TFT NOSPILL | 2.395 | +10.96% | 2.454 | +8.65% |
| NBEATSx SPILL | 2.272 | +15.52% | — | — |
| NBEATSx NOSPILL | 2.505 | +6.85% | — | — |
| Hist-Exp | 2.689 | 0.00% | 2.686 | 0.00% |

$R^2_\text{OOS}$ is computed relative to Hist-Exp (historical expectation baseline).

### Table III — Diebold-Mariano Tests vs. HAR

Source: `audit/v6/audit_v6_dm_ensemble.csv`

After Bonferroni correction ($m=12$, all candidate-vs-HAR comparisons at two horizons), the smallest adjusted $p$ among deep learning models is 0.098. No deep learning configuration differs from HAR at any conventional significance level.

### QLIKE Evaluation

```bash
# Reproduce the QLIKE ranking and MCS
python3 audit/run_task5_qlike_evaluation.py
```

Source: `audit_outputs/task5_qlike_evaluation.csv` · MCS: `audit_outputs/followup/followup3_mcs_qlike_7models.csv`

---

## 💾 Dataset Preparation

The study uses eight financial instruments from January 3, 2020 to September 4, 2026:

| Ticker | Asset | Class | Source |
| :--- | :--- | :--- | :--- |
| `NVDA` | NVIDIA Corporation | US Equity (AI Hardware) | Yahoo Finance (`yfinance`) |
| `MSFT` | Microsoft Corporation | US Equity (AI Platform) | Yahoo Finance (`yfinance`) |
| `TSLA` | Tesla, Inc. | US Equity (Autonomous AI) | Yahoo Finance (`yfinance`) |
| `GOOGL` | Alphabet Inc. | US Equity (AI Research) | Yahoo Finance (`yfinance`) |
| `META` | Meta Platforms, Inc. | US Equity (Open AI / Social) | Yahoo Finance (`yfinance`) |
| `BTC-USD` | Bitcoin / US Dollar | Cryptocurrency | Yahoo Finance (`yfinance`) |
| `ETH-USD` | Ethereum / US Dollar | Cryptocurrency | Yahoo Finance (`yfinance`) |
| `BTC/IDR` | Bitcoin / Indonesian Rupiah | Domestic Crypto | Indodax REST API |

Both `data/raw/` (1.93 MB) and `data/clean/` (1.26 MB) are included in the repository. To re-download from source:

```bash
python3 src/collect_data.py        # Hard-coded END_DATE = "2026-09-05" (src/collect_data.py, line 50)
python3 src/preprocess_data.py
```

### Calendar Harmonization
- **Stage 1 (TVP-VAR):** US equity trading days only ($T=1{,}556$, yielding 1,555 log returns). Weekend and holiday crypto returns accumulate into the next equity session.
- **Stage 2 (Forecasting):** $T=1{,}677$ trading days (1,676 effective observations). Weekend returns are captured via `crypto_btc_usd_weekend_return` and indicator variables `gap_days`, `is_monday`, `is_ihsg_holiday`.

---

## 🚀 Reproducing the Full Pipeline

Execute steps in this order:

| Step | Command / Notebook | Environment | Runtime | Output |
| :--- | :--- | :--- | :--- | :--- |
| 1. Data collection | `python3 src/collect_data.py` | Local CPU | ~1 min | `data/raw/*.csv` |
| 2. Preprocessing | `python3 src/preprocess_data.py` | Local CPU | ~15 sec | `data/clean/*.csv` |
| 3. EDA | `notebooks/01_eda.ipynb` | Local CPU | ~2 min | Stationarity tests, correlations |
| 4. TVP-VAR (Stage 1) | `notebooks/02_tvp_var_spillover.ipynb` | Local CPU | ~12 min | `results/RQ1_Average_Spillover_Table_vol.csv` |
| 5. Deep Learning (Stage 2) | `notebooks/06_dl_v6_multiseed.ipynb` | Kaggle Tesla T4 | ~3.5 hr | `results/v6/v6_metrics_ensemble.csv` |
| 6. Summary & Figures | `notebooks/07_summary_results_visuals.ipynb` | Local CPU | ~1 min | Tables and figures |
| 7. Supplementary Refit (optional) | `notebooks/08-dl-v7-audit-loss-refit.ipynb` | Kaggle Tesla T4 | ~6.5 hr | `results/v7-tuned/run_log.csv` |
| 8. Proxy Sensitivity Audit | `python3 audit/run_checkpoint3a.py` | Local CPU | ~2 min | `audit_outputs/checkpoint3a/` |

### Kaggle Setup
Notebooks `06` and `08` reference Kaggle dataset mount paths (`/kaggle/input/spillover-v5-inputs/`). Upload the local folder `inputs/v5/` to Kaggle as a private dataset named `spillover-v5-inputs`.

### GPU Determinism Notice
Non-deterministic CUDA floating-point operations across GPU hardware architectures mean that re-running notebooks `06` and `08` on a different GPU will not produce bit-for-bit identical predictions. The predictions stored in `results/v6/` are the verified reference.

---

## 🗺️ Paper-to-File Map

| Paper Element | Description | Source File | Generating Notebook / Script |
| :--- | :--- | :--- | :--- |
| Table I | Net spillover × 4 proxies | `results/RQ1_proxy_sensitivity_table.csv` | `audit/run_checkpoint3a.py` |
| Table II | MSE, MAE, $R^2_\text{OOS}$ ($h=1,10$) | `results/v6/v6_metrics_ensemble.csv` | `notebooks/06_dl_v6_multiseed.ipynb` |
| Table III | DM tests vs HAR & Hist-Exp | `audit/v6/audit_v6_dm_ensemble.csv` | `notebooks/06_dl_v6_multiseed.ipynb` |
| Text: Pairwise connectedness | BTC-USD ↔ BTC/IDR shares (30.78% / 29.35%) | `audit_outputs/followup/followup1_pairwise_spillovers.csv` | `audit/run_task1_stage1_sample.py` |
| Text: Contemporaneous dominance | $p=0$ TSI (35.17%) vs $p=1$ (38.74%) | `audit_outputs/task2_p0_comparison.csv` | `audit/run_task2_lag0_shuffle.py` |
| Text: Circular shuffle test | 200-iter shuffle mean TSI (16.83%, $p<0.005$) | `audit_outputs/task2_shuffle_distribution.csv` | `audit/run_task2_lag0_shuffle.py` |
| Text: QLIKE & MCS | EGARCH and GARCH rank above HAR at $h=1$ | `audit_outputs/followup/followup3_mcs_qlike_7models.csv` | `audit/run_task5_qlike_evaluation.py` |
| Text: TVP-VAR grid sensitivity | 9-combination $\lambda \times \kappa$ grid | `audit_outputs/task8_tvp_var_sensitivity.csv` | `audit/run_task8_tvp_var_sensitivity.py` |
| Text: Loss & refit re-check | Loss-matched retraining and 60-day refit | `audit_outputs/v7ALL/followup/followup4_dm_vs_har_m12_vs_m16.csv` | `notebooks/08-dl-v7-audit-loss-refit.ipynb` |
| Text: Regulatory transition | Chow and Sup-Wald structural break tests | `audit_outputs/regime/structural_break_tests.csv` | `audit/run_task10_regimes.py` |

---

## ⚙️ Environment & Requirements

- **Python:** 3.10+ (reference Kaggle environment: Python 3.13.15)
- **Key pinned packages:** `neuralforecast==3.3.0`, `torch==2.14.0`, `pytorch-lightning==2.6.6`, `arch==8.0.0`, `statsmodels==0.15.0`, `pandas==2.3.3`, `scipy==1.18.1`, `scikit-learn==1.9.1`
- **Fixed seeds:** `[42, 7, 13, 21, 99]`
- **Sample splits:** 1,165 training / 167 validation / 344 test observations; 335 rolling-origin cutoffs

Full dependency list: `requirements.txt`

---

## ⚠️ Limitations

- **Single target and split:** Analysis covers one domestic trading pair (BTC/IDR) and one chronological split. Multiple pairs and rolling walk-forward splits are left for future work.
- **Proxy sensitivity:** The sign of BTC/IDR net connectedness depends on the volatility proxy; it reverses between compressed ($\ln(r^2)$) and raw squared-return proxies.
- **Lag order:** The TVP-VAR uses $p=1$; AIC and BIC select $p=0$, indicating predominantly contemporaneous connectedness.
- **Asset system scope:** Broad market indices (S&P 500, NASDAQ 100) are excluded; market-wide equity risk is not separately controlled.
- **Regulatory transition & unconstrained forecasts:** The sample spans Indonesia's crypto regulatory transition (CoFTRA/Bappebti → OJK under Law No. 4/2023); N-BEATSx produces unconstrained predictions that require zero-clipping post-processing.

---

## 📜 How to Cite

If you use this codebase or find our findings useful, please cite the manuscript in preparation:

```bibtex
@misc{meilani2026global,
  author = {Meilani, Ferrari and Bin Amir, Supri},
  title  = {Do Global AI Stock Volatility Spillovers Improve {BTC/IDR} Volatility Forecasts?},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/ferastoro/ai-crypto-spillover}
}
```

This entry will be updated if the paper is published.

---

## 📄 License and Data Terms

No license file has been added yet; all rights are reserved by default.

This repository and its contents are for academic research purposes only and do not constitute financial, investment, or trading advice. Data originates from Yahoo Finance and the Indodax public REST API; check their terms before redistributing.

Contact: GitHub issues or `@ferastoro` only.
