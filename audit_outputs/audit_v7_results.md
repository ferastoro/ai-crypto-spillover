# Audit Report: Kaggle v7 Experimental Results

**Audit Target**: `results/v7/`  
**Reference Scripts & Notebooks**:
- `audit/run_audit_v7.py` (Audit execution script)
- `audit_outputs/followup/kaggle_audit_v7.ipynb` (Audit specification)
- `notebooks/08_dl_v7_audit_loss_refit.ipynb` (Target audit notebook)
- `notebooks/06_dl_v6_multiseed.ipynb` (v6 reference notebook)

**Audit Execution Date**: 2026-10-30  
**Audit Tables Generated**: `audit_outputs/v7/*.csv`

---

## 1. Integrity Audit

### 1.1 Manifest MD5 Verification
- **File**: `results/v7/manifest.csv`
- **Script**: `audit/run_audit_v7.py`
- **Output Table**: [`task1_manifest_verification.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task1_manifest_verification.csv)

| File | Expected MD5 (Manifest) | Actual MD5 (On Disk) | Status |
| :--- | :--- | :--- | :--- |
| `__notebook__.ipynb` | `186848e19a47a34c4bb728b5a51a7d2e` | `NOT_FOUND` | **FAIL (Missing)** |
| `v7_ensemble_predictions.csv` | `7f49b7f2118cf715d7bb0b94f038e42c` | `7f49b7f2118cf715d7bb0b94f038e42c` | **PASS** |
| `v7_expA_partial_NBEATSx_MSE_NOSPILL_s42.csv` | `22ab1cba53543d1b2cbdfea1913633e7` | `22ab1cba53543d1b2cbdfea1913633e7` | **PASS** |
| `v7_expA_partial_NBEATSx_MSE_SPILL_s42.csv` | `6d40376b576fd9f3856f744e73975b55` | `6d40376b576fd9f3856f744e73975b55` | **PASS** |
| `v7_expA_partial_TFT_MAE_NOSPILL_s42.csv` | `ff1b08990acab4fcae5d1006e9eca82d` | `ff1b08990acab4fcae5d1006e9eca82d` | **PASS** |
| `v7_expA_partial_TFT_MAE_SPILL_s42.csv` | `fbf1a066dae4a4529e04213e6196b8a8` | `fbf1a066dae4a4529e04213e6196b8a8` | **PASS** |
| `v7_expB_full_NBEATSx_REFIT60_NOSPILL.csv` | `c042990f78340e9ee2d641b7ceda09e2` | `c042990f78340e9ee2d641b7ceda09e2` | **PASS** |
| `v7_expB_full_NBEATSx_REFIT60_SPILL.csv` | `f854aab0e515a628c8a90c7c4803e509` | `f854aab0e515a628c8a90c7c4803e509` | **PASS** |
| `v7_expB_full_TFT_REFIT60_NOSPILL.csv` | `15cc081ba11fd5b37904322a227aca6a` | `15cc081ba11fd5b37904322a227aca6a` | **PASS** |
| `v7_expB_full_TFT_REFIT60_SPILL.csv` | `140bb713bdf51ae49c4f6ddc48760683` | `140bb713bdf51ae49c4f6ddc48760683` | **PASS** |
| `v7_expB_partial_NBEATSx_REFIT60_NOSPILL_block1.csv` | `c042990f78340e9ee2d641b7ceda09e2` | `c042990f78340e9ee2d641b7ceda09e2` | **PASS** |
| `v7_expB_partial_NBEATSx_REFIT60_SPILL_block1.csv` | `f854aab0e515a628c8a90c7c4803e509` | `f854aab0e515a628c8a90c7c4803e509` | **PASS** |
| `v7_expB_partial_TFT_REFIT60_NOSPILL_block1.csv` | `15cc081ba11fd5b37904322a227aca6a` | `15cc081ba11fd5b37904322a227aca6a` | **PASS** |
| `v7_expB_partial_TFT_REFIT60_SPILL_block1.csv` | `140bb713bdf51ae49c4f6ddc48760683` | `140bb713bdf51ae49c4f6ddc48760683` | **PASS** |

### 1.2 File Completeness
- **Output Table**: [`task1_file_completeness.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task1_file_completeness.csv)

| Component | Expected | Actual | Missing | Status |
| :--- | :--- | :--- | :--- | :--- |
| Experiment A Partials (5 seeds x 4 configs) | 20 files | 4 files | 16 files | **FAIL** (Only seed 42 present; seeds 7, 13, 21, 99 missing) |
| Experiment B Partials (6 blocks x 4 configs) | 24 files | 4 files | 20 files | **FAIL** (Only block 1 present; blocks 2–6 missing) |
| Experiment B Full Concat (4 configs) | 4 files | 4 files | 0 files | **FAIL** (Contains only block 1: 30 rows instead of 3,350) |
| Ensemble Predictions (`v7_ensemble_predictions.csv`) | 1 file (13,400 rows) | 1 file (240 rows) | 0 files | **FAIL** (Only 3 cutoffs x 8 configs x 10 horizons = 240 rows) |
| Training Logs (`results/v7/logs/`) | 1 directory | 0 directories | 1 directory | **FAIL** (Folder `logs/` not present) |
| Notebook File (`__notebook__.ipynb`) | 1 file | 0 files | 1 file | **FAIL** (Listed in manifest, missing on disk) |

### 1.3 Schema, Row Counts, and Range Alignment
- **Output Table**: [`task1_schema_row_counts.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task1_schema_row_counts.csv)
- **NaN Count**: 0 across all 13 CSVs (**PASS**).
- **Duplicate `(cutoff, h)`**: 0 across all 13 CSVs (**PASS**).
- **Row Count per Partial File**: 30 rows (Expected: 3,350 rows) (**FAIL**).
- **Row Count `v7_ensemble_predictions.csv`**: 240 rows (Expected: 13,400 rows) (**FAIL**).
- **Cutoff Range**: `2023-08-24` to `2023-08-26` (3 cutoffs; v6 expected: 335 cutoffs `2023-08-24` to `2024-07-23`) (**FAIL**).
- **Real Date Range**: `2025-04-24` to `2025-05-09` (v6 expected: `2025-04-24` to `2026-09-04`) (**FAIL**).

### 1.4 Setup and Configuration Audit
- **Source**: `results/v7/README.txt`
- **STAGE**: `"SMOKE"` (**FAIL** vs production target `"ALL"`).
- **neuralforecast version**: `3.1.2` (**PASS**; matches v6 output `3.1.2`).
- **Data Split**: Train 1165, Val 167, Target 344 (**PASS**; matches v6).
- **Seeds in README**: Listed as `42, 7, 13, 21, 99`, but in SMOKE execution, loop truncated after seed `42` (**FAIL**).
- **Hyperparameters**:
  - TFT: `input_size=50`, `learning_rate=0.001` (**PASS**).
  - NBEATSx: `input_size=20`, `learning_rate=0.0003` (**PASS**).
  - SMOKE overrides: `max_steps=20`, `val_check_steps=10`, `early_stop_patience_steps=2` (**FAIL** vs v6 production `max_steps=1000`, `val_check_steps=25`, `early_stop_patience_steps=5`).

---

## 2. Training Behavior Audit

### 2.1 Negative Predictions
- **Script**: `audit/run_audit_v7.py`
- **Output Table**: [`task2_negative_predictions_audit.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task2_negative_predictions_audit.csv)

| Config | Target Loss Function | Evaluated Rows | Negative Pct (All) | Negative Pct ($h=1$) | Negative Pct ($h=10$) | Min Forecast | Max Forecast | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TFT_MAE_SPILL` | MAE | 30 | 0.00% | 0.00% | 0.00% | $2.769 \times 10^{-4}$ | $2.858 \times 10^{-4}$ | **PASS** |
| `TFT_MAE_NOSPILL` | MAE | 30 | 0.00% | 0.00% | 0.00% | $1.173 \times 10^{-4}$ | $1.902 \times 10^{-4}$ | **PASS** |
| `NBEATSx_MSE_SPILL` | MSE | 30 | 0.00% | 0.00% | 0.00% | $4.268 \times 10^{-4}$ | $1.155 \times 10^{-3}$ | **PASS** |
| `NBEATSx_MSE_NOSPILL` | MSE | 30 | 0.00% | 0.00% | 0.00% | $4.745 \times 10^{-4}$ | $1.092 \times 10^{-3}$ | **PASS** |
| `TFT_REFIT60_SPILL` | MSE (v6 original) | 30 | 0.00% | 0.00% | 0.00% | $9.485 \times 10^{-4}$ | $1.037 \times 10^{-3}$ | **PASS** |
| `TFT_REFIT60_NOSPILL` | MSE (v6 original) | 30 | 0.00% | 0.00% | 0.00% | $6.508 \times 10^{-4}$ | $7.295 \times 10^{-4}$ | **PASS** |
| `NBEATSx_REFIT60_SPILL` | MAE (v6 original) | 30 | 23.33% | 66.67% | 0.00% | $-9.495 \times 10^{-5}$ | $3.759 \times 10^{-4}$ | **FAIL** |
| `NBEATSx_REFIT60_NOSPILL` | MAE (v6 original) | 30 | 13.33% | 0.00% | 33.33% | $-6.863 \times 10^{-5}$ | $3.902 \times 10^{-4}$ | **FAIL** |

*Finding*: Retraining N-BEATSx with MSE loss yields 0.00% negative predictions (down from 13.73% in v6 MAE). Retraining TFT with MAE loss yields 0.00% negative predictions.

### 2.2 Training Log Verification
- **Log Files**: Missing (`results/v7/logs/` not present).
- **Status**: **UNVERIFIED** (Per-run best step, stop step, and validation loss cannot be extracted directly without log files).

---

## 3. Experiment B Structure and Data Leakage Audit

### 3.1 Refit Block Coverage
- Expected: 6 blocks (Block 1: 60 cutoffs, Block 2: 60 cutoffs, Block 3: 60 cutoffs, Block 4: 60 cutoffs, Block 5: 60 cutoffs, Block 6: 35 cutoffs = 335 cutoffs).
- Actual: 1 block (`block1`) with 3 cutoffs (`2023-08-24`, `2023-08-25`, `2023-08-26`). Blocks 2 through 6 missing.
- Status: **FAIL**.

### 3.2 Block 1 Seed 42 Equivalence Check vs v6
- **Script**: `audit/run_audit_v7.py`
- **Output Table**: [`task3_refit_vs_v6_seed42.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task3_refit_vs_v6_seed42.csv)

| Config | Evaluated Cutoffs | Max Abs Diff | Mean Abs Diff | Pearson Correlation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TFT_REFIT60_SPILL` | 3 | $1.865 \times 10^{-4}$ | $1.012 \times 10^{-4}$ | 0.7243 | **doesn't match** |
| `TFT_REFIT60_NOSPILL` | 3 | $2.890 \times 10^{-4}$ | $2.613 \times 10^{-4}$ | 0.8409 | **doesn't match** |
| `NBEATSx_REFIT60_SPILL` | 3 | $4.630 \times 10^{-4}$ | $1.411 \times 10^{-4}$ | 0.4472 | **doesn't match** |
| `NBEATSx_REFIT60_NOSPILL` | 3 | $3.900 \times 10^{-4}$ | $1.335 \times 10^{-4}$ | 0.4494 | **doesn't match** |

*Finding*: v7 Block 1 does not match v6 Seed 42 due to `max_steps = 20` in the SMOKE run vs `max_steps = 1000` in v6.

---

## 4. Independent Recomputation

### 4.1 Recomputed Metrics on Common Cutoffs ($N=3$)
- **Script**: `audit/run_audit_v7.py`
- **Output Table**: [`task4_independent_recomputation_common3.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task4_independent_recomputation_common3.csv)

| Group | Model / Config | $h$ | MSE | MAE | $R^2_{\text{OOS}}$ | QLIKE | Non-Pos (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benchmark** | HAR | $h=1$ | $5.629 \times 10^{-7}$ | $7.452 \times 10^{-4}$ | 0.4915 | 2.4367 | 0.00% |
| **Benchmark** | Hist-Exp | $h=1$ | $1.107 \times 10^{-6}$ | $1.049 \times 10^{-3}$ | 0.0000 | 2.7215 | 0.00% |
| **Benchmark** | Roll-20 | $h=1$ | $3.500 \times 10^{-7}$ | $5.850 \times 10^{-4}$ | 0.6839 | 2.2463 | 0.00% |
| **v6 Ensemble** | v6_TFT_SPILL_T0 | $h=1$ | $7.914 \times 10^{-7}$ | $8.832 \times 10^{-4}$ | 0.2851 | 2.5759 | 0.00% |
| **v6 Ensemble** | v6_NBEATSx_SPILL_T0 | $h=1$ | $3.201 \times 10^{-8}$ | $1.311 \times 10^{-4}$ | 0.9711 | 1.2050 | 0.00% |
| **v7 Exp A** | TFT_MAE_SPILL | $h=1$ | $4.753 \times 10^{-8}$ | $2.006 \times 10^{-4}$ | 0.9571 | 1.5601 | 0.00% |
| **v7 Exp A** | TFT_MAE_NOSPILL | $h=1$ | $1.606 \times 10^{-8}$ | $1.157 \times 10^{-4}$ | 0.9855 | 1.2659 | 0.00% |
| **v7 Exp A** | NBEATSx_MSE_SPILL | $h=1$ | $3.428 \times 10^{-7}$ | $5.700 \times 10^{-4}$ | 0.6904 | 2.2310 | 0.00% |
| **v7 Exp A** | NBEATSx_MSE_NOSPILL | $h=1$ | $5.190 \times 10^{-7}$ | $7.135 \times 10^{-4}$ | 0.5312 | 2.3982 | 0.00% |
| **v7 Exp B** | TFT_REFIT60_SPILL | $h=1$ | $8.706 \times 10^{-7}$ | $9.288 \times 10^{-4}$ | 0.2136 | 2.6182 | 0.00% |
| **v7 Exp B** | TFT_REFIT60_NOSPILL | $h=1$ | $3.740 \times 10^{-7}$ | $6.059 \times 10^{-4}$ | 0.6622 | 2.2709 | 0.00% |
| **v7 Exp B** | NBEATSx_REFIT60_SPILL | $h=1$ | $3.316 \times 10^{-8}$ | $1.512 \times 10^{-4}$ | 0.9700 | N/A | 66.67% |
| **v7 Exp B** | NBEATSx_REFIT60_NOSPILL | $h=1$ | $2.404 \times 10^{-8}$ | $1.370 \times 10^{-4}$ | 0.9783 | 1.2270 | 0.00% |
| **Benchmark** | HAR | $h=10$ | $1.575 \times 10^{-6}$ | $1.179 \times 10^{-3}$ | 0.0268 | 3.0240 | 0.00% |
| **Benchmark** | Hist-Exp | $h=10$ | $1.619 \times 10^{-6}$ | $1.255 \times 10^{-3}$ | 0.0000 | 3.0427 | 0.00% |
| **Benchmark** | Roll-20 | $h=10$ | $1.636 \times 10^{-6}$ | $1.102 \times 10^{-3}$ | -0.0105 | 3.0813 | 0.00% |
| **v6 Ensemble** | v6_TFT_SPILL_T0 | $h=10$ | $1.564 \times 10^{-6}$ | $1.195 \times 10^{-3}$ | 0.0336 | 3.0164 | 0.00% |
| **v6 Ensemble** | v6_NBEATSx_SPILL_T0 | $h=10$ | $1.791 \times 10^{-6}$ | $8.459 \times 10^{-4}$ | -0.1062 | 2.5771 | 0.00% |
| **v7 Exp A** | TFT_MAE_SPILL | $h=10$ | $1.973 \times 10^{-6}$ | $9.708 \times 10^{-4}$ | -0.2187 | 4.1116 | 0.00% |
| **v7 Exp A** | TFT_MAE_NOSPILL | $h=10$ | $2.206 \times 10^{-6}$ | $9.297 \times 10^{-4}$ | -0.3626 | 7.8191 | 0.00% |
| **v7 Exp A** | NBEATSx_MSE_SPILL | $h=10$ | $1.752 \times 10^{-6}$ | $1.281 \times 10^{-3}$ | -0.0823 | 3.1086 | 0.00% |
| **v7 Exp A** | NBEATSx_MSE_NOSPILL | $h=10$ | $1.708 \times 10^{-6}$ | $1.183 \times 10^{-3}$ | -0.0548 | 3.1312 | 0.00% |
| **v7 Exp B** | TFT_REFIT60_SPILL | $h=10$ | $1.549 \times 10^{-6}$ | $1.195 \times 10^{-3}$ | 0.0430 | 3.0098 | 0.00% |
| **v7 Exp B** | TFT_REFIT60_NOSPILL | $h=10$ | $1.585 \times 10^{-6}$ | $1.102 \times 10^{-3}$ | 0.0212 | 3.0275 | 0.00% |
| **v7 Exp B** | NBEATSx_REFIT60_SPILL | $h=10$ | $2.350 \times 10^{-6}$ | $9.742 \times 10^{-4}$ | -0.4517 | 21.3849 | 0.00% |
| **v7 Exp B** | NBEATSx_REFIT60_NOSPILL | $h=10$ | $2.524 \times 10^{-6}$ | $1.022 \times 10^{-3}$ | -0.5594 | N/A | 33.33% |

### 4.2 Benchmark Baseline Reference on Full 335 Cutoffs
- **Output Table**: [`task4_benchmarks_full_335cutoffs.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task4_benchmarks_full_335cutoffs.csv)

| Model | $h$ | MSE | MAE | $R^2_{\text{OOS}}$ | QLIKE | Non-Pos (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HAR | $h=1$ | $2.235 \times 10^{-6}$ | $6.717 \times 10^{-4}$ | 0.1691 | 1.7020 | 0.00% |
| Hist-Exp | $h=1$ | $2.689 \times 10^{-6}$ | $9.673 \times 10^{-4}$ | 0.0000 | 1.9694 | 0.00% |
| Roll-20 | $h=1$ | $2.407 \times 10^{-6}$ | $5.670 \times 10^{-4}$ | 0.1050 | 1.7475 | 0.00% |
| GARCH(1,1) | $h=1$ | $2.313 \times 10^{-6}$ | $6.502 \times 10^{-4}$ | 0.1399 | 1.6537 | 0.00% |
| EGARCH(1,1) | $h=1$ | $2.257 \times 10^{-6}$ | $6.480 \times 10^{-4}$ | 0.1607 | 1.6406 | 0.00% |
| v6_TFT_SPILL_T0 | $h=1$ | $2.388 \times 10^{-6}$ | $5.688 \times 10^{-4}$ | 0.1119 | 1.7738 | 0.00% |
| v6_NBEATSx_SPILL_T0 | $h=1$ | $2.272 \times 10^{-6}$ | $5.511 \times 10^{-4}$ | 0.1552 | N/A | 13.73% |
| HAR | $h=10$ | $2.440 \times 10^{-6}$ | $7.194 \times 10^{-4}$ | 0.0917 | 1.7992 | 0.00% |
| Hist-Exp | $h=10$ | $2.686 \times 10^{-6}$ | $9.638 \times 10^{-4}$ | 0.0000 | 1.9434 | 0.00% |
| Roll-20 | $h=10$ | $2.556 \times 10^{-6}$ | $5.874 \times 10^{-4}$ | 0.0485 | 2.0094 | 0.00% |
| v6_TFT_SPILL_T0 | $h=10$ | $2.447 \times 10^{-6}$ | $6.012 \times 10^{-4}$ | 0.0890 | 1.8032 | 0.00% |
| v6_NBEATSx_SPILL_T0 | $h=10$ | $2.745 \times 10^{-6}$ | $5.978 \times 10^{-4}$ | -0.0218 | N/A | 12.54% |

---

## 5. Statistical Tests Audit

### 5.1 Diebold-Mariano Tests vs Baselines
- **Specification**: HLN finite-sample correction, HAC bandwidth $h-1$. Evaluated on common cutoffs ($N=3$).
- **Family Multiplicity**: Paper family $m=12$; Full family $m=24$.
- **Output Table**: [`task5_dm_tests_vs_baselines.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task5_dm_tests_vs_baselines.csv)
- **Results vs HAR ($h=1$, MSE Loss)**:
  - `TFT_MAE_SPILL`: DM stat = -13.067, $p_{\text{raw}} = 0.0058$, $p_{\text{bonf,12}} = 0.0697$ (**better**).
  - `TFT_MAE_NOSPILL`: DM stat = -15.176, $p_{\text{raw}} = 0.0043$, $p_{\text{bonf,12}} = 0.0519$ (**better**).
  - `NBEATSx_MSE_SPILL`: DM stat = -4.137, $p_{\text{raw}} = 0.0538$, $p_{\text{bonf,12}} = 0.6455$ (**tie**).
  - `NBEATSx_MSE_NOSPILL`: DM stat = -0.730, $p_{\text{raw}} = 0.5413$, $p_{\text{bonf,12}} = 1.0000$ (**tie**).
  - `TFT_REFIT60_SPILL`: DM stat = +9.190, $p_{\text{raw}} = 0.0117$, $p_{\text{bonf,12}} = 0.1399$ (**tie**).
  - `TFT_REFIT60_NOSPILL`: DM stat = -3.766, $p_{\text{raw}} = 0.0638$, $p_{\text{bonf,12}} = 0.7656$ (**tie**).
  - `NBEATSx_REFIT60_SPILL`: DM stat = -13.882, $p_{\text{raw}} = 0.0051$, $p_{\text{bonf,12}} = 0.0617$ (**better**).
  - `NBEATSx_REFIT60_NOSPILL`: DM stat = -15.420, $p_{\text{raw}} = 0.0042$, $p_{\text{bonf,12}} = 0.0502$ (**better**).

### 5.2 Pairwise Diebold-Mariano Tests ($m=4$)
- **Output Table**: [`task5_pairwise_dm_tests.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task5_pairwise_dm_tests.csv)

| Comparison | Horizon | DM Stat | $p_{\text{raw}}$ | $p_{\text{bonf}}$ | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TFT Loss Effect (MAE vs MSE) - SPILL | $h=1$ | -7.6948 | 0.0000 | 0.0000 | **better** |
| TFT Loss Effect (MAE vs MSE) - NOSPILL | $h=1$ | -3.8630 | 0.0001 | 0.0004 | **better** |
| NBEATSx Loss Effect (MSE vs MAE) - SPILL | $h=1$ | +4.6418 | 0.0000 | 0.0000 | **worse** |
| NBEATSx Loss Effect (MSE vs MAE) - NOSPILL | $h=1$ | +6.9278 | 0.0000 | 0.0000 | **worse** |
| SPILL vs NOSPILL (TFT MAE) | $h=1$ | +2.7322 | 0.0063 | 0.0252 | **worse** |
| SPILL vs NOSPILL (NBEATSx MSE) | $h=1$ | -7.0397 | 0.0000 | 0.0000 | **better** |
| Refit vs Single-Fit (TFT SPILL Seed 42) | $h=1$ | +2.9446 | 0.0032 | 0.0129 | **worse** |
| Refit vs HAR (TFT SPILL) | $h=1$ | +9.1897 | 0.0000 | 0.0000 | **worse** |

### 5.3 Model Confidence Set (MCS)
- **Specification**: Hansen, Lunde, Nason (2011), $\alpha = 0.10$, 5,000 bootstrap replications. Evaluated on common cutoffs ($N=3$).
- **Output Table**: [`task5_mcs_results_common3.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task5_mcs_results_common3.csv)
- **Result**: All 8 v7 configs and baselines retained in $\mathcal{M}^*_{90\%}$ with $p = 1.0000$ (**tie**; test lacks power with $N=3$).

---

## 6. Answers to Direct Audit Questions

- **Output Table**: [`task6_direct_questions_answers.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7/task6_direct_questions_answers.csv)

### 6a. Does TFT-vs-NBEATSx MSE gap shrink when loss is matched?
- **Answer**: **doesn't match**.
- **Values**:
  - v6 full gap ($N=335$, $h=1$): $+1.163 \times 10^{-7}$ (TFT $2.388 \times 10^{-6}$ vs N-BEATSx $2.272 \times 10^{-6}$).
  - v7 loss-matched gap ($N=3$, $h=1$): $-2.952 \times 10^{-7}$ (TFT MAE $4.753 \times 10^{-8}$ vs N-BEATSx MSE $3.428 \times 10^{-7}$).
- **Evidence**: `results/v6/v6_ensemble_predictions.csv` and `results/v7/v7_ensemble_predictions.csv`.

### 6b. Does any new config have lower MSE than HAR at $h=1$ or $h=10$?
- **Answer**: **better on point MSE ($N=3$), but unverified on full sample**.
- **Values**:
  - At $h=1$ ($N=3$): `TFT_MAE_SPILL` MSE is $4.753 \times 10^{-8}$ vs HAR $5.629 \times 10^{-7}$ ($p_{\text{bonf}} = 0.0697$).
  - Full sample ($N=335$): **UNVERIFIED** due to SMOKE execution.
- **Evidence**: `audit_outputs/v7/task4_independent_recomputation_common3.csv`.

### 6c. Does periodic refit change ranking vs HAR by more than seed spread?
- **Answer**: **tie**.
- **Values**: On $N=3$, `TFT_REFIT60_SPILL` MSE is $8.706 \times 10^{-7}$ vs HAR $5.629 \times 10^{-7}$ ($p_{\text{bonf}} = 0.1399$, tie).
- **Evidence**: `audit_outputs/v7/task5_dm_tests_vs_baselines.csv`.

### 6d. Does SPILL vs NOSPILL stay a tie in every new config?
- **Answer**: **doesn't match on $N=3$; UNVERIFIED on full sample**.
- **Values**: On $N=3$, pairwise DM yields $p_{\text{bonf}} = 0.0252$ (TFT MAE) and $p_{\text{bonf}} = 0.0000$ (N-BEATSx MSE) due to small sample size.
- **Evidence**: `audit_outputs/v7/task5_pairwise_dm_tests.csv`.

### 6e. Does any new config enter or leave the MCS compared with v6?
- **Answer**: **matches (all retained in $\mathcal{M}^*_{90\%}$)**.
- **Values**: All 8 v7 configurations have $p_{\text{MCS}} = 1.0000$.
- **Evidence**: `audit_outputs/v7/task5_mcs_results_common3.csv`.

### 6f. Do 2026-02-05 and 2026-02-06 contribute most of the squared error for new configs?
- **Answer**: **UNVERIFIED (Sample Truncated)**.
- **Values**: v7 SMOKE sample covers real dates `2025-04-24` to `2025-05-09` only. Dates `2026-02-05` and `2026-02-06` are not present in `results/v7/`.
- **Evidence**: `results/v7/v7_ensemble_predictions.csv` date range.

---

## 7. Comparability Warnings

1. **Sample Truncation**: Downloaded results contain only 3 rolling cutoffs ($N=3$) instead of 335 cutoffs ($N=335$). The evaluation period spans only 15 calendar days (`2025-04-24` to `2025-05-09`).
2. **Step Count Disparity**: Models were trained with `max_steps = 20` (SMOKE) instead of `max_steps = 1000` (production).
3. **Missing Seeds**: Experiment A contains only seed `42`. Seeds `7`, `13`, `21`, and `99` are missing. No true 5-seed ensemble exists.
4. **Missing Refit Blocks**: Experiment B contains only Block 1. Blocks 2 through 6 are missing.
5. **Statistical Power**: DM tests and MCS on $N=3$ observations lack asymptotic statistical validity.

---

## 8. Reproducibility Trace Table

| Asset | Path | Hash / Status | Note |
| :--- | :--- | :--- | :--- |
| **Audit Script** | `audit/run_audit_v7.py` | `md5: verified` | Generated all task tables |
| **V7 Predictions** | `results/v7/v7_ensemble_predictions.csv` | `7f49b7f2118cf715d7bb0b94f038e42c` | 240 rows (3 cutoffs) |
| **V7 Manifest** | `results/v7/manifest.csv` | `md5: verified` | 13 CSVs match; ipynb missing |
| **V7 Readme** | `results/v7/README.txt` | `md5: verified` | Documents `STAGE = SMOKE` |
| **Git Commit** | Repository HEAD | `git rev-parse HEAD` | Unmodified base branch |

---

## 9. Final Verdict

- **Experiment A (Loss-Matched Retrain)**: **partially valid**  
  *Execution pipeline and loss reparameterization are functional (0.00% negative predictions for both TFT MAE and NBEATSx MSE), but results originate from a `STAGE = "SMOKE"` test run with only 3 cutoffs and seed 42.*

- **Experiment B (Periodic Refit 60 Days)**: **partially valid**  
  *Pipeline architecture is functional, but results originate from a `STAGE = "SMOKE"` test run with only Block 1 and 3 cutoffs.*

### Blocking Issues
1. Kaggle notebook must be executed with **`STAGE = "ALL"`** (not `"SMOKE"`) on GPU accelerator.
2. Full production run must complete all 335 cutoffs, all 5 seeds (`42, 7, 13, 21, 99`) for Experiment A, and all 6 refit blocks for Experiment B.
3. Execution log directory (`logs/`) and `__notebook__.ipynb` must be retained and downloaded alongside predictions.

### Numbers Safe to Put in Paper
- **None of the numerical values in `results/v7/` are safe to put in the paper** as final empirical results due to `STAGE = "SMOKE"` execution ($N=3$ cutoffs, `max_steps = 20`).
- **All thesis tables must continue using results from `results/v6/`** until `STAGE = "ALL"` is executed.

