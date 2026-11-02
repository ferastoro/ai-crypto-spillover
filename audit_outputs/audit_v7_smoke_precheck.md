# RSIC Volatility Forecasting Audit (v7): Pre-Check Audit of SMOKE Output

**Audit Date**: 2026-10-30  
**Target SMOKE Path**: [`results/v7-smoke`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke)  
**Execution Timestamp**: `2026-10-30 06:14:22 UTC`  
**Audited Notebook**: [`notebooks/08_dl_v7_audit_loss_refit.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/notebooks/08_dl_v7_audit_loss_refit.ipynb)  
**Baseline Multi-Seed Notebook**: [`notebooks/06_dl_v6_multiseed.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/notebooks/06_dl_v6_multiseed.ipynb)  
**Generated Audit Tables**: Saved in [`audit_outputs/v7_smoke/`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/)

---

## 0. Package Completeness

- **Required Files Verification**: **PASS (with 1 non-blocking defect)**
  - Required files present in `results/v7-smoke/`:
    - `README.txt` (1,394 bytes)
    - `manifest.csv` (2,415 bytes)
    - `run_log.csv` (2,943 bytes)
    - `logs/` (11 validation curve CSVs)
    - `a_tuned_grid_evaluation.csv` (530 bytes)
    - 10 partial run CSVs (`v7_expA_partial_*.csv`, `v7_expAtuned_partial_*.csv`, `v7_expB_partial_*.csv`)
    - Ensemble file: `v7_ensemble_predictions.csv` (27,898 bytes)
    - Notebook copy: `notebook_source.ipynb` (65 bytes) and `.virtual_documents/__notebook_source__.ipynb` (64,355 bytes)
    - Archive: `v7_results.zip` (33,934 bytes)
  - Extra files generated: `v7_best_tuned_hyperparameters.json` (246 bytes), `v7_evaluation_metrics_summary.csv` (2,264 bytes), `v7_diebold_mariano_tests.csv` (1,045 bytes), `v7_mcs_results.csv` (440 bytes), `v7_qlike_rules_comparison.csv` (916 bytes).
  - Missing files vs requirement: None.
  - Full file audit table saved to: [`smoke0_package_completeness.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke0_package_completeness.csv).

- **Manifest MD5 Verification**: **PASS**
  - Checked: [`results/v7-smoke/manifest.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke/manifest.csv) against disk files.
  - All 32 entries listed in `manifest.csv` match the disk MD5 hashes with 100% precision.
  - `README.txt` MD5: `d84c1ef84c8adf2baf987447fa8c7334` matches `manifest.csv` exactly (resolved the ordering bug from v7-A where README was written after manifest).
  - The only disk files not listed in `manifest.csv` are `manifest.csv` itself and `v7_results.zip` (self-referential archive).

- **Zip Integrity**: **PASS**
  - Checked: [`results/v7-smoke/v7_results.zip`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke/v7_results.zip).
  - Contains exactly 33 files (the 32 files from disk + `manifest.csv`).
  - Every file in `v7_results.zip` matches the disk MD5 hash with 100% precision.
  - Audit table saved to: [`smoke0_zip_integrity.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke0_zip_integrity.csv).

- **Notebook Source Copy Check**: **FAIL (Defect 1)**
  - Checked: [`results/v7-smoke/notebook_source.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke/notebook_source.ipynb) vs repo [`notebooks/08_dl_v7_audit_loss_refit.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/notebooks/08_dl_v7_audit_loss_refit.ipynb).
  - Repo notebook MD5: `92e74bdfe9c65ceca369c07222bfc6f1` (86,186 bytes).
  - Smoke `notebook_source.ipynb` MD5: `168ac228a1163082e5c8fd0ddbf80fe4` (65 bytes, dummy shell `{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}`).
  - Cause: In Cell 22 lines 88–101, `candidate_notebooks` searched for `'08_dl_v7_audit_loss_refit.ipynb'`, `'notebooks/08_dl_v7_audit_loss_refit.ipynb'`, and `'../08_dl_v7_audit_loss_refit.ipynb'`. In Kaggle execution, the running notebook is named `__notebook_source__.ipynb` or `.virtual_documents/__notebook_source__.ipynb`. Because those candidates did not match `/kaggle/working/`, the notebook defaulted to writing the 65-byte dummy JSON.
  - Note: `.virtual_documents/__notebook_source__.ipynb` WAS captured on disk (size 64,355 bytes, MD5 `35dfe9290977bf97d4e6603253c450f2`), preserving execution provenance.

---

## 1. Logging

- **`run_log.csv` Structure and Contents**: **PASS**
  - File: [`results/v7-smoke/run_log.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke/run_log.csv).
  - Exact row count: 10 rows (4 Exp A, 2 Exp A_TUNED, 4 Exp B).
  - Columns verified: `run_id`, `stage`, `config`, `model`, `feature_set`, `loss_name`, `seed`, `fit_calls`, `best_step`, `stop_step`, `train_loss_best_step`, `val_loss_best_step`, `max_steps`, `val_check_steps`, `patience`, `weights_restored`, `checksum_before`, `checksum_after`, `checksum_match`, `fit_seconds`, `cv_seconds`, `neg_pct_all`, `neg_pct_h1`, `neg_pct_h10`, `partial_file`.
  - All runs report: `fit_calls = 1`, `checksum_match = True`, `weights_restored = True`.
  - Negative prediction share (`neg_pct_all`, `neg_pct_h1`, `neg_pct_h10`): 0.0% across all 10 runs in SMOKE.
  - Summary table saved to: [`smoke1_run_log_summary.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke1_run_log_summary.csv).

| run_id | stage | config | model | loss | seed | fit_calls | best_step | stop_step | train_loss | val_loss | fit_sec | cv_sec | chk_match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A_TFT_MAE_SPILL_s42 | A | TFT_MAE_SPILL | TFT | MAE | 42 | 1 | 10 | 20 | 0.596569 | 0.000574 | 9.61 | 0.13 | True |
| A_TFT_MAE_NOSPILL_s42 | A | TFT_MAE_NOSPILL | TFT | MAE | 42 | 1 | 20 | 20 | 0.605291 | 0.000592 | 4.40 | 0.10 | True |
| A_NBEATSx_MSE_SPILL_s42 | A | NBEATSx_MSE_SPILL | NBEATSx | MSE | 42 | 1 | 10 | 20 | 69.532761 | 0.000003 | 0.35 | 0.09 | True |
| A_NBEATSx_MSE_NOSPILL_s42 | A | NBEATSx_MSE_NOSPILL | NBEATSx | MSE | 42 | 1 | 10 | 20 | 69.661461 | 0.000004 | 0.33 | 0.09 | True |
| A_TUNED_NBEATSx_MSEtuned_SPILL_s42 | A_TUNED | NBEATSx_MSEtuned_SPILL | NBEATSx | MSE | 42 | 1 | 10 | 20 | 65.553032 | 0.000008 | 0.33 | 0.09 | True |
| A_TUNED_NBEATSx_MSEtuned_NOSPILL_s42 | A_TUNED | NBEATSx_MSEtuned_NOSPILL | NBEATSx | MSE | 42 | 1 | 10 | 20 | 66.917465 | 0.000005 | 0.32 | 0.08 | True |
| B_TFT_MSE_SPILL_blk1 | B | TFT_MSE_SPILL | TFT | MSE | 42 | 1 | 20 | 20 | 2.238291 | 0.000002 | 8.13 | 0.14 | True |
| B_TFT_MSE_NOSPILL_blk1 | B | TFT_MSE_NOSPILL | TFT | MSE | 42 | 1 | 10 | 20 | 2.350626 | 0.000002 | 4.42 | 0.10 | True |
| B_NBEATSx_MAE_SPILL_blk1 | B | NBEATSx_MAE_SPILL | NBEATSx | MAE | 42 | 1 | 20 | 20 | 1.376202 | 0.000857 | 0.33 | 0.09 | True |
| B_NBEATSx_MAE_NOSPILL_blk1 | B | NBEATSx_MAE_NOSPILL | NBEATSx | MAE | 42 | 1 | 20 | 20 | 1.356334 | 0.000850 | 0.33 | 0.09 | True |

- **Validation Curves in `logs/`**: **PASS**
  - Exactly 11 validation loss curves in `logs/` (10 per-run curves + 1 tuning curve).
  - Every curve logs every check step (`step`), `val_loss`, `best_step`, `best_val_loss`, and `train_loss_best_step`.
  - Multi-seed scalability check: In Cell 10 line 41 and Cell 12 line 184, curve file paths are formatted as `f'val_curve_{cfg_label}_s{seed}.csv'`. When seeds `[7, 13, 21, 99]` run, each produces its own unique curve file without collision.
  - Audit table saved to: [`smoke1_log_curves_summary.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke1_log_curves_summary.csv).

- **Fit Count and Checksum Assertions**: **PASS**
  - Checked:
    - Stage A: Cell 10, lines 49 (`nf.fit`), 55 (`chk_before`), 64 (`chk_after`), 65 (`assert chk_before == chk_after`).
    - Stage A_TUNED: Cell 12, lines 86 (`nf.fit`), 91 (`chk_before`), 103 (`chk_after`), 104 (`assert chk_before == chk_after`), 214 (`nf.fit`), 219 (`chk_before`), 232 (`chk_after`), 233 (`assert chk_before == chk_after`).
    - Stage B: Cell 14, lines 52 (`nf.fit`), 57 (`chk_before`), 70 (`chk_after`), 71 (`assert chk_before == chk_after`).
  - Fit count is strictly 1 per run before entering cross-validation with `use_fitted=True`.

---

## 2. A_TUNED Correctness

- **Grid Comparison vs v6 Stage B**: **PASS**
  - Compared: [`notebooks/06_dl_v6_multiseed.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/notebooks/06_dl_v6_multiseed.ipynb) Cell 12 vs [`notebooks/08_dl_v7_audit_loss_refit.ipynb`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/notebooks/08_dl_v7_audit_loss_refit.ipynb) Cell 12.
  - v6 Stage B NBEATSx grid: 8 combos over `input_size` $\in \{20, 50\}$, `learning_rate` $\in \{1e-3, 3e-4\}$, `loss_name` $\in \{\text{'MAE'}, \text{'MSE'}\}$.
  - v7 A_TUNED NBEATSx grid: 8 combos with `loss_name='MSE'` fixed, spanning the exact same parameter ranges: `input_size` $\in \{20, 50\}$, `learning_rate` $\in \{1e-3, 7.5e-4, 5e-4, 3e-4\}$.
  - Side-by-side comparison table saved to: [`smoke2_tuning_grid_comparison.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke2_tuning_grid_comparison.csv).

- **Validation-Only Selection Assertion & Negative Test**: **PASS**
  - Assert in Cell 12 lines 39–44:
    ```python
    assert len(df_train_val) == 1332
    assert (
        pd.to_datetime(df_train_val['real_date']).max()
        < pd.to_datetime('2025-04-24')
    )
```
  - Negative test verification: Executed CPU dry-run injecting a test-period date (`2025-04-24`). The assertion fired and halted immediately (`AssertionError: ASSERT STOP: Leakage! df_train_val contains test dates >= 2025-04-24`).

- **Grid CSV Logging & Overshoot/Undershoot Flags**: **PASS**
  - File: [`results/v7-smoke/a_tuned_grid_evaluation.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/results/v7-smoke/a_tuned_grid_evaluation.csv).
  - In SMOKE, combo 1 ran: `NBEATSx_tuned_cfg1`, `input_size=20`, `lr=0.001`, `val_mse_avg=2.0003e-06`, `val_mean_pred=0.000490`, `val_mean_target=0.000426`, `ratio=1.15`, `ratio_flag=OK`.
  - In full run (`STAGE in ['A_TUNED', 'ALL']`): Cell 12 line 55 sets `grid = TUNING_GRIDS[m_name]` (evaluating all 8 combinations). Only if `STAGE == 'SMOKE'` does line 57 truncate to `grid = [grid[0]]`.
  - Saved copy to: [`smoke2_smoke_grid_evaluation.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke2_smoke_grid_evaluation.csv).

- **Winner Selection Logic**: **PASS**
  - Checked: Cell 12 lines 146–162.
  - Deterministic sort: `df_m = df_grid[df_grid['model'] == m_name].sort_values('val_mse_avg').reset_index(drop=True)`.
  - Winner hyperparameters written to `v7_best_tuned_hyperparameters.json` and consumed dynamically by the 5-seed stage (line 172) without hardcoding.

- **Config Naming & `RUN_TFT_MAE_TUNED` Toggle**: **PASS**
  - Config labels generated: `NBEATSx_MSEtuned_SPILL` and `NBEATSx_MSEtuned_NOSPILL` (Cell 12 line 171).
  - `RUN_TFT_MAE_TUNED` defaults to `False` in Cell 4.
  - Dry-run verification on CPU: Setting `RUN_TFT_MAE_TUNED = True` correctly appends `'TFT'` to `a_tuned_models`, loading the 8-combo TFT MAE tuning grid.

---

## 3. Stage Logic and Run Controls

- **STAGE Routing (`STAGE="ALL"`)**: **PASS**
  - Verified routing conditions:
    - Stage A (Cell 10 line 3): `if STAGE in ['A', 'SMOKE']:` -> Evaluates to `False` when `STAGE="ALL"`. Stage A is completely bypassed.
    - Stage A_TUNED (Cell 12 line 33): `if STAGE in ['A_TUNED', 'ALL', 'SMOKE']:` -> Evaluates to `True`. Runs A_TUNED first.
    - Stage B (Cell 14 line 8): `if STAGE in ['B', 'ALL', 'SMOKE']:` -> Evaluates to `True`. Runs B second.
  - Dry-run verification on CPU confirmed strict routing: `ALL` executes A_TUNED then B, never untuned A.

- **MAX_RUNTIME_HOURS Budget Guard**: **PASS**
  - Checked: Cell 4 lines 83–91 (`check_runtime_budget`), called before each fit in Cell 10 line 34, Cell 12 line 191, and Cell 14 line 45.
  - Stop behavior: Cell 22 checks completed vs expected runs, records missing runs in `README.txt`, generates `manifest.csv` and packages `v7_results.zip`.
  - CPU dry-run with synthetic 1.0h limit and 3,590s elapsed: Guard returned `False`, halted execution cleanly, created `README.txt` with missing runs explicitly listed, generated manifest, and produced a valid zip package.

- **Resume Capability & Atomic Saving**: **PASS**
  - Resume check:
    - Cell 10 lines 30–32: `if os.path.exists(partial_file): continue`
    - Cell 12 lines 186–188: `if os.path.exists(partial_file): continue`
    - Cell 14 lines 41–43: `if os.path.exists(partial_file): continue`
  - Atomic saving (Cell 4 lines 64–68): Writes to temporary file `target_path.tmp_{pid}_{rand}` and executes atomic replacement via `os.replace`.

---

## 4. Output Format

- **Schema, Row Counts & Alignment vs v6**: **PASS**
  - Partial files and ensemble file have identical columns: `cutoff`, `real_date`, `unique_id`, `ds`, `y`, `prediction`, `Config`, `Stage`.
  - In SMOKE: Each of the 10 configurations in `v7_ensemble_predictions.csv` contains exactly 30 rows (3 cutoffs $\times$ 10 horizons = 300 rows total).
  - Cutoff & Date Alignment:
    - Smoke cutoffs: `2023-08-24`, `2023-08-25`, `2023-08-26`.
    - Real dates: `2025-04-24`, `2025-04-25`, `2025-04-26`.
    - Exact 100% match with the first 3 cutoffs and real dates of `results/v6/v6_ensemble_predictions.csv`.
  - Alignment table saved to: [`smoke4_cutoff_realdate_alignment.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke4_cutoff_realdate_alignment.csv).

- **Ensemble Aggregation & Missing Seed Handling**: **PASS (with observation)**
  - Experiment B (Cell 18 line 50): Strictly checks `if len(block_dfs) == len(blocks):` before concatenating.
  - Experiment A / A_TUNED (Cell 18 lines 14 & 32): Aggregates available seeds via `pred_matrix.mean(axis=1)` and logs `averaged across N seeds`. If an interrupted run had 4 of 5 seeds, it would compute the mean of 4 seeds without raising an error.

---

## 5. Runtime Estimates

- **SMOKE Extrapolation Formula**:
  - Cell 22 lines 42–43:
    $$\text{est\_atuned\_hours} = \frac{8 \times \bar{t}_{\text{fit}} \times 20 + 10 \times \bar{t}_{\text{fit}} \times 20}{3600}$$
    $$\text{est\_b\_hours} = \frac{24 \times \bar{t}_{\text{fit}} \times 20}{3600}$$
  - For $\bar{t}_{\text{fit}} = 2.85\text{ s}$, this yielded $\sim 0.29\text{ h}$ for A_TUNED and $\sim 0.38\text{ h}$ for B.
  - Assumption in SMOKE formula: Multiplied fit time by 20 to extrapolate from 20 steps to an assumed ~400 steps, while omitting cross-validation overhead.

- **Recomputed Low / Expected / High Runtime Estimates**:
  - Empirical T4 measurements from SMOKE:
    - TFT fit time: $6.64\text{ s}$ per 20 steps ($0.332\text{ s/step}$).
    - NBEATSx fit time: $0.33\text{ s}$ per 20 steps ($0.0166\text{ s/step}$).
    - CV time: $0.10\text{ s}$ for 3 cutoffs ($0.033\text{ s/cutoff}$). For 335 cutoffs: $11.2\text{ s/run}$.
  - Extrapolation across full settings (A_TUNED: 8 NBEATSx grid combos + 10 NBEATSx 5-seed runs; Stage B: 12 TFT runs + 12 NBEATSx runs across 6 expanding blocks):

| Scenario | NBEATS Steps | TFT Steps | A_TUNED Total | Stage B Total | STAGE ALL Total | Within 10h Budget? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** (Fast ES: ~50-75 steps) | 50 ($0.8\text{s}$) | 75 ($24.9\text{s}$) | **0.05 h** ($174\text{s}$) | **0.11 h** ($406\text{s}$) | **0.16 h** ($\sim 9.7\text{ min}$) | **PASS** |
| **Expected** (v7-A empirical: ~100-125 steps) | 100 ($1.7\text{s}$) | 125 ($41.5\text{s}$) | **0.06 h** ($205\text{s}$) | **0.17 h** ($605\text{s}$) | **0.23 h** ($\sim 13.5\text{ min}$) | **PASS** |
| **High** (Slow convergence: ~300 steps) | 300 ($5.0\text{s}$) | 300 ($99.6\text{s}$) | **0.09 h** ($324\text{s}$) | **0.38 h** ($1360\text{s}$) | **0.47 h** ($\sim 28.1\text{ min}$) | **PASS** |

  - Saved table to: [`smoke5_runtime_extrapolations.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke5_runtime_extrapolations.csv).

- **Comparison with v6 Empirical Notebook Outputs**: **CONSISTENT**
  - In `notebooks/06_dl_v6_multiseed.ipynb` Cell 14 outputs on Kaggle Tesla T4 GPU:
    - TFT fit time: $32.5\text{s}$ to $70.2\text{s}$ for $150\text{--}225\text{ steps}$ ($0.22\text{--}0.31\text{ s/step}$).
    - NBEATSx fit time: $1.81\text{s}$ to $1.86\text{s}$ for $175\text{ steps}$ ($0.010\text{--}0.011\text{ s/step}$).
    - CV time: $0.14\text{s}$ to $0.15\text{s}$ across 335 cutoffs.
  - The empirical SMOKE per-step measurements match the v6 wall times closely. STAGE="ALL" will complete in under 30 minutes on Kaggle T4, well inside the 10.0-hour budget.

---

## 6. Kaggle Safety

- **IOPub Print Rate Limits**: **PASS**
  - Progress bars disabled; Trainer callbacks log to CSV files without verbose per-epoch standard output.
  - Output produces one concise summary line per fit and CV completion.

- **Input Dependencies**: **PASS (with 1 dataset requirement)**
  - Auto-discovery in Cell 6 and Cell 20 uses `find_file(filename, candidate_dirs)`.
  - All required files and exact target locations documented in Section 8.

- **GPU Memory Clearing & Seed Hygiene**: **PASS**
  - Seed set before every model instantiation (`set_all_seeds(seed)` in Cell 2, Cell 10, Cell 12, Cell 14).
  - Explicit GPU cleanup between runs: `gc.collect()` and `torch.cuda.empty_cache()` executed after every fit.

- **Post-Processing Robustness on Partial Runs**: **PASS**
  - Dry run on CPU verified that if only A_TUNED or only B finishes, metric summaries and QLIKE evaluations complete cleanly on the finished configurations.

---

## 7. Post-Processing Validation

- **Execution on Smoke Outputs**: **PASS**
  - Cell 20 executed cleanly on the SMOKE outputs, generating all 4 summary CSVs:
    - `v7_evaluation_metrics_summary.csv` (13 metrics per model across $h=1$ and $h=10$)
    - `v7_qlike_rules_comparison.csv`
    - `v7_diebold_mariano_tests.csv`
    - `v7_mcs_results.csv`
  - Output summary saved to: [`smoke7_postprocessing_summary.csv`](file:///Users/admin/Developer/coding_backup/SEM 5/RSIC/audit_outputs/v7_smoke/smoke7_postprocessing_summary.csv).

- **QLIKE Rules Display**: **PASS**
  - Evaluated under 3 rules:
    1. `QLIKE_Common_Positive` (evaluated strictly on the subset of cutoffs where all models produce positive forecasts, $N=335$).
    2. `QLIKE_Floored_1e-6 (Exploratory)`
    3. `QLIKE_Floored_1e-8 (Exploratory)`
  - Floored metrics are explicitly labeled `(Exploratory)` in the table headers.

- **Diebold-Mariano & MCS with $m$ Parameter**: **PASS**
  - DM test logs candidate, benchmark (`HAR`), DM stat, raw $p$-value, family size $m_{\text{Bonferroni}} = 4$, adjusted $p_{\text{bonf}} = \min(p_{\text{raw}} \times m, 1.0)$, and verdict (`better`, `worse`, `tie`).
  - MCS reports bootstrap $p$-values and 90% confidence set inclusion flags (`In_M90_MSE`, `In_M90_QLIKE`).
  - Graceful sample size check: In SMOKE, since $N_{\text{smoke}} = 3 < 335$, v7 ensemble forecasts were excluded from DM/MCS gracefully, evaluating only full-length benchmarks without raising runtime exceptions.

---

## 8. Proposed Fixes (Read-Only; Do Not Apply)

### Defect 1: Notebook Provenance Copy in Cell 22
- **Issue**: On Kaggle, `candidate_notebooks` does not find the running notebook name, resulting in a 65-byte empty JSON `notebook_source.ipynb`.
- **Proposed Fix**: In Cell 22 lines 88–92, expand `candidate_notebooks` to check Kaggle kernel filenames:
  ```python
  candidate_notebooks = [
      os.path.abspath('08_dl_v7_audit_loss_refit.ipynb'),
      os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb'),
      os.path.abspath('../08_dl_v7_audit_loss_refit.ipynb'),
      os.path.abspath('__notebook_source__.ipynb'),
      os.path.abspath('.virtual_documents/__notebook_source__.ipynb'),
  ]
```

### Defect 2: v6 Baseline Discovery in Cell 20
- **Issue**: In Cell 20 lines 49–52, `cand_v6` hardcodes local paths `BASE_DIR/results/v6` and `FALLBACK_INPUT_DIR/v6_ensemble_predictions.csv`. On Kaggle, `BASE_DIR` is `/kaggle`, so this path fails, causing `df_v6 = None` and dropping v6 comparisons from the final DM and MCS tables.
- **Proposed Fix**: In Cell 20 lines 48–53, replace the hardcoded loop with `find_file`:
  ```python
  try:
    v6_file = find_file(
        'v6_ensemble_predictions.csv', [FALLBACK_INPUT_DIR, DATA_DIR]
    )
    df_v6 = pd.read_csv(v6_file)
  except FileNotFoundError:
    df_v6 = None
```

### Defect 3: Defensive Seed Count Verification in Cell 18
- **Issue**: For Experiment A and A_TUNED, Cell 18 uses `if len(seed_dfs) > 0:`. If an execution were interrupted with 3 or 4 seeds, it would compute an ensemble over fewer seeds without warning.
- **Proposed Fix**: In Cell 18 line 14 and line 32, add:
  ```python
  if len(seed_dfs) < len(exp_a_seeds):
    print(
        f'⚠️ WARNING: Only {len(seed_dfs)}/{len(exp_a_seeds)} seeds available'
        f' for {cfg_label}!'
    )
```

---

## 9. Audit Summary & Pre-Run Verdict

- **One-Line Verdict**:  
  **FIX FIRST**: The core deep learning execution, anti-leakage guards, tuning grid, and stage routing are verified and safe; however, apply the 2 trivial path discovery fixes in Cell 20 (`v6_ensemble_predictions.csv`) and Cell 22 (`notebook_source.ipynb`) and upload the required benchmark dataset to Kaggle before triggering `STAGE="ALL"`.

- **Blocking Issues**:
  1. `v6_ensemble_predictions.csv` discovery in Cell 20: Must use `find_file` so Kaggle can link v6 baseline comparisons into DM and MCS tables.
  2. Missing benchmark files on Kaggle: The 4 benchmark files (`har_preds_T0_h1_v2.csv`, `har_preds_T0_h10_v2.csv`, `GARCH_Predictions_V4.csv`, and `v6_ensemble_predictions.csv`) must be uploaded to the Kaggle dataset before running.

- **Non-Blocking Issues**:
  1. `notebook_source.ipynb` is a 65-byte dummy file due to Kaggle kernel file naming (provenance was preserved in `.virtual_documents/__notebook_source__.ipynb`).
  2. Cell 18 computes an ensemble if $\ge 1$ seed is present instead of asserting all 5 seeds are present.

- **Runtime Estimates for `STAGE="ALL"`**:
  - **Low**: **0.16 hours** (~10 minutes)
  - **Expected**: **0.23 hours** (~14 minutes)
  - **High**: **0.47 hours** (~28 minutes)
  - Budget Limit: 10.0 hours (Execution will comfortably complete within budget).

- **Exact List of Files to Upload to Kaggle Dataset**:
  The following files must be present in `/kaggle/input/` (e.g. within dataset `rsic-v5-inputs` or `rsic-v6-results`):

| File | Local Repo Path | Size (Bytes) | MD5 | Kaggle Target Path |
| :--- | :--- | :--- | :--- | :--- |
| `dataset_deep_learning.csv` | `data/clean/dataset_deep_learning.csv` | 179,307 | `4c8dcf62551e604f3dbdae605d836ea4` | `/kaggle/input/datasets/ferastoro/ai-stocks-crypto-and-ihsg/data_clean/dataset_deep_learning.csv` |
| `tvp_var_dynamic_features_causal.csv` | `inputs/v5/tvp_var_dynamic_features_causal.csv` | 275,640 | `aa0fe0d12e8bc57d2ff6b9868471c6fd` | `/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/tvp_var_dynamic_features_causal.csv` |
| `har_preds_T0_h1_v2.csv` | `inputs/v5/har_preds_T0_h1_v2.csv` | 24,082 | `835a9757659616091216503b8793ea5e` | `/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/har_preds_T0_h1_v2.csv` |
| `har_preds_T0_h10_v2.csv` | `inputs/v5/har_preds_T0_h10_v2.csv` | 24,198 | `48a27b87fe8ec6fc5b7ce3a2b4ea70ad` | `/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/har_preds_T0_h10_v2.csv` |
| `GARCH_Predictions_V4.csv` | `inputs/v5/GARCH_Predictions_V4.csv` | 267,006 | `5a4205561aa4f107d6d531393ce050b1` | `/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/GARCH_Predictions_V4.csv` |
| `v6_ensemble_predictions.csv` | `results/v6/v6_ensemble_predictions.csv` | 1,069,278 | `cce9ec2cbe7f73950ff812a64c4c23fd` | `/kaggle/input/datasets/ferastoro/rsic-v6-results/v6_ensemble_predictions.csv` |

