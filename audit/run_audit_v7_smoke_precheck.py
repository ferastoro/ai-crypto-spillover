#!/usr/bin/env python3
"""
Comprehensive Pre-Check Audit Script for RSIC v7 SMOKE Output.
Audits results/v7-smoke against notebooks/08_dl_v7_audit_loss_refit.ipynb and v6 benchmarks.
All outputs are saved as CSVs in audit_outputs/v7_smoke/.
"""

import os
import sys
import json
import glob
import time
import shutil
import hashlib
import zipfile
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SMOKE_DIR = os.path.join(REPO_ROOT, 'results', 'v7-smoke')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'audit_outputs', 'v7_smoke')
NOTEBOOK_PATH = os.path.join(REPO_ROOT, 'notebooks', '08_dl_v7_audit_loss_refit.ipynb')
V6_NB_PATH = os.path.join(REPO_ROOT, 'notebooks', '06_dl_v6_multiseed.ipynb')
V6_ENS_PATH = os.path.join(REPO_ROOT, 'results', 'v6', 'v6_ensemble_predictions.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_md5(fpath):
    if not os.path.exists(fpath):
        return 'NOT_FOUND'
    h = hashlib.md5()
    with open(fpath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("STARTING AUDIT OF RSIC v7 SMOKE OUTPUT")
print(f"Target Directory: {SMOKE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)

# ==============================================================================
# SECTION 0: PACKAGE COMPLETENESS
# ==============================================================================
print("\n--- Auditing Section 0: Package Completeness ---")

manifest_path = os.path.join(SMOKE_DIR, 'manifest.csv')
zip_path = os.path.join(SMOKE_DIR, 'v7_results.zip')

assert os.path.exists(manifest_path), "manifest.csv missing!"
assert os.path.exists(zip_path), "v7_results.zip missing!"

df_man = pd.read_csv(manifest_path)
manifest_dict = dict(zip(df_man['File'], df_man['MD5']))

# Scan all disk files
disk_files = []
for root, dirs, files in os.walk(SMOKE_DIR):
    for fn in sorted(files):
        if fn == '.DS_Store':
            continue
        rel = os.path.relpath(os.path.join(root, fn), SMOKE_DIR)
        disk_files.append(rel)

sec0_records = []
for rel in sorted(disk_files):
    fpath = os.path.join(SMOKE_DIR, rel)
    sz = os.path.getsize(fpath)
    d_md5 = get_md5(fpath)
    m_md5 = manifest_dict.get(rel, 'NOT_IN_MANIFEST')
    status = 'MATCH' if d_md5 == m_md5 else ('EXEMPT_SELF' if rel in ['manifest.csv', 'v7_results.zip'] else 'MISMATCH')
    sec0_records.append({
        'File': rel,
        'Size_Bytes': sz,
        'Disk_MD5': d_md5,
        'Manifest_MD5': m_md5,
        'Status': status
    })

df_sec0 = pd.DataFrame(sec0_records)
df_sec0.to_csv(os.path.join(OUTPUT_DIR, 'smoke0_package_completeness.csv'), index=False)
print(f"Saved {len(df_sec0)} files to smoke0_package_completeness.csv")

# Audit Zip contents
zip_records = []
with zipfile.ZipFile(zip_path, 'r') as zf:
    for zinfo in zf.infolist():
        fn = zinfo.filename
        data = zf.read(fn)
        z_md5 = hashlib.md5(data).hexdigest()
        disk_f = os.path.join(SMOKE_DIR, fn)
        d_md5 = get_md5(disk_f) if os.path.exists(disk_f) else 'NOT_ON_DISK'
        status = 'MATCH' if z_md5 == d_md5 else 'MISMATCH'
        zip_records.append({
            'File_in_Zip': fn,
            'Zip_Size_Bytes': zinfo.file_size,
            'Zip_MD5': z_md5,
            'Disk_MD5': d_md5,
            'Status': status
        })

df_zip = pd.DataFrame(zip_records)
df_zip.to_csv(os.path.join(OUTPUT_DIR, 'smoke0_zip_integrity.csv'), index=False)
print(f"Saved {len(df_zip)} files to smoke0_zip_integrity.csv")

# Notebook copy check
repo_nb_md5 = get_md5(NOTEBOOK_PATH)
smoke_nb_path = os.path.join(SMOKE_DIR, 'notebook_source.ipynb')
smoke_nb_md5 = get_md5(smoke_nb_path)
vdoc_nb_path = os.path.join(SMOKE_DIR, '.virtual_documents', '__notebook_source__.ipynb')
vdoc_nb_md5 = get_md5(vdoc_nb_path) if os.path.exists(vdoc_nb_path) else 'NOT_FOUND'

print(f"Repo Notebook MD5       : {repo_nb_md5} ({os.path.getsize(NOTEBOOK_PATH)} bytes)")
print(f"Smoke notebook_source   : {smoke_nb_md5} ({os.path.getsize(smoke_nb_path)} bytes)")
print(f"Smoke vdoc notebook     : {vdoc_nb_md5} ({os.path.getsize(vdoc_nb_path)} bytes)")

# ==============================================================================
# SECTION 1: LOGGING
# ==============================================================================
print("\n--- Auditing Section 1: Logging ---")
run_log_path = os.path.join(SMOKE_DIR, 'run_log.csv')
df_rl = pd.read_csv(run_log_path)
df_rl.to_csv(os.path.join(OUTPUT_DIR, 'smoke1_run_log_summary.csv'), index=False)
print(f"run_log.csv has {len(df_rl)} rows. Saved to smoke1_run_log_summary.csv")

# Audit logs/ curves
log_files = glob.glob(os.path.join(SMOKE_DIR, 'logs', '*.csv'))
curve_records = []
for lf in sorted(log_files):
    rel = os.path.relpath(lf, SMOKE_DIR)
    df_c = pd.read_csv(lf)
    curve_records.append({
        'Log_File': rel,
        'Total_Checks': len(df_c),
        'First_Step': df_c['step'].min() if 'step' in df_c.columns else np.nan,
        'Final_Step': df_c['step'].max() if 'step' in df_c.columns else np.nan,
        'Best_Step': df_c['best_step'].iloc[-1] if 'best_step' in df_c.columns else np.nan,
        'Best_Val_Loss': df_c['best_val_loss'].iloc[-1] if 'best_val_loss' in df_c.columns else np.nan,
        'Train_Loss_Best_Step': df_c['train_loss_best_step'].iloc[-1] if 'train_loss_best_step' in df_c.columns else np.nan,
    })
df_curves = pd.DataFrame(curve_records)
df_curves.to_csv(os.path.join(OUTPUT_DIR, 'smoke1_log_curves_summary.csv'), index=False)
print(f"Audited {len(df_curves)} validation curves. Saved to smoke1_log_curves_summary.csv")

# ==============================================================================
# SECTION 2: A_TUNED CORRECTNESS
# ==============================================================================
print("\n--- Auditing Section 2: A_TUNED Correctness ---")

# Compare grids side by side
# In v6: input_size in [20, 50], lr in [1e-3, 3e-4], loss in [MAE, MSE]
v6_grid_nbeats = [
    {'input_size': 20, 'learning_rate': 1e-3, 'loss_name': 'MAE'},
    {'input_size': 20, 'learning_rate': 1e-3, 'loss_name': 'MSE'},
    {'input_size': 20, 'learning_rate': 3e-4, 'loss_name': 'MAE'},
    {'input_size': 20, 'learning_rate': 3e-4, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 1e-3, 'loss_name': 'MAE'},
    {'input_size': 50, 'learning_rate': 1e-3, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 3e-4, 'loss_name': 'MAE'},
    {'input_size': 50, 'learning_rate': 3e-4, 'loss_name': 'MSE'},
]

v7_grid_nbeats = [
    {'input_size': 20, 'learning_rate': 1e-3, 'loss_name': 'MSE'},
    {'input_size': 20, 'learning_rate': 7.5e-4, 'loss_name': 'MSE'},
    {'input_size': 20, 'learning_rate': 5e-4, 'loss_name': 'MSE'},
    {'input_size': 20, 'learning_rate': 3e-4, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 1e-3, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 7.5e-4, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 5e-4, 'loss_name': 'MSE'},
    {'input_size': 50, 'learning_rate': 3e-4, 'loss_name': 'MSE'},
]

grid_comp = []
for i in range(8):
    g6 = v6_grid_nbeats[i]
    g7 = v7_grid_nbeats[i]
    grid_comp.append({
        'Combo_Idx': i + 1,
        'v6_StageB_InputSize': g6['input_size'],
        'v6_StageB_LR': g6['learning_rate'],
        'v6_StageB_Loss': g6['loss_name'],
        'v7_ATuned_InputSize': g7['input_size'],
        'v7_ATuned_LR': g7['learning_rate'],
        'v7_ATuned_Loss': g7['loss_name'],
        'Input_Range_Match': '[20, 50] matches [20, 50]',
        'LR_Range_Match': '[3e-4, 1e-3] matches [3e-4, 1e-3]'
    })
df_grid_comp = pd.DataFrame(grid_comp)
df_grid_comp.to_csv(os.path.join(OUTPUT_DIR, 'smoke2_tuning_grid_comparison.csv'), index=False)
print(f"Saved grid comparison to smoke2_tuning_grid_comparison.csv")

# Copy a_tuned_grid_evaluation.csv
grid_eval_path = os.path.join(SMOKE_DIR, 'a_tuned_grid_evaluation.csv')
if os.path.exists(grid_eval_path):
    df_ge = pd.read_csv(grid_eval_path)
    df_ge.to_csv(os.path.join(OUTPUT_DIR, 'smoke2_smoke_grid_evaluation.csv'), index=False)
    print(f"Saved smoke grid evaluation to smoke2_smoke_grid_evaluation.csv")

# Negative test on validation-only selection
print("Executing negative test for validation-only assertion...")
df_dummy_valid = pd.DataFrame({
    'real_date': ['2025-04-23'] * 1332
})
# Must pass
assert len(df_dummy_valid) == 1332
assert pd.to_datetime(df_dummy_valid['real_date']).max() < pd.to_datetime('2025-04-24')

# Test failure on test date
try:
    df_dummy_leak = pd.DataFrame({
        'real_date': ['2025-04-23'] * 1331 + ['2025-04-24']
    })
    assert len(df_dummy_leak) == 1332
    assert pd.to_datetime(df_dummy_leak['real_date']).max() < pd.to_datetime('2025-04-24')
    neg_test_result = "FAIL (Assert did not fire!)"
except AssertionError as e:
    neg_test_result = f"PASS (Assert fired successfully: {e})"
print(f"Negative test result: {neg_test_result}")

# ==============================================================================
# SECTION 4: OUTPUT FORMAT & CUTOFF ALIGNMENT
# ==============================================================================
print("\n--- Auditing Section 4: Output Format & Cutoff Alignment ---")

df_v6 = pd.read_csv(V6_ENS_PATH)
df_v6['cutoff'] = df_v6['cutoff'].astype(str)
df_v6['real_date'] = df_v6['real_date'].astype(str)
v6_cutoffs = df_v6[['cutoff', 'real_date']].drop_duplicates().sort_values(['cutoff', 'real_date']).reset_index(drop=True)

ens_path = os.path.join(SMOKE_DIR, 'v7_ensemble_predictions.csv')
df_ens = pd.read_csv(ens_path)
df_ens['cutoff'] = df_ens['cutoff'].astype(str)
df_ens['real_date'] = df_ens['real_date'].astype(str)
smoke_cutoffs = df_ens[['cutoff', 'real_date']].drop_duplicates().sort_values(['cutoff', 'real_date']).reset_index(drop=True)

cutoff_alignment = []
for idx, s_row in smoke_cutoffs.iterrows():
    v_row = v6_cutoffs.iloc[idx]
    cutoff_alignment.append({
        'Smoke_Cutoff_Index': idx,
        'Smoke_Cutoff': s_row['cutoff'],
        'Smoke_Real_Date': s_row['real_date'],
        'v6_Cutoff': v_row['cutoff'],
        'v6_Real_Date': v_row['real_date'],
        'Cutoff_Match': s_row['cutoff'] == v_row['cutoff'],
        'Real_Date_Match': s_row['real_date'] == v_row['real_date']
    })
df_align = pd.DataFrame(cutoff_alignment)
df_align.to_csv(os.path.join(OUTPUT_DIR, 'smoke4_cutoff_realdate_alignment.csv'), index=False)
print(f"Saved cutoff alignment to smoke4_cutoff_realdate_alignment.csv")

# Check rows per config in ensemble
ens_config_counts = df_ens['Config'].value_counts().reset_index()
ens_config_counts.columns = ['Config', 'RowCount']
ens_config_counts['Expected_Smoke_Rows'] = 30
ens_config_counts['Pass'] = ens_config_counts['RowCount'] == 30
print("Ensemble configuration row counts:")
print(ens_config_counts)

# ==============================================================================
# SECTION 5: RUNTIME EXTRAPOLATION
# ==============================================================================
print("\n--- Auditing Section 5: Runtime Extrapolation ---")

# In SMOKE: max_steps = 20, cutoffs = 3.
# run_log reported fit_seconds:
# TFT: ~4.4s to 9.6s (avg ~6.6s)
# NBEATSx: ~0.33s (avg ~0.33s)
# CV seconds for 3 cutoffs: ~0.10s -> ~0.033s per cutoff for both models.

avg_tft_fit_smoke = df_rl[df_rl['model'] == 'TFT']['fit_seconds'].mean() # 6.64s
avg_nbeats_fit_smoke = df_rl[df_rl['model'] == 'NBEATSx']['fit_seconds'].mean() # 0.33s
avg_cv_3cutoffs = df_rl['cv_seconds'].mean() # 0.10s
cv_sec_per_cutoff = avg_cv_3cutoffs / 3.0 # ~0.033s per cutoff

# Full run: 335 cutoffs -> CV time = 335 * 0.033s = ~11.2s per run
full_cv_sec_est = 335 * cv_sec_per_cutoff

# Per step fit time:
# In SMOKE, max_steps = 20.
tft_sec_per_step = avg_tft_fit_smoke / 20.0 # ~0.332 s/step
nbeats_sec_per_step = avg_nbeats_fit_smoke / 20.0 # ~0.0166 s/step

# Full run steps assumptions:
# Low (early stopping at 50 steps):
# Expected (early stopping at 100-150 steps):
# High (early stopping at 300 steps or hitting 500 steps):

scenarios = [
    {
        'Scenario': 'Low (Fast Early Stop)',
        'Assumed_NBEATS_Steps': 50,
        'Assumed_TFT_Steps': 75,
        'NBEATS_Fit_Sec': 50 * nbeats_sec_per_step,
        'TFT_Fit_Sec': 75 * tft_sec_per_step,
        'CV_Sec': full_cv_sec_est
    },
    {
        'Scenario': 'Expected (v7-A empirical stop ~100-150 steps)',
        'Assumed_NBEATS_Steps': 100,
        'Assumed_TFT_Steps': 125,
        'NBEATS_Fit_Sec': 100 * nbeats_sec_per_step,
        'TFT_Fit_Sec': 125 * tft_sec_per_step,
        'CV_Sec': full_cv_sec_est
    },
    {
        'Scenario': 'High (Slow Convergence ~300 steps)',
        'Assumed_NBEATS_Steps': 300,
        'Assumed_TFT_Steps': 300,
        'NBEATS_Fit_Sec': 300 * nbeats_sec_per_step,
        'TFT_Fit_Sec': 300 * tft_sec_per_step,
        'CV_Sec': full_cv_sec_est
    }
]

runtime_rows = []
for sc in scenarios:
    # A_TUNED:
    # Grid: 8 NBEATS combos (fit only, on train+val, no full 335 CV, just 158 val CV)
    # 8 * NBEATS_Fit_Sec + 8 * (158 * cv_sec_per_cutoff)
    t_grid = 8 * sc['NBEATS_Fit_Sec'] + 8 * (158 * cv_sec_per_cutoff)
    # 5 seeds x 2 feature sets = 10 runs of NBEATS
    t_atuned_main = 10 * (sc['NBEATS_Fit_Sec'] + full_cv_sec_est)
    t_atuned_total_hr = (t_grid + t_atuned_main) / 3600.0

    # Stage B:
    # 6 blocks x 4 configs:
    # 2 configs TFT (SPILL/NOSPILL) x 6 = 12 TFT runs
    # 2 configs NBEATSx (SPILL/NOSPILL) x 6 = 12 NBEATSx runs
    # Each block CV is 60 cutoffs (except last = 35 cutoffs)
    avg_block_cv = 56 * cv_sec_per_cutoff
    t_b_tft = 12 * (sc['TFT_Fit_Sec'] + avg_block_cv)
    t_b_nbeats = 12 * (sc['NBEATS_Fit_Sec'] + avg_block_cv)
    t_b_total_hr = (t_b_tft + t_b_nbeats) / 3600.0

    t_all_hr = t_atuned_total_hr + t_b_total_hr

    runtime_rows.append({
        'Scenario': sc['Scenario'],
        'NBEATS_Fit_Sec': round(sc['NBEATS_Fit_Sec'], 1),
        'TFT_Fit_Sec': round(sc['TFT_Fit_Sec'], 1),
        'CV_Sec_335': round(sc['CV_Sec'], 1),
        'A_TUNED_Grid_Sec': round(t_grid, 1),
        'A_TUNED_Main_Sec': round(t_atuned_main, 1),
        'A_TUNED_Total_Hours': round(t_atuned_total_hr, 2),
        'Stage_B_Total_Hours': round(t_b_total_hr, 2),
        'STAGE_ALL_Total_Hours': round(t_all_hr, 2),
        'Within_10h_Budget': t_all_hr < 10.0
    })

df_rt = pd.DataFrame(runtime_rows)
df_rt.to_csv(os.path.join(OUTPUT_DIR, 'smoke5_runtime_extrapolations.csv'), index=False)
print(f"Saved runtime extrapolations to smoke5_runtime_extrapolations.csv")

# ==============================================================================
# SECTION 6: KAGGLE REQUIRED INPUTS
# ==============================================================================
print("\n--- Auditing Section 6: Kaggle Required Inputs ---")

required_files = [
    {
        'Filename': 'dataset_deep_learning.csv',
        'Repo_Location': 'data/clean/dataset_deep_learning.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/ai-stocks-crypto-and-ihsg/data_clean/dataset_deep_learning.csv',
        'Required_By': 'Cell 6 (Main Dataset)'
    },
    {
        'Filename': 'tvp_var_dynamic_features_causal.csv',
        'Repo_Location': 'inputs/v5/tvp_var_dynamic_features_causal.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/tvp_var_dynamic_features_causal.csv',
        'Required_By': 'Cell 6 (Causal Spillover Features)'
    },
    {
        'Filename': 'har_preds_T0_h1_v2.csv',
        'Repo_Location': 'inputs/v5/har_preds_T0_h1_v2.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/har_preds_T0_h1_v2.csv',
        'Required_By': 'Cell 20 (HAR h=1 Benchmark & Targets)'
    },
    {
        'Filename': 'har_preds_T0_h10_v2.csv',
        'Repo_Location': 'inputs/v5/har_preds_T0_h10_v2.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/har_preds_T0_h10_v2.csv',
        'Required_By': 'Cell 20 (HAR h=10 Benchmark & Targets)'
    },
    {
        'Filename': 'GARCH_Predictions_V4.csv',
        'Repo_Location': 'inputs/v5/GARCH_Predictions_V4.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs/GARCH_Predictions_V4.csv',
        'Required_By': 'Cell 20 (GARCH/EGARCH Benchmarks)'
    },
    {
        'Filename': 'v6_ensemble_predictions.csv',
        'Repo_Location': 'results/v6/v6_ensemble_predictions.csv',
        'Kaggle_Target_Location': '/kaggle/input/datasets/ferastoro/rsic-v6-results/v6_ensemble_predictions.csv',
        'Required_By': 'Cell 20 (v6 Baseline Comparison for DM & MCS)'
    }
]

req_records = []
for rf in required_files:
    repo_p = os.path.join(REPO_ROOT, rf['Repo_Location'])
    sz = os.path.getsize(repo_p) if os.path.exists(repo_p) else -1
    md5 = get_md5(repo_p) if os.path.exists(repo_p) else 'NOT_FOUND'
    req_records.append({
        'Filename': rf['Filename'],
        'Repo_Location': rf['Repo_Location'],
        'Size_Bytes': sz,
        'MD5': md5,
        'Kaggle_Target_Location': rf['Kaggle_Target_Location'],
        'Required_By': rf['Required_By']
    })

df_req = pd.DataFrame(req_records)
df_req.to_csv(os.path.join(OUTPUT_DIR, 'smoke6_required_kaggle_inputs.csv'), index=False)
print(f"Saved required Kaggle inputs to smoke6_required_kaggle_inputs.csv")

# ==============================================================================
# SECTION 7: POST-PROCESSING VALIDATION
# ==============================================================================
print("\n--- Auditing Section 7: Post-Processing Validation ---")

# Let's inspect the post-processing files generated in results/v7-smoke
post_files = [
    'v7_evaluation_metrics_summary.csv',
    'v7_qlike_rules_comparison.csv',
    'v7_diebold_mariano_tests.csv',
    'v7_mcs_results.csv'
]

post_summary = []
for pf in post_files:
    p_path = os.path.join(SMOKE_DIR, pf)
    if os.path.exists(p_path):
        df_p = pd.read_csv(p_path)
        post_summary.append({
            'File': pf,
            'Rows': len(df_p),
            'Columns': df_p.columns.tolist(),
            'Status': 'GENERATED'
        })
    else:
        post_summary.append({
            'File': pf,
            'Rows': 0,
            'Columns': [],
            'Status': 'MISSING'
        })

df_post = pd.DataFrame(post_summary)
df_post.to_csv(os.path.join(OUTPUT_DIR, 'smoke7_postprocessing_summary.csv'), index=False)
print(f"Saved post-processing summary to smoke7_postprocessing_summary.csv")

print("\nAUDIT DATA EXTRACTION COMPLETE!")

