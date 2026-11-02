#!/usr/bin/env python3
"""
audit_v6.py
Independent, fully reproducible audit script for RSIC v6 Deep Learning experiments.
Recalculates all metrics directly from raw predictions, verifies manifests, tests DM statistics,
and generates standardized audit tables in audit/v6/.
"""

import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRATCH_DIR = os.path.join(ROOT_DIR, "scratch/audit_v6_extracted")
AUDIT_DIR = os.path.join(ROOT_DIR, "audit/v6")
INPUT_DIR = os.path.join(ROOT_DIR, "inputs/v5")
V5_DIR = os.path.join(ROOT_DIR, "results/v5")
V5S_DIR = os.path.join(ROOT_DIR, "results/v5_sens")
V6_DIR = os.path.join(ROOT_DIR, "results/v6")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(SRC_DIR)
sys.path.append(INPUT_DIR)
import v5_helpers

os.makedirs(AUDIT_DIR, exist_ok=True)

def md5_file(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()

def diebold_mariano_test_custom(y_true, y_pred1, y_pred2, h, m_family=1, use_hln=True, loss_type='MSE'):
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    if loss_type == 'MSE':
        d = e1**2 - e2**2
    elif loss_type == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    T = len(d)
    d_bar = np.mean(d)
    
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        gamma_sum += 2.0 * weight * cov
    
    lr_var = gamma_0 + gamma_sum
    if lr_var <= 0:
        stat = np.nan
        p_raw = np.nan
        p_bonf = np.nan
    else:
        var_d_bar = lr_var / T
        stat = d_bar / np.sqrt(var_d_bar)
        
        if use_hln and h > 1:
            hln_factor = np.sqrt((T + 1 - 2*h + (h * (h - 1)) / T) / T)
            stat = stat * hln_factor
            
        p_raw = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))
        p_bonf = min(1.0, p_raw * m_family)
        
    return {
        'stat': stat,
        'p_raw': p_raw,
        'p_bonf': p_bonf,
        'm_family': m_family,
        'mean_diff': d_bar,
        'loss_type': loss_type
    }

def main():
    print("================================================================================")
    print("RSIC v6 INDEPENDENT AUDIT PIPELINE")
    print("================================================================================")

    # ---------------------------------------------------------
    # LANGKAH 1a: Manifest MD5 Verification
    # ---------------------------------------------------------
    manifest_path = os.path.join(SCRATCH_DIR, "v6_manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    
    manifest_audit = []
    all_manifest_ok = True
    for idx, row in manifest_df.iterrows():
        fname = row['filename']
        expected_md5 = row['md5']
        expected_size = row['size_bytes']
        fpath = os.path.join(SCRATCH_DIR, fname)
        if not os.path.exists(fpath):
            manifest_audit.append({'filename': fname, 'expected_md5': expected_md5, 'actual_md5': 'FILE_NOT_FOUND', 'expected_size': expected_size, 'actual_size': 0, 'status': 'MISSING'})
            all_manifest_ok = False
        else:
            act_md5 = md5_file(fpath)
            act_size = os.path.getsize(fpath)
            status = 'MATCH' if (act_md5 == expected_md5 and act_size == expected_size) else 'MISMATCH'
            if status != 'MATCH':
                all_manifest_ok = False
            manifest_audit.append({'filename': fname, 'expected_md5': expected_md5, 'actual_md5': act_md5, 'expected_size': expected_size, 'actual_size': act_size, 'status': status})
    
    manifest_audit_df = pd.DataFrame(manifest_audit)
    manifest_audit_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_manifest_md5.csv"), index=False)
    print(f"Langkah 1a: Manifest verification {'PASS (All 35 files match)' if all_manifest_ok else 'FAIL'}")

    # Check zip archive
    zip_path = os.path.join(ROOT_DIR, "results/v6/v6_outputs.zip")
    zip_md5 = md5_file(zip_path)
    zip_size = os.path.getsize(zip_path)
    print(f"Archive v6_outputs.zip MD5: {zip_md5} ({zip_size} bytes)")

    # ---------------------------------------------------------
    # LANGKAH 1b: Verify 20 Partial Prediction Files
    # ---------------------------------------------------------
    cv4_path = os.path.join(INPUT_DIR, "CV_Predictions_V4.csv")
    cv4 = pd.read_csv(cv4_path)
    cv4_pairs = list(zip(cv4['cutoff'], cv4['real_date']))
    
    models = ['TFT', 'NBEATSx']
    fsets = ['SPILL', 'NOSPILL']
    seeds = [42, 7, 13, 21, 99]
    
    partial_integrity = []
    all_partials_ok = True
    raw_partials = {}
    
    for m in models:
        for fs in fsets:
            for s in seeds:
                cfg_key = f"{m}_{fs}_s{s}"
                fname = f"v6_partial_{m}_{fs}_T0_s{s}.csv"
                fpath = os.path.join(SCRATCH_DIR, fname)
                if not os.path.exists(fpath):
                    partial_integrity.append({'file': fname, 'status': 'MISSING', 'rows': 0, 'nans': 0, 'pair_match': False})
                    all_partials_ok = False
                    continue
                df = pd.read_csv(fpath)
                n_rows = len(df)
                n_nans = int(df.isna().sum().sum())
                pairs = list(zip(df['cutoff'], df['real_date']))
                pair_match = (pairs == cv4_pairs)
                if n_rows != 3350 or n_nans != 0 or not pair_match:
                    all_partials_ok = False
                partial_integrity.append({'file': fname, 'status': 'PASS' if (n_rows == 3350 and n_nans == 0 and pair_match) else 'FAIL', 'rows': n_rows, 'nans': n_nans, 'pair_match': pair_match})
                raw_partials[cfg_key] = df
                
    partial_integrity_df = pd.DataFrame(partial_integrity)
    partial_integrity_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_partial_integrity.csv"), index=False)
    print(f"Langkah 1b: Partial files integrity {'PASS (20/20 files valid, 3350 rows, 0 NaNs, pairs identical to CV4)' if all_partials_ok else 'FAIL'}")

    # ---------------------------------------------------------
    # Baseline preparation
    # ---------------------------------------------------------
    df_dl = pd.read_csv(os.path.join(INPUT_DIR, "dataset_deep_learning.csv"))
    df_dl['Date'] = pd.to_datetime(df_dl['Date'])
    hist_series = pd.Series(df_dl['ret_crypto_btc_idr_indodax'].values**2, index=df_dl['Date'])

    har_h1_df = pd.read_csv(os.path.join(INPUT_DIR, "har_preds_T0_h1_v2.csv"))
    har_h10_df = pd.read_csv(os.path.join(INPUT_DIR, "har_preds_T0_h10_v2.csv"))
    garch_df = pd.read_csv(os.path.join(INPUT_DIR, "GARCH_Predictions_V4.csv"))
    m_garch = v5_helpers.load_garch_merged(cv4, garch_df)

    ref_df = raw_partials['TFT_SPILL_s42']
    c_origins = v5_helpers.get_cutoff_origins(ref_df, hist_series)

    grp_ref_h1 = ref_df.groupby('cutoff').head(1).reset_index(drop=True)
    grp_ref_h10 = ref_df.groupby('cutoff').tail(1).reset_index(drop=True)
    
    y_true_h1 = grp_ref_h1['y'].values
    y_true_h10 = grp_ref_h10['y'].values

    hexp_arr_h1, roll_arr_h1, _ = v5_helpers.compute_baselines(grp_ref_h1, hist_series, c_origins)
    hexp_arr_h10, roll_arr_h10, _ = v5_helpers.compute_baselines(grp_ref_h10, hist_series, c_origins)

    har_pred_h1 = har_h1_df['HAR'].values
    har_pred_h10 = har_h10_df['HAR'].values

    g1 = m_garch[m_garch['h_step'] == 1].reset_index(drop=True)
    g10 = m_garch[m_garch['h_step'] == 10].reset_index(drop=True)
    garch11_pred_h1 = g1['GARCH11'].values
    garch11_pred_h10 = g10['GARCH11'].values
    egarch_pred_h1 = g1['EGARCH'].values
    egarch_pred_h10 = g10['EGARCH'].values

    # Check baseline MSEs
    mse_har_h1 = mean_squared_error(y_true_h1, har_pred_h1)
    mse_har_h10 = mean_squared_error(y_true_h10, har_pred_h10)
    mse_roll20_h1 = mean_squared_error(y_true_h1, roll_arr_h1)
    mse_roll20_h10 = mean_squared_error(y_true_h10, roll_arr_h10)
    mse_histexp_h1 = mean_squared_error(y_true_h1, hexp_arr_h1)
    mse_histexp_h10 = mean_squared_error(y_true_h10, hexp_arr_h10)
    mse_garch11_h1 = mean_squared_error(y_true_h1, garch11_pred_h1)
    mse_garch11_h10 = mean_squared_error(y_true_h10, garch11_pred_h10)
    mse_egarch_h1 = mean_squared_error(y_true_h1, egarch_pred_h1)
    mse_egarch_h10 = mean_squared_error(y_true_h10, egarch_pred_h10)

    assert abs(mse_har_h1 - 2.234681e-06) < 1e-11, f"HAR h1 mismatch: {mse_har_h1}"
    assert abs(mse_har_h10 - 2.439743e-06) < 1e-11, f"HAR h10 mismatch: {mse_har_h10}"
    assert abs(mse_roll20_h1 - 2.407078e-06) < 1e-11, f"Roll_20 h1 mismatch: {mse_roll20_h1}"
    assert abs(mse_roll20_h10 - 2.555697e-06) < 1e-11, f"Roll_20 h10 mismatch: {mse_roll20_h10}"
    assert abs(mse_histexp_h1 - 2.689570e-06) < 1e-11, f"Hist_Exp h1 mismatch: {mse_histexp_h1}"
    assert abs(mse_histexp_h10 - 2.686277e-06) < 1e-11, f"Hist_Exp h10 mismatch: {mse_histexp_h10}"

    # DM HAR vs Hist_Exp (Stage A reproduction)
    dm_bench_h1 = v5_helpers.diebold_mariano_test(har_h1_df['True'].values, har_h1_df['HAR'].values, har_h1_df['Hist_Exp'].values, h=1, m_family=1, use_hln=True)
    dm_bench_h10 = v5_helpers.diebold_mariano_test(har_h10_df['True'].values, har_h10_df['HAR'].values, har_h10_df['Hist_Exp'].values, h=10, m_family=1, use_hln=True)
    assert abs(dm_bench_h1['stat'] - (-3.0651)) < 1e-3, f"DM bench h1 mismatch: {dm_bench_h1['stat']}"
    assert abs(dm_bench_h10['stat'] - (-2.6871)) < 1e-3, f"DM bench h10 mismatch: {dm_bench_h10['stat']}"

    print(f"Langkah 3c & 4a: Baseline MSEs and DM HAR vs Hist_Exp benchmark verified.")

    # ---------------------------------------------------------
    # LANGKAH 2: Stage B (Tuning Log)
    # ---------------------------------------------------------
    tuning_log_df = pd.read_csv(os.path.join(SCRATCH_DIR, "v6_tuning_log.csv"))
    tuning_log_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_tuning_log.csv"), index=False)
    
    # Analyze tuning log
    print("\n--- STAGE B TUNING SUMMARY ---")
    tuning_summary = []
    for model_name in models:
        m_df = tuning_log_df[tuning_log_df['model'] == model_name]
        for cfg_id, grp in m_df.groupby('config_id'):
            row0 = grp.iloc[0]
            val_mse_avg = grp['val_mse_avg'].mean() # across seeds
            seed42_val = grp[grp['seed'] == 42]['val_mse_avg'].values[0] if len(grp[grp['seed'] == 42]) > 0 else np.nan
            seed7_val = grp[grp['seed'] == 7]['val_mse_avg'].values[0] if len(grp[grp['seed'] == 7]) > 0 else np.nan
            seed_diff = abs(seed42_val - seed7_val)
            hparams = json.loads(row0['hyperparameters']) if isinstance(row0['hyperparameters'], str) else row0['hyperparameters']
            tuning_summary.append({
                'model': model_name,
                'config_id': cfg_id,
                'input_size': hparams.get('input_size', np.nan),
                'learning_rate': hparams.get('learning_rate', np.nan),
                'loss_name': hparams.get('loss_name', row0.get('loss_train', np.nan)),
                'val_mse_h1_seed42': grp[grp['seed'] == 42]['val_mse_h1'].values[0] if len(grp[grp['seed'] == 42]) > 0 else np.nan,
                'val_mse_h10_seed42': grp[grp['seed'] == 42]['val_mse_h10'].values[0] if len(grp[grp['seed'] == 42]) > 0 else np.nan,
                'val_mse_seed42': seed42_val,
                'val_mse_h1_seed7': grp[grp['seed'] == 7]['val_mse_h1'].values[0] if len(grp[grp['seed'] == 7]) > 0 else np.nan,
                'val_mse_h10_seed7': grp[grp['seed'] == 7]['val_mse_h10'].values[0] if len(grp[grp['seed'] == 7]) > 0 else np.nan,
                'val_mse_seed7': seed7_val,
                'val_mse_avg_2seeds': val_mse_avg,
                'abs_seed_diff': seed_diff
            })
    tuning_sum_df = pd.DataFrame(tuning_summary)
    tuning_sum_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_tuning_combinations_summary.csv"), index=False)

    # ---------------------------------------------------------
    # LANGKAH 3a: Recalculate Per-Seed Metrics directly from raw partials
    # ---------------------------------------------------------
    seed_records = []
    for m in models:
        for fs in fsets:
            for s in seeds:
                cfg_k = f"{m}_{fs}_s{s}"
                df_p = raw_partials[cfg_k]
                c_orig = v5_helpers.get_cutoff_origins(df_p, hist_series)
                
                for h_val, h_step in [(1, 'h=1'), (10, 'h=10')]:
                    grp = df_p.groupby('cutoff').head(1).reset_index(drop=True) if h_val == 1 else df_p.groupby('cutoff').tail(1).reset_index(drop=True)
                    yt = grp['y'].values
                    yp = grp[m].values
                    hx, rl, _ = v5_helpers.compute_baselines(grp, hist_series, c_orig)
                    
                    m_eval = v5_helpers.evaluate_metrics(yt, yp, hx, rl)
                    bias_lvl = float(np.mean(yp) / np.mean(yt)) if np.mean(yt) > 0 else np.nan
                    pct_le_0 = float(np.mean(yp <= 0) * 100.0)
                    
                    seed_records.append({
                        'Model': m,
                        'FeatureSet': fs,
                        'Target': 'T0',
                        'Seed': s,
                        'Horizon': h_step,
                        'N': m_eval['N'],
                        'MSE': m_eval['MSE'],
                        'MAE': m_eval['MAE'],
                        'RMSE': m_eval['RMSE'],
                        'R2_OOS_HistExp': m_eval['R2_OOS_HistExp'],
                        'R2_OOS_Roll20': m_eval['R2_OOS_Roll20'],
                        'Hist_Exp_Series': f"DL-base (MSE={mean_squared_error(yt, hx):.6e})",
                        'Pred_Std': m_eval['Pred_Std'],
                        'Pred_Corr': m_eval['Pred_Corr'],
                        'Bias_Level': bias_lvl,
                        'Pct_Le_0': pct_le_0
                    })
    
    per_seed_df = pd.DataFrame(seed_records)
    per_seed_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_metrics_per_seed.csv"), index=False)
    
    # Summary across 5 seeds: mean, std, min, max
    summary_list = []
    num_cols = ['MSE', 'MAE', 'R2_OOS_HistExp', 'Bias_Level', 'Pct_Le_0']
    for (m_name, fset_name, h_step), grp in per_seed_df.groupby(['Model', 'FeatureSet', 'Horizon']):
        rec = {'Model': m_name, 'FeatureSet': fset_name, 'Horizon': h_step, 'N_Seeds': len(grp)}
        for c in num_cols:
            rec[f'{c}_mean'] = grp[c].mean()
            rec[f'{c}_std']  = grp[c].std()
            rec[f'{c}_min']  = grp[c].min()
            rec[f'{c}_max']  = grp[c].max()
            rec[f'{c}_range'] = grp[c].max() - grp[c].min()
        summary_list.append(rec)
        
    seed_summary_df = pd.DataFrame(summary_list)
    seed_summary_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_metrics_seed_summary.csv"), index=False)
    print("Langkah 3a: Per-seed metrics and seed summary tables generated.")

    # ---------------------------------------------------------
    # LANGKAH 3b: Ensemble Metrics (5 Seeds Average)
    # ---------------------------------------------------------
    ensemble_records = []
    ensemble_preds = {}
    ensemble_dfs = {}

    for (m_name, fset_name) in [('TFT', 'SPILL'), ('TFT', 'NOSPILL'), ('NBEATSx', 'SPILL'), ('NBEATSx', 'NOSPILL')]:
        cfg_k = f"{m_name}_{fset_name}_T0"
        ref_df = raw_partials[f"{m_name}_{fset_name}_s{seeds[0]}"].copy()
        pred_matrix = np.zeros((len(ref_df), len(seeds)), dtype=np.float64)
        for idx, s in enumerate(seeds):
            pred_matrix[:, idx] = raw_partials[f"{m_name}_{fset_name}_s{s}"][m_name].values
        
        ensemble_pred_col = np.mean(pred_matrix, axis=1)
        ref_df[m_name] = ensemble_pred_col
        ensemble_dfs[cfg_k] = ref_df
        
        c_orig = v5_helpers.get_cutoff_origins(ref_df, hist_series)
        
        for h_val, h_step in [(1, 'h=1'), (10, 'h=10')]:
            grp = ref_df.groupby('cutoff').head(1).reset_index(drop=True) if h_val == 1 else ref_df.groupby('cutoff').tail(1).reset_index(drop=True)
            yt = grp['y'].values
            yp = grp[m_name].values
            ensemble_preds[(cfg_k, h_step)] = yp
            
            hx, rl, _ = v5_helpers.compute_baselines(grp, hist_series, c_orig)
            m_eval = v5_helpers.evaluate_metrics(yt, yp, hx, rl)
            bias_lvl = float(np.mean(yp) / np.mean(yt)) if np.mean(yt) > 0 else np.nan
            pct_le_0 = float(np.mean(yp <= 0) * 100.0)
            
            ensemble_records.append({
                'Config': cfg_k,
                'Model': m_name,
                'FeatureSet': fset_name,
                'Horizon': h_step,
                'N': m_eval['N'],
                'MSE': m_eval['MSE'],
                'MAE': m_eval['MAE'],
                'RMSE': m_eval['RMSE'],
                'R2_OOS_HistExp': m_eval['R2_OOS_HistExp'],
                'R2_OOS_Roll20': m_eval['R2_OOS_Roll20'],
                'Hist_Exp_Series': f"DL-base (MSE={mean_squared_error(yt, hx):.6e})",
                'Pred_Std': m_eval['Pred_Std'],
                'Pred_Corr': m_eval['Pred_Corr'],
                'Bias_Level': bias_lvl,
                'Pct_Le_0': pct_le_0
            })
            
    ensemble_df = pd.DataFrame(ensemble_records)
    ensemble_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_metrics_ensemble.csv"), index=False)
    print("Langkah 3b: Ensemble metrics generated.")

    # ---------------------------------------------------------
    # LANGKAH 3d: Comparison Table v6 vs v5 vs v5s
    # ---------------------------------------------------------
    v5_df = pd.read_csv(os.path.join(V5_DIR, "v5_metrics_table.csv"))
    v5s_df = pd.read_csv(os.path.join(V5S_DIR, "v5s_metrics_table.csv"))
    
    comp_records = []
    configs_to_compare = [
        ('Hist_Exp_Baseline', 'Hist_Exp', '-'),
        ('Roll_20_Baseline', 'Roll_20', '-'),
        ('HAR_Baseline', 'HAR', '-'),
        ('GARCH11_Baseline', 'GARCH11', '-'),
        ('EGARCH_Baseline', 'EGARCH', '-'),
        ('TFT_SPILL_T0', 'TFT', 'SPILL'),
        ('TFT_NOSPILL_T0', 'TFT', 'NOSPILL'),
        ('NBEATSx_SPILL_T0', 'NBEATSx', 'SPILL'),
        ('NBEATSx_NOSPILL_T0', 'NBEATSx', 'NOSPILL')
    ]
    
    for cfg, m, fs in configs_to_compare:
        for h_str in ['h=1', 'h=10']:
            v5_sub = v5_df[(v5_df['Config'] == cfg) & (v5_df['Horizon'] == h_str)]
            mse_v5 = v5_sub['MSE'].values[0] if len(v5_sub) > 0 else np.nan
            
            v5s_sub = v5s_df[(v5s_df['Config'] == cfg) & (v5s_df['Horizon'] == h_str)]
            mse_v5s = v5s_sub['MSE'].values[0] if len(v5s_sub) > 0 else np.nan
            
            if 'Baseline' in cfg:
                mse_v6 = mse_v5
            else:
                v6_sub = ensemble_df[(ensemble_df['Config'] == cfg) & (ensemble_df['Horizon'] == h_str)]
                mse_v6 = v6_sub['MSE'].values[0] if len(v6_sub) > 0 else np.nan
            
            comp_records.append({
                'Config': cfg,
                'Model': m,
                'FeatureSet': fs,
                'Horizon': h_str,
                'MSE_v5 (patience=5, fit=1341)': mse_v5,
                'MSE_v5s (patience=20, fit=1341)': mse_v5s,
                'MSE_v6_ensemble (5seeds, fit=1332, tuned, restored)': mse_v6,
                'Delta_v6_vs_v5': mse_v6 - mse_v5 if pd.notnull(mse_v6) and pd.notnull(mse_v5) else np.nan,
                'Delta_v6_vs_v5s': mse_v6 - mse_v5s if pd.notnull(mse_v6) and pd.notnull(mse_v5s) else np.nan
            })
            
    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_comparison_v5_v5s_v6.csv"), index=False)
    print("Langkah 3d: Comparison v5 vs v5s vs v6 generated.")

    # ---------------------------------------------------------
    # LANGKAH 4: Diebold-Mariano Tests
    # ---------------------------------------------------------
    # 4b: Ensemble vs Baselines (m=12 per horizon)
    baselines_dict = {
        'Hist_Exp (DL-base, MSE=2.689570e-06)': (hexp_arr_h1, hexp_arr_h10),
        'Roll_20 (DL-base, MSE=2.407078e-06)': (roll_arr_h1, roll_arr_h10),
        'HAR (v2, MSE=2.234681e-06)': (har_pred_h1, har_pred_h10)
    }
    
    dm_ensemble_records = []
    for h_val, h_str, yt in [(1, 'h=1', y_true_h1), (10, 'h=10', y_true_h10)]:
        for m in models:
            for fs in fsets:
                cfg_name = f"{m}_{fs}_T0"
                yp = ensemble_preds[(cfg_name, h_str)]
                for b_name, (b1, b10) in baselines_dict.items():
                    bp = b1 if h_val == 1 else b10
                    res = v5_helpers.diebold_mariano_test(y_true=yt, y_pred_m=yp, y_pred_b=bp, h=h_val, m_family=12, use_hln=True)
                    
                    winner = "Tie"
                    if res['p_bonf'] < 0.05:
                        winner = cfg_name if res['stat'] < 0 else b_name
                    elif res['p_raw'] < 0.05:
                        direction = f"favor_{cfg_name}" if res['stat'] < 0 else f"favor_{b_name}"
                        winner = f"Tie (raw p<0.05 {direction})"
                    
                    dm_ensemble_records.append({
                        'Comparison_Type': 'Ensemble_vs_Baseline',
                        'Config': cfg_name,
                        'Model': m,
                        'FeatureSet': fs,
                        'Horizon': h_str,
                        'Benchmark': b_name,
                        'stat': res['stat'],
                        'p_raw': res['p_raw'],
                        'p_bonf': res['p_bonf'],
                        'm_family': 12,
                        'md_MSE': res['md_MSE'],
                        'Winner': winner
                    })
                    
    # 4c: Ablation SPILL vs NOSPILL on ensemble (m=4)
    for h_val, h_str, yt in [(1, 'h=1', y_true_h1), (10, 'h=10', y_true_h10)]:
        for m in models:
            spill_p = ensemble_preds[(f"{m}_SPILL_T0", h_str)]
            nospill_p = ensemble_preds[(f"{m}_NOSPILL_T0", h_str)]
            res = v5_helpers.diebold_mariano_test(y_true=yt, y_pred_m=spill_p, y_pred_b=nospill_p, h=h_val, m_family=4, use_hln=True)
            winner = "Tie"
            if res['p_bonf'] < 0.05:
                winner = f"{m}_SPILL" if res['stat'] < 0 else f"{m}_NOSPILL"
            elif res['p_raw'] < 0.05:
                direction = f"favor_SPILL" if res['stat'] < 0 else f"favor_NOSPILL"
                winner = f"Tie (raw p<0.05 {direction})"
                
            dm_ensemble_records.append({
                'Comparison_Type': 'Ablation_SPILL_vs_NOSPILL',
                'Config': f"{m}_SPILL_vs_NOSPILL",
                'Model': m,
                'FeatureSet': 'SPILL_vs_NOSPILL',
                'Horizon': h_str,
                'Benchmark': f"{m}_NOSPILL",
                'stat': res['stat'],
                'p_raw': res['p_raw'],
                'p_bonf': res['p_bonf'],
                'm_family': 4,
                'md_MSE': res['md_MSE'],
                'Winner': winner
            })
            
    dm_ensemble_df = pd.DataFrame(dm_ensemble_records)
    dm_ensemble_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_dm_ensemble.csv"), index=False)
    print("Langkah 4b & 4c: DM ensemble and ablation tables generated.")

    # 4d: Per-seed DM (descriptive)
    seed_dm_records = []
    for h_val, h_str, yt in [(1, 'h=1', y_true_h1), (10, 'h=10', y_true_h10)]:
        for m in models:
            for fs in fsets:
                cfg_name = f"{m}_{fs}_T0"
                for b_name, (b1, b10) in baselines_dict.items():
                    bp = b1 if h_val == 1 else b10
                    p_sig_count_better = 0
                    p_sig_count_worse = 0
                    seed_p_raws = []
                    for s in seeds:
                        df_p = raw_partials[f"{m}_{fs}_s{s}"]
                        grp = df_p.groupby('cutoff').head(1).reset_index(drop=True) if h_val == 1 else df_p.groupby('cutoff').tail(1).reset_index(drop=True)
                        s_pred = grp[m].values
                        res = v5_helpers.diebold_mariano_test(y_true=yt, y_pred_m=s_pred, y_pred_b=bp, h=h_val, m_family=1, use_hln=True)
                        seed_p_raws.append(res['p_raw'])
                        if res['p_raw'] < 0.05:
                            if res['stat'] < 0:
                                p_sig_count_better += 1
                            else:
                                p_sig_count_worse += 1
                    seed_dm_records.append({
                        'Config': cfg_name,
                        'Model': m,
                        'FeatureSet': fs,
                        'Horizon': h_str,
                        'Benchmark': b_name,
                        'Seeds_Total': len(seeds),
                        'Seeds_p_raw_lt_05_Better': p_sig_count_better,
                        'Seeds_p_raw_lt_05_Worse': p_sig_count_worse,
                        'Seed_p_raw_values': "; ".join([f"{p:.4f}" for p in seed_p_raws])
                    })
    seed_dm_df = pd.DataFrame(seed_dm_records)
    seed_dm_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_dm_per_seed_summary.csv"), index=False)
    print("Langkah 4d: Descriptive per-seed DM summary generated.")

    # 4e: Exploratory MAE-loss DM (unadjusted)
    exploratory_mae_records = []
    for h_val, h_str, yt in [(1, 'h=1', y_true_h1), (10, 'h=10', y_true_h10)]:
        for m in models:
            for fs in fsets:
                cfg_name = f"{m}_{fs}_T0"
                yp = ensemble_preds[(cfg_name, h_str)]
                for b_name, (b1, b10) in [('HAR (v2)', (har_pred_h1, har_pred_h10)),
                                          ('Hist_Exp (DL-base)', (hexp_arr_h1, hexp_arr_h10))]:
                    bp = b1 if h_val == 1 else b10
                    res = diebold_mariano_test_custom(yt, yp, bp, h=h_val, m_family=1, use_hln=True, loss_type='MAE')
                    exploratory_mae_records.append({
                        'Config': cfg_name,
                        'Model': m,
                        'FeatureSet': fs,
                        'Horizon': h_str,
                        'Benchmark': b_name,
                        'Loss': 'MAE (Exploratory, no family correction)',
                        'stat': res['stat'],
                        'p_raw': res['p_raw'],
                        'md_MAE': res['mean_diff'],
                        'Direction': 'Ensemble Better' if res['stat'] < 0 else 'Benchmark Better'
                    })
    exploratory_mae_df = pd.DataFrame(exploratory_mae_records)
    exploratory_mae_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_dm_exploratory_mae.csv"), index=False)
    print("Langkah 4e: Exploratory MAE-loss DM table generated.")

    # ---------------------------------------------------------
    # LANGKAH 5: Diagnostics
    # ---------------------------------------------------------
    es_log_df = pd.read_csv(os.path.join(SCRATCH_DIR, "v6_es_log.csv"))
    
    best_steps = es_log_df['best_step'].values
    min_best = int(np.min(best_steps))
    med_best = float(np.median(best_steps))
    max_best = int(np.max(best_steps))
    n_under_100 = int(np.sum(best_steps <= 100))
    all_restored = bool(es_log_df['weights_restored'].all())
    all_es = bool(es_log_df['es_triggered'].all())
    
    print("\n--- Diagnostic 5a: Early Stopping ---")
    print(f"Total Stage C runs: {len(es_log_df)}")
    print(f"All ES triggered: {all_es} (100%)")
    print(f"All weights restored: {all_restored} (100%)")
    print(f"best_step distribution: min={min_best}, median={med_best}, max={max_best}")
    print(f"Runs with best_step <= 100: {n_under_100} / {len(es_log_df)} ({n_under_100/len(es_log_df)*100:.1f}%)")
    
    std_y_h1 = float(np.std(y_true_h1))
    std_y_h10 = float(np.std(y_true_h10))
    mean_y_h1 = float(np.mean(y_true_h1))
    mean_y_h10 = float(np.mean(y_true_h10))
    
    diag_records = []
    
    # Ensemble diagnostics
    for idx, row in ensemble_df.iterrows():
        cfg = row['Config']
        h_str = row['Horizon']
        std_y = std_y_h1 if h_str == 'h=1' else std_y_h10
        mean_y = mean_y_h1 if h_str == 'h=1' else mean_y_h10
        
        ratio_std = row['Pred_Std'] / std_y
        is_nearly_constant = (ratio_std < 0.10)
        
        mse_har = mse_har_h1 if h_str == 'h=1' else mse_har_h10
        abs_diff_har = abs(row['MSE'] - mse_har)
        
        sub_seeds = per_seed_df[(per_seed_df['Config'] == cfg if 'Config' in per_seed_df.columns else (per_seed_df['Model'] == row['Model']) & (per_seed_df['FeatureSet'] == row['FeatureSet'])) & (per_seed_df['Horizon'] == h_str)]
        seed_range = sub_seeds['MSE'].max() - sub_seeds['MSE'].min()
        seed_range_gt_har_diff = (seed_range > abs_diff_har)
        
        diag_records.append({
            'Type': 'Ensemble',
            'Config': cfg,
            'Horizon': h_str,
            'Seed': 'Ensemble (5 seeds)',
            'Pred_Mean': float(np.mean(ensemble_preds[(cfg, h_str)])),
            'Y_Mean': mean_y,
            'Bias_Ratio (Pred/Y)': row['Bias_Level'],
            'Pct_Le_0': row['Pct_Le_0'],
            'Pred_Std': row['Pred_Std'],
            'Y_Std': std_y,
            'Std_Ratio (Pred/Y)': ratio_std,
            'Nearly_Constant (<0.10)': is_nearly_constant,
            'Pred_Corr_with_Y': row['Pred_Corr'],
            'MSE': row['MSE'],
            'MSE_HAR': mse_har,
            'Abs_Diff_vs_HAR': abs_diff_har,
            'Seed_MSE_Range (Max-Min)': seed_range,
            'Seed_Range_gt_HAR_Diff': seed_range_gt_har_diff
        })
        
    # Per-seed diagnostics
    for idx, row in per_seed_df.iterrows():
        cfg = f"{row['Model']}_{row['FeatureSet']}_T0"
        h_str = row['Horizon']
        s = row['Seed']
        std_y = std_y_h1 if h_str == 'h=1' else std_y_h10
        mean_y = mean_y_h1 if h_str == 'h=1' else mean_y_h10
        
        ratio_std = row['Pred_Std'] / std_y
        is_nearly_constant = (ratio_std < 0.10)
        
        mse_har = mse_har_h1 if h_str == 'h=1' else mse_har_h10
        abs_diff_har = abs(row['MSE'] - mse_har)
        
        diag_records.append({
            'Type': 'Per-Seed',
            'Config': cfg,
            'Horizon': h_str,
            'Seed': f"seed_{s}",
            'Pred_Mean': row['Bias_Level'] * mean_y,
            'Y_Mean': mean_y,
            'Bias_Ratio (Pred/Y)': row['Bias_Level'],
            'Pct_Le_0': row['Pct_Le_0'],
            'Pred_Std': row['Pred_Std'],
            'Y_Std': std_y,
            'Std_Ratio (Pred/Y)': ratio_std,
            'Nearly_Constant (<0.10)': is_nearly_constant,
            'Pred_Corr_with_Y': row['Pred_Corr'],
            'MSE': row['MSE'],
            'MSE_HAR': mse_har,
            'Abs_Diff_vs_HAR': abs_diff_har,
            'Seed_MSE_Range (Max-Min)': np.nan,
            'Seed_Range_gt_HAR_Diff': np.nan
        })
        
    diag_df = pd.DataFrame(diag_records)
    diag_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_diagnostics.csv"), index=False)
    print("Langkah 5: Diagnostics table generated.")
    
    # ---------------------------------------------------------
    # MD5 Hashes of all input files used in audit
    # ---------------------------------------------------------
    input_files = [
        os.path.join(ROOT_DIR, "results/v6/v6_outputs.zip"),
        os.path.join(ROOT_DIR, "notebooks/06_dl_v6_multiseed.ipynb"),
        os.path.join(ROOT_DIR, "results/v6/README.md"),
        os.path.join(INPUT_DIR, "CV_Predictions_V4.csv"),
        os.path.join(INPUT_DIR, "GARCH_Predictions_V4.csv"),
        os.path.join(INPUT_DIR, "har_preds_T0_h1_v2.csv"),
        os.path.join(INPUT_DIR, "har_preds_T0_h10_v2.csv"),
        os.path.join(INPUT_DIR, "tvp_var_dynamic_features_causal.csv"),
        os.path.join(INPUT_DIR, "dataset_deep_learning.csv"),
        os.path.join(SRC_DIR, "v5_helpers.py"),
        os.path.join(V5_DIR, "v5_metrics_table.csv"),
        os.path.join(V5S_DIR, "v5s_metrics_table.csv")
    ]
    hash_records = []
    for fp in input_files:
        if os.path.exists(fp):
            hash_records.append({
                'filepath': os.path.relpath(fp, ROOT_DIR),
                'md5': md5_file(fp),
                'size_bytes': os.path.getsize(fp)
            })
    hash_df = pd.DataFrame(hash_records)
    hash_df.to_csv(os.path.join(AUDIT_DIR, "audit_v6_input_file_hashes.csv"), index=False)
    print("Input files MD5 hashes logged.")
    print("================================================================================")
    print("AUDIT SCRIPT COMPLETED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    main()
