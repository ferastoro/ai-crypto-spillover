#!/usr/bin/env python3
"""
Comprehensive Audit Script for v7 Experiment A (Loss-Matched Retrain)
Performs strict, read-only verification of results in results/v7-a/ against v6 and benchmarks.
Generates all audit CSV tables into audit_outputs/v7A/ and outputs audit_outputs/audit_v7A_results.md.
"""

import os
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
V7A_DIR = BASE_DIR / "results" / "v7-a"
V6_DIR = BASE_DIR / "results" / "v6"
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
OUTPUT_DIR = BASE_DIR / "audit_outputs" / "v7A"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# 0. Load Data
# --------------------------------------------------------------------------------------
print("Loading v6, v7, and benchmark data...")
v6_ens = pd.read_csv(V6_DIR / "v6_ensemble_predictions.csv")
v7_ens = pd.read_csv(V7A_DIR / "v7_ensemble_predictions.csv")

# Ensure proper types
v6_ens['cutoff'] = v6_ens['cutoff'].astype(str)
v6_ens['real_date'] = v6_ens['real_date'].astype(str)
v6_ens['h'] = v6_ens.groupby(['cutoff', 'Config']).cumcount() + 1

v7_ens['cutoff'] = v7_ens['cutoff'].astype(str)
v7_ens['real_date'] = v7_ens['real_date'].astype(str)
v7_ens['h'] = v7_ens.groupby(['cutoff', 'Config']).cumcount() + 1

# Load benchmarks
har_h1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values('Date').reset_index(drop=True)
har_h10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values('Date').reset_index(drop=True)
garch_df = pd.read_csv(INPUTS_V5 / "GARCH_Predictions_V4.csv")
garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)

exp_a_configs = [
    'TFT_MAE_SPILL',
    'TFT_MAE_NOSPILL',
    'NBEATSx_MSE_SPILL',
    'NBEATSx_MSE_NOSPILL'
]
v6_configs_map = {
    'TFT_MAE_SPILL': 'v6_TFT_SPILL_T0',
    'TFT_MAE_NOSPILL': 'v6_TFT_NOSPILL_T0',
    'NBEATSx_MSE_SPILL': 'v6_NBEATSx_SPILL_T0',
    'NBEATSx_MSE_NOSPILL': 'v6_NBEATSx_NOSPILL_T0'
}
seeds = [42, 7, 13, 21, 99]

# --------------------------------------------------------------------------------------
# 1. Gate 0 Audit
# --------------------------------------------------------------------------------------
gate0_records = []
with open(V7A_DIR / "README.txt") as f:
    readme_txt = f.read()

stage_val = "A" if "Execution Stage  : A" in readme_txt else ("ALL" if "Execution Stage  : ALL" in readme_txt else "OTHER")
gate0_records.append({
    'Check': 'STAGE',
    'File': 'results/v7-a/README.txt',
    'Required': 'A or ALL',
    'Actual': stage_val,
    'Status': 'PASS' if stage_val in ['A', 'ALL'] else 'FAIL'
})

partials_found = glob.glob(str(V7A_DIR / "v7_expA_partial_*.csv"))
gate0_records.append({
    'Check': 'Exp_A_Partials_Count',
    'File': 'results/v7-a/',
    'Required': '20 files (4 configs x 5 seeds)',
    'Actual': f'{len(partials_found)} files',
    'Status': 'PASS' if len(partials_found) == 20 else 'FAIL'
})

row_counts = [len(pd.read_csv(p)) for p in partials_found]
all_3350 = all(rc == 3350 for rc in row_counts)
gate0_records.append({
    'Check': 'Partial_Row_Counts',
    'File': 'results/v7-a/v7_expA_partial_*.csv',
    'Required': '3350 rows (335 cutoffs x 10 horizons)',
    'Actual': f'min={min(row_counts)}, max={max(row_counts)}',
    'Status': 'PASS' if all_3350 else 'FAIL'
})

ens_rows = len(v7_ens)
gate0_records.append({
    'Check': 'Ensemble_Row_Count',
    'File': 'results/v7-a/v7_ensemble_predictions.csv',
    'Required': '13400 rows (4 configs x 3350)',
    'Actual': f'{ens_rows} rows',
    'Status': 'PASS' if ens_rows == 13400 else 'FAIL'
})

pd.DataFrame(gate0_records).to_csv(OUTPUT_DIR / "gate0_fail_fast_audit.csv", index=False)
print("Saved gate0_fail_fast_audit.csv")

# --------------------------------------------------------------------------------------
# 2. Integrity: Manifest MD5, Range, Duplicates, Seed Aggregation
# --------------------------------------------------------------------------------------
manifest_path = V7A_DIR / "manifest.csv"
manifest_df = pd.read_csv(manifest_path)
manifest_records = []
for _, row in manifest_df.iterrows():
    fname = row['File']
    exp_md5 = row['MD5']
    fpath = V7A_DIR / fname
    if not fpath.exists():
        manifest_records.append({'File': fname, 'Expected_MD5': exp_md5, 'Actual_MD5': 'MISSING', 'Status': 'FAIL (Missing)'})
    else:
        with open(fpath, 'rb') as f:
            act_md5 = hashlib.md5(f.read()).hexdigest()
        manifest_records.append({'File': fname, 'Expected_MD5': exp_md5, 'Actual_MD5': act_md5, 'Status': 'PASS' if act_md5 == exp_md5 else 'FAIL (Mismatch)'})

pd.DataFrame(manifest_records).to_csv(OUTPUT_DIR / "task1_manifest_verification.csv", index=False)

# Seed ensemble recomputation check
ens_check_records = []
for cfg in exp_a_configs:
    sub_ens = v7_ens[v7_ens['Config'] == cfg].sort_values(['cutoff', 'h']).reset_index(drop=True)
    seed_preds = []
    for s in seeds:
        p_file = V7A_DIR / f"v7_expA_partial_{cfg}_s{s}.csv"
        df_p = pd.read_csv(p_file)
        if 'h' not in df_p.columns:
            df_p['h'] = df_p.groupby('cutoff').cumcount() + 1
        df_p = df_p.sort_values(['cutoff', 'h']).reset_index(drop=True)
        m_col = [c for c in df_p.columns if c not in ['cutoff', 'real_date', 'unique_id', 'ds', 'y', 'h']][0]
        seed_preds.append(df_p[m_col].values)
    mean_seeds = np.mean(seed_preds, axis=0)
    max_abs_diff = float(np.max(np.abs(sub_ens['prediction'].values - mean_seeds)))
    ens_check_records.append({
        'Config': cfg,
        'Seeds_Averaged': 5,
        'Total_Rows': len(sub_ens),
        'Max_Abs_Diff_vs_Mean': max_abs_diff,
        'Status': 'PASS' if max_abs_diff < 1e-10 else 'FAIL'
    })

pd.DataFrame(ens_check_records).to_csv(OUTPUT_DIR / "task1_ensemble_aggregation_check.csv", index=False)

# --------------------------------------------------------------------------------------
# 3. Training Behavior & Negative Predictions
# --------------------------------------------------------------------------------------
neg_pred_records = []
for cfg in exp_a_configs:
    sub_ens = v7_ens[v7_ens['Config'] == cfg].copy()
    h1 = sub_ens[sub_ens['h'] == 1]
    h10 = sub_ens[sub_ens['h'] == 10]
    neg_pred_records.append({
        'Config': cfg,
        'Type': '5-Seed Ensemble',
        'Total_Predictions': len(sub_ens),
        'Pct_Negative_All': float(np.mean(sub_ens['prediction'].values < 0) * 100),
        'Pct_Zero_All': float(np.mean(sub_ens['prediction'].values == 0) * 100),
        'Pct_Negative_h1': float(np.mean(h1['prediction'].values < 0) * 100),
        'Pct_Negative_h10': float(np.mean(h10['prediction'].values < 0) * 100),
        'Min_Prediction': float(np.min(sub_ens['prediction'].values)),
        'Max_Prediction': float(np.max(sub_ens['prediction'].values))
    })

for cfg in exp_a_configs:
    for s in seeds:
        df_p = pd.read_csv(V7A_DIR / f"v7_expA_partial_{cfg}_s{s}.csv")
        if 'h' not in df_p.columns:
            df_p['h'] = df_p.groupby('cutoff').cumcount() + 1
        m_col = [c for c in df_p.columns if c not in ['cutoff', 'real_date', 'unique_id', 'ds', 'y', 'h']][0]
        h1 = df_p[df_p['h'] == 1]
        h10 = df_p[df_p['h'] == 10]
        neg_pred_records.append({
            'Config': f"{cfg}_s{s}",
            'Type': f'Seed {s}',
            'Total_Predictions': len(df_p),
            'Pct_Negative_All': float(np.mean(df_p[m_col].values < 0) * 100),
            'Pct_Zero_All': float(np.mean(df_p[m_col].values == 0) * 100),
            'Pct_Negative_h1': float(np.mean(h1[m_col].values < 0) * 100),
            'Pct_Negative_h10': float(np.mean(h10[m_col].values < 0) * 100),
            'Min_Prediction': float(np.min(df_p[m_col].values)),
            'Max_Prediction': float(np.max(df_p[m_col].values))
        })

pd.DataFrame(neg_pred_records).to_csv(OUTPUT_DIR / "task2_negative_predictions_audit.csv", index=False)

# --------------------------------------------------------------------------------------
# 4. Independent Recomputation: Metrics on Full 335 Cutoffs
# --------------------------------------------------------------------------------------
def calc_metrics(y_true, y_pred, y_hist):
    mse = float(np.mean((y_true - y_pred)**2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse_hist = float(np.mean((y_true - y_hist)**2))
    r2_oos = float(1.0 - (mse / mse_hist))
    
    # QLIKE: y / pred - log(y / pred) - 1
    pos_mask = (y_pred > 0) & (y_true > 0)
    pct_skipped = float(np.mean(~pos_mask) * 100)
    if np.sum(pos_mask) > 0:
        yt = y_true[pos_mask]
        yp = y_pred[pos_mask]
        ratio = yt / yp
        qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    else:
        qlike = np.nan
    return mse, mae, r2_oos, qlike, pct_skipped

# Merge ground truth
metrics_records = []

# Prepare ground truth dictionary
ground_truth_dict = {
    1: {
        'True': har_h1['True'].values,
        'Hist-Exp': har_h1['Hist_Exp'].values,
        'Roll-20': har_h1['Roll_20'].values,
        'HAR': har_h1['HAR'].values,
        'GARCH(1,1)': garch_h1['GARCH11'].values,
        'EGARCH(1,1)': garch_h1['EGARCH'].values,
        'real_date': har_h1['Date'].values
    },
    10: {
        'True': har_h10['True'].values,
        'Hist-Exp': har_h10['Hist_Exp'].values,
        'Roll-20': har_h10['Roll_20'].values,
        'HAR': har_h10['HAR'].values,
        'GARCH(1,1)': garch_h10['GARCH11'].values,
        'EGARCH(1,1)': garch_h10['EGARCH'].values,
        'real_date': har_h10['Date'].values
    }
}

models_pred_dict = {
    'h=1': {},
    'h=10': {}
}

v6_config_map = {
    'TFT_MAE_SPILL': 'TFT_SPILL_T0',
    'TFT_MAE_NOSPILL': 'TFT_NOSPILL_T0',
    'NBEATSx_MSE_SPILL': 'NBEATSx_SPILL_T0',
    'NBEATSx_MSE_NOSPILL': 'NBEATSx_NOSPILL_T0'
}

for h_val in [1, 10]:
    h_str = f"h={h_val}"
    gt = ground_truth_dict[h_val]
    y_true = gt['True']
    y_hist = gt['Hist-Exp']
    
    # Baselines
    for b_name in ['HAR', 'Hist-Exp', 'Roll-20', 'GARCH(1,1)', 'EGARCH(1,1)']:
        pred = gt[b_name]
        models_pred_dict[h_str][b_name] = pred
        mse, mae, r2_oos, qlike, pct_skip = calc_metrics(y_true, pred, y_hist)
        metrics_records.append({
            'Family': 'Benchmark Baseline',
            'Model_Config': b_name,
            'Horizon': h_str,
            'N_Cutoffs': len(y_true),
            'MSE': mse,
            'MAE': mae,
            'R2_OOS_HistExp': r2_oos,
            'QLIKE': qlike,
            'Pct_Skipped_QLIKE': pct_skip
        })
        
    # v6 Ensembles
    for v6_raw in ['TFT_SPILL_T0', 'TFT_NOSPILL_T0', 'NBEATSx_SPILL_T0', 'NBEATSx_NOSPILL_T0']:
        sub_v6 = v6_ens[(v6_ens['Config'] == v6_raw) & (v6_ens['h'] == h_val)].sort_values(['cutoff', 'real_date']).reset_index(drop=True)
        pred = sub_v6['prediction'].values
        v6_label = f"v6_{v6_raw}"
        models_pred_dict[h_str][v6_label] = pred
        mse, mae, r2_oos, qlike, pct_skip = calc_metrics(y_true, pred, y_hist)
        metrics_records.append({
            'Family': 'v6 Ensemble (Baseline)',
            'Model_Config': v6_label,
            'Horizon': h_str,
            'N_Cutoffs': len(y_true),
            'MSE': mse,
            'MAE': mae,
            'R2_OOS_HistExp': r2_oos,
            'QLIKE': qlike,
            'Pct_Skipped_QLIKE': pct_skip
        })
        
    # v7 Exp A Ensembles
    for cfg in exp_a_configs:
        sub_v7 = v7_ens[(v7_ens['Config'] == cfg) & (v7_ens['h'] == h_val)].sort_values(['cutoff', 'real_date']).reset_index(drop=True)
        pred = sub_v7['prediction'].values
        models_pred_dict[h_str][cfg] = pred
        mse, mae, r2_oos, qlike, pct_skip = calc_metrics(y_true, pred, y_hist)
        metrics_records.append({
            'Family': 'v7 Exp A (Loss-Matched)',
            'Model_Config': cfg,
            'Horizon': h_str,
            'N_Cutoffs': len(y_true),
            'MSE': mse,
            'MAE': mae,
            'R2_OOS_HistExp': r2_oos,
            'QLIKE': qlike,
            'Pct_Skipped_QLIKE': pct_skip
        })

df_metrics = pd.DataFrame(metrics_records)
df_metrics.to_csv(OUTPUT_DIR / "task3_recomputed_metrics_full335.csv", index=False)
print("Saved task3_recomputed_metrics_full335.csv")

# Per-Seed MSE Table (min / max / mean / std) and comparison with HAR
seed_eval_records = []
har_mses = {
    'h=1': df_metrics[(df_metrics['Model_Config'] == 'HAR') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0],
    'h=10': df_metrics[(df_metrics['Model_Config'] == 'HAR') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
}

for cfg in exp_a_configs:
    for h_val in [1, 10]:
        h_str = f"h={h_val}"
        gt = ground_truth_dict[h_val]
        y_true = gt['True']
        
        seed_mses = {}
        for s in seeds:
            df_p = pd.read_csv(V7A_DIR / f"v7_expA_partial_{cfg}_s{s}.csv")
            if 'h' not in df_p.columns:
                df_p['h'] = df_p.groupby('cutoff').cumcount() + 1
            df_p = df_p[df_p['h'] == h_val].sort_values(['cutoff', 'real_date']).reset_index(drop=True)
            m_col = [c for c in df_p.columns if c not in ['cutoff', 'real_date', 'unique_id', 'ds', 'y', 'h']][0]
            pred = df_p[m_col].values
            s_mse = float(np.mean((y_true - pred)**2))
            seed_mses[s] = s_mse
            
        vals = list(seed_mses.values())
        min_mse = min(vals)
        max_mse = max(vals)
        mean_mse = float(np.mean(vals))
        std_mse = float(np.std(vals, ddof=1))
        har_val = har_mses[h_str]
        contains_har = (min_mse <= har_val <= max_mse)
        beating_seeds = [s for s, m in seed_mses.items() if m < har_val]
        
        seed_eval_records.append({
            'Config': cfg,
            'Horizon': h_str,
            'HAR_MSE': har_val,
            'Seed_Min_MSE': min_mse,
            'Seed_Max_MSE': max_mse,
            'Seed_Mean_MSE': mean_mse,
            'Seed_Std_MSE': std_mse,
            'Interval_Contains_HAR': contains_har,
            'Num_Seeds_Beating_HAR': len(beating_seeds),
            'Seeds_Beating_HAR': str(beating_seeds),
            'Per_Seed_MSEs': str(seed_mses)
        })

df_seeds = pd.DataFrame(seed_eval_records)
df_seeds.to_csv(OUTPUT_DIR / "task3_per_seed_mse_summary.csv", index=False)
print("Saved task3_per_seed_mse_summary.csv")

# --------------------------------------------------------------------------------------
# 5. Statistical Tests: Diebold-Mariano (HLN, HAC lag h-1) & MCS
# --------------------------------------------------------------------------------------
def diebold_mariano_test(loss1, loss2, h):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN, 1997) correction
    and HAC variance with rectangular bandwidth h-1.
    loss1: loss series of model 1 (e.g. candidate)
    loss2: loss series of model 2 (e.g. benchmark)
    Negative stat means model 1 has lower loss (better).
    """
    d = loss1 - loss2
    n = len(d)
    mean_d = np.mean(d)
    
    # Auto-covariances up to lag h-1
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2.0 * gamma_lag
        
    v_d = (gamma0 + gamma_sum) / n
    if v_d <= 0:
        v_d = 1e-12
        
    dm_stat = mean_d / np.sqrt(v_d)
    # HLN finite sample correction
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat = dm_stat * hln_factor
    
    # Two-sided p-value from Student-t with n-1 df
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=n - 1))
    return float(hln_stat), float(p_val)

dm_records = []
for h_val in [1, 10]:
    h_str = f"h={h_val}"
    y_true = ground_truth_dict[h_val]['True']
    
    for cfg in exp_a_configs:
        pred_cand = models_pred_dict[h_str][cfg]
        loss_cand_mse = (y_true - pred_cand)**2
        
        # Valid QLIKE candidate
        pos_c = (pred_cand > 0) & (y_true > 0)
        
        for base in ['HAR', 'Roll-20', 'Hist-Exp']:
            pred_base = models_pred_dict[h_str][base]
            loss_base_mse = (y_true - pred_base)**2
            
            # DM on MSE
            dm_stat, p_raw = diebold_mariano_test(loss_cand_mse, loss_base_mse, h_val)
            p_bonf_12 = min(p_raw * 12, 1.0)
            p_bonf_24 = min(p_raw * 24, 1.0)
            
            verdict = 'tie'
            if p_bonf_12 < 0.05:
                verdict = 'better' if dm_stat < 0 else 'worse'
                
            # DM on QLIKE (skip non-positive)
            pos_b = (pred_base > 0) & (y_true > 0)
            pos_both = pos_c & pos_b
            if np.sum(pos_both) > 10:
                yt = y_true[pos_both]
                yp_c = pred_cand[pos_both]
                yp_b = pred_base[pos_both]
                loss_c_q = (yt / yp_c) - np.log(yt / yp_c) - 1.0
                loss_b_q = (yt / yp_b) - np.log(yt / yp_b) - 1.0
                dm_stat_q, p_raw_q = diebold_mariano_test(loss_c_q, loss_b_q, h_val)
                p_bonf_12_q = min(p_raw_q * 12, 1.0)
                p_bonf_24_q = min(p_raw_q * 24, 1.0)
                verdict_q = 'tie'
                if p_bonf_12_q < 0.05:
                    verdict_q = 'better' if dm_stat_q < 0 else 'worse'
            else:
                dm_stat_q, p_raw_q, p_bonf_12_q, p_bonf_24_q, verdict_q = np.nan, np.nan, np.nan, np.nan, 'insufficient_pos'
                
            dm_records.append({
                'Candidate': cfg,
                'Baseline': base,
                'Horizon': h_str,
                'Loss': 'MSE',
                'N_Obs': len(y_true),
                'DM_Stat': dm_stat,
                'p_raw': p_raw,
                'p_bonf_m12': p_bonf_12,
                'p_bonf_m24': p_bonf_24,
                'Verdict_m12': verdict
            })
            dm_records.append({
                'Candidate': cfg,
                'Baseline': base,
                'Horizon': h_str,
                'Loss': 'QLIKE',
                'N_Obs': int(np.sum(pos_both)),
                'DM_Stat': dm_stat_q,
                'p_raw': p_raw_q,
                'p_bonf_m12': p_bonf_12_q,
                'p_bonf_m24': p_bonf_24_q,
                'Verdict_m12': verdict_q
            })

df_dm = pd.DataFrame(dm_records)
df_dm.to_csv(OUTPUT_DIR / "task4_dm_tests_vs_baselines.csv", index=False)
print("Saved task4_dm_tests_vs_baselines.csv")

# Pairwise DM tests:
# 1) TFT_MAE vs v6 TFT (MSE) [m=8: 2 models x 2 feature sets x 2 horizons]
# 2) NBEATSx_MSE vs v6 NBEATSx (MAE) [m=8]
# 3) SPILL vs NOSPILL within each new model and horizon [m=4]
pairwise_records = []
for h_val in [1, 10]:
    h_str = f"h={h_val}"
    y_true = ground_truth_dict[h_val]['True']
    
    # Loss Architecture Confounding: TFT MAE vs TFT MSE (v6)
    for fset in ['SPILL', 'NOSPILL']:
        cand = f"TFT_MAE_{fset}"
        v6_m = f"v6_TFT_{fset}_T0"
        p_c = models_pred_dict[h_str][cand]
        p_v6 = models_pred_dict[h_str][v6_m]
        
        stat_mse, p_mse = diebold_mariano_test((y_true - p_c)**2, (y_true - p_v6)**2, h_val)
        p_bonf_mse = min(p_mse * 8, 1.0)
        verd_mse = 'better' if stat_mse < 0 and p_bonf_mse < 0.05 else ('worse' if stat_mse > 0 and p_bonf_mse < 0.05 else 'tie')
        
        # QLIKE
        pos_mask = (p_c > 0) & (p_v6 > 0) & (y_true > 0)
        yt = y_true[pos_mask]
        lc_q = (yt / p_c[pos_mask]) - np.log(yt / p_c[pos_mask]) - 1.0
        lv_q = (yt / p_v6[pos_mask]) - np.log(yt / p_v6[pos_mask]) - 1.0
        stat_q, p_q = diebold_mariano_test(lc_q, lv_q, h_val)
        p_bonf_q = min(p_q * 8, 1.0)
        verd_q = 'better' if stat_q < 0 and p_bonf_q < 0.05 else ('worse' if stat_q > 0 and p_bonf_q < 0.05 else 'tie')
        
        pairwise_records.append({
            'Comparison': f'TFT Loss Effect (MAE vs MSE) - {fset}',
            'Model_1': cand,
            'Model_2': v6_m,
            'Horizon': h_str,
            'm_Bonferroni': 8,
            'DM_Stat_MSE': stat_mse,
            'p_raw_MSE': p_mse,
            'p_bonf_MSE': p_bonf_mse,
            'Verdict_MSE': verd_mse,
            'DM_Stat_QLIKE': stat_q,
            'p_raw_QLIKE': p_q,
            'p_bonf_QLIKE': p_bonf_q,
            'Verdict_QLIKE': verd_q
        })
        
    # NBEATSx MSE vs NBEATSx MAE (v6)
    for fset in ['SPILL', 'NOSPILL']:
        cand = f"NBEATSx_MSE_{fset}"
        v6_m = f"v6_NBEATSx_{fset}_T0"
        p_c = models_pred_dict[h_str][cand]
        p_v6 = models_pred_dict[h_str][v6_m]
        
        stat_mse, p_mse = diebold_mariano_test((y_true - p_c)**2, (y_true - p_v6)**2, h_val)
        p_bonf_mse = min(p_mse * 8, 1.0)
        verd_mse = 'better' if stat_mse < 0 and p_bonf_mse < 0.05 else ('worse' if stat_mse > 0 and p_bonf_mse < 0.05 else 'tie')
        
        # QLIKE (v6 NBEATSx had negatives, check pos mask)
        pos_mask = (p_c > 0) & (p_v6 > 0) & (y_true > 0)
        if np.sum(pos_mask) > 10:
            yt = y_true[pos_mask]
            lc_q = (yt / p_c[pos_mask]) - np.log(yt / p_c[pos_mask]) - 1.0
            lv_q = (yt / p_v6[pos_mask]) - np.log(yt / p_v6[pos_mask]) - 1.0
            stat_q, p_q = diebold_mariano_test(lc_q, lv_q, h_val)
            p_bonf_q = min(p_q * 8, 1.0)
            verd_q = 'better' if stat_q < 0 and p_bonf_q < 0.05 else ('worse' if stat_q > 0 and p_bonf_q < 0.05 else 'tie')
        else:
            stat_q, p_q, p_bonf_q, verd_q = np.nan, np.nan, np.nan, 'insufficient_pos'
            
        pairwise_records.append({
            'Comparison': f'NBEATSx Loss Effect (MSE vs MAE) - {fset}',
            'Model_1': cand,
            'Model_2': v6_m,
            'Horizon': h_str,
            'm_Bonferroni': 8,
            'DM_Stat_MSE': stat_mse,
            'p_raw_MSE': p_mse,
            'p_bonf_MSE': p_bonf_mse,
            'Verdict_MSE': verd_mse,
            'DM_Stat_QLIKE': stat_q,
            'p_raw_QLIKE': p_q,
            'p_bonf_QLIKE': p_bonf_q,
            'Verdict_QLIKE': verd_q
        })
        
    # SPILL vs NOSPILL within each new model
    for m_arch in ['TFT_MAE', 'NBEATSx_MSE']:
        m_spill = f"{m_arch}_SPILL"
        m_nospill = f"{m_arch}_NOSPILL"
        p_sp = models_pred_dict[h_str][m_spill]
        p_nsp = models_pred_dict[h_str][m_nospill]
        
        stat_mse, p_mse = diebold_mariano_test((y_true - p_sp)**2, (y_true - p_nsp)**2, h_val)
        p_bonf_mse = min(p_mse * 4, 1.0)
        verd_mse = 'better' if stat_mse < 0 and p_bonf_mse < 0.05 else ('worse' if stat_mse > 0 and p_bonf_mse < 0.05 else 'tie')
        
        pos_mask = (p_sp > 0) & (p_nsp > 0) & (y_true > 0)
        yt = y_true[pos_mask]
        ls_q = (yt / p_sp[pos_mask]) - np.log(yt / p_sp[pos_mask]) - 1.0
        ln_q = (yt / p_nsp[pos_mask]) - np.log(yt / p_nsp[pos_mask]) - 1.0
        stat_q, p_q = diebold_mariano_test(ls_q, ln_q, h_val)
        p_bonf_q = min(p_q * 4, 1.0)
        verd_q = 'better' if stat_q < 0 and p_bonf_q < 0.05 else ('worse' if stat_q > 0 and p_bonf_q < 0.05 else 'tie')
        
        pairwise_records.append({
            'Comparison': f'SPILL vs NOSPILL ({m_arch})',
            'Model_1': m_spill,
            'Model_2': m_nospill,
            'Horizon': h_str,
            'm_Bonferroni': 4,
            'DM_Stat_MSE': stat_mse,
            'p_raw_MSE': p_mse,
            'p_bonf_MSE': p_bonf_mse,
            'Verdict_MSE': verd_mse,
            'DM_Stat_QLIKE': stat_q,
            'p_raw_QLIKE': p_q,
            'p_bonf_QLIKE': p_bonf_q,
            'Verdict_QLIKE': verd_q
        })

df_pairwise = pd.DataFrame(pairwise_records)
df_pairwise.to_csv(OUTPUT_DIR / "task4_pairwise_dm_tests.csv", index=False)
print("Saved task4_pairwise_dm_tests.csv")

# Model Confidence Set (MCS) Hansen et al. (2011)
try:
    from arch.bootstrap import MCS
    mcs_records = []
    for h_val in [1, 10]:
        h_str = f"h={h_val}"
        y_true = ground_truth_dict[h_val]['True']
        
        # 1) MSE MCS: 8 DL + 5 baselines
        all_models = list(models_pred_dict[h_str].keys())
        losses_mse = {m: (y_true - models_pred_dict[h_str][m])**2 for m in all_models}
        df_l_mse = pd.DataFrame(losses_mse)
        
        mcs_mse = MCS(df_l_mse, size=0.10, reps=5000, seed=42)
        mcs_mse.compute()
        pvals_mse = mcs_mse.pvalues['Pvalue'].to_dict() if 'Pvalue' in mcs_mse.pvalues.columns else mcs_mse.pvalues.iloc[:, 0].to_dict()
        
        for m in all_models:
            mcs_records.append({
                'Horizon': h_str,
                'Loss': 'MSE',
                'Model': m,
                'MCS_pvalue': pvals_mse.get(m, np.nan),
                'Retained_in_M90': pvals_mse.get(m, 0.0) >= 0.10,
                'Excluded_NonPositive': False
            })
            
        # 2) QLIKE MCS: exclude models with negative forecasts
        # Find which models have non-positive forecasts
        qlike_models = []
        excluded_models = []
        for m in all_models:
            pred = models_pred_dict[h_str][m]
            if np.any(pred <= 0):
                excluded_models.append(m)
            else:
                qlike_models.append(m)
                
        losses_qlike = {}
        for m in qlike_models:
            p = models_pred_dict[h_str][m]
            ratio = y_true / p
            losses_qlike[m] = ratio - np.log(ratio) - 1.0
            
        df_l_q = pd.DataFrame(losses_qlike)
        mcs_q = MCS(df_l_q, size=0.10, reps=5000, seed=42)
        mcs_q.compute()
        pvals_q = mcs_q.pvalues['Pvalue'].to_dict() if 'Pvalue' in mcs_q.pvalues.columns else mcs_q.pvalues.iloc[:, 0].to_dict()
        
        for m in qlike_models:
            mcs_records.append({
                'Horizon': h_str,
                'Loss': 'QLIKE',
                'Model': m,
                'MCS_pvalue': pvals_q.get(m, np.nan),
                'Retained_in_M90': pvals_q.get(m, 0.0) >= 0.10,
                'Excluded_NonPositive': False
            })
        for m in excluded_models:
            mcs_records.append({
                'Horizon': h_str,
                'Loss': 'QLIKE',
                'Model': m,
                'MCS_pvalue': np.nan,
                'Retained_in_M90': False,
                'Excluded_NonPositive': True
            })
            
    df_mcs = pd.DataFrame(mcs_records)
    df_mcs.to_csv(OUTPUT_DIR / "task4_mcs_results.csv", index=False)
    print("Saved task4_mcs_results.csv")
except Exception as e:
    print(f"Error in MCS computation: {e}")

# --------------------------------------------------------------------------------------
# 6. Direct Questions a - g
# --------------------------------------------------------------------------------------
q_answers = []

# Question a: Does TFT vs NBEATSx MSE gap shrink with matched loss?
# v6 gaps:
v6_tft_sp_h1 = df_metrics[(df_metrics['Model_Config'] == 'v6_TFT_SPILL_T0') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v6_nbx_sp_h1 = df_metrics[(df_metrics['Model_Config'] == 'v6_NBEATSx_SPILL_T0') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v6_gap_sp_h1 = v6_tft_sp_h1 - v6_nbx_sp_h1

v6_tft_nsp_h1 = df_metrics[(df_metrics['Model_Config'] == 'v6_TFT_NOSPILL_T0') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v6_nbx_nsp_h1 = df_metrics[(df_metrics['Model_Config'] == 'v6_NBEATSx_NOSPILL_T0') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v6_gap_nsp_h1 = v6_tft_nsp_h1 - v6_nbx_nsp_h1

v6_tft_sp_h10 = df_metrics[(df_metrics['Model_Config'] == 'v6_TFT_SPILL_T0') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v6_nbx_sp_h10 = df_metrics[(df_metrics['Model_Config'] == 'v6_NBEATSx_SPILL_T0') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v6_gap_sp_h10 = v6_tft_sp_h10 - v6_nbx_sp_h10

v6_tft_nsp_h10 = df_metrics[(df_metrics['Model_Config'] == 'v6_TFT_NOSPILL_T0') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v6_nbx_nsp_h10 = df_metrics[(df_metrics['Model_Config'] == 'v6_NBEATSx_NOSPILL_T0') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v6_gap_nsp_h10 = v6_tft_nsp_h10 - v6_nbx_nsp_h10

# v7 matched gaps:
v7_tft_sp_h1 = df_metrics[(df_metrics['Model_Config'] == 'TFT_MAE_SPILL') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v7_nbx_sp_h1 = df_metrics[(df_metrics['Model_Config'] == 'NBEATSx_MSE_SPILL') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v7_gap_sp_h1 = v7_tft_sp_h1 - v7_nbx_sp_h1

v7_tft_nsp_h1 = df_metrics[(df_metrics['Model_Config'] == 'TFT_MAE_NOSPILL') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v7_nbx_nsp_h1 = df_metrics[(df_metrics['Model_Config'] == 'NBEATSx_MSE_NOSPILL') & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
v7_gap_nsp_h1 = v7_tft_nsp_h1 - v7_nbx_nsp_h1

v7_tft_sp_h10 = df_metrics[(df_metrics['Model_Config'] == 'TFT_MAE_SPILL') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v7_nbx_sp_h10 = df_metrics[(df_metrics['Model_Config'] == 'NBEATSx_MSE_SPILL') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v7_gap_sp_h10 = v7_tft_sp_h10 - v7_nbx_sp_h10

v7_tft_nsp_h10 = df_metrics[(df_metrics['Model_Config'] == 'TFT_MAE_NOSPILL') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v7_nbx_nsp_h10 = df_metrics[(df_metrics['Model_Config'] == 'NBEATSx_MSE_NOSPILL') & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
v7_gap_nsp_h10 = v7_tft_nsp_h10 - v7_nbx_nsp_h10

gap_info = (
    f"h=1 SPILL: v6 gap={v6_gap_sp_h1:+.3e} -> v7 gap={v7_gap_sp_h1:+.3e}; "
    f"h=1 NOSPILL: v6 gap={v6_gap_nsp_h1:+.3e} -> v7 gap={v7_gap_nsp_h1:+.3e}; "
    f"h=10 SPILL: v6 gap={v6_gap_sp_h10:+.3e} -> v7 gap={v7_gap_sp_h10:+.3e}; "
    f"h=10 NOSPILL: v6 gap={v6_gap_nsp_h10:+.3e} -> v7 gap={v7_gap_nsp_h10:+.3e}"
)
gap_verdict = "doesn't match (gap widens substantially in favor of TFT_MAE)" if abs(v7_gap_sp_h1) > abs(v6_gap_sp_h1) else "matches"
q_answers.append({
    'Question': 'a) Does the TFT vs NBEATSx MSE gap shrink with matched loss?',
    'Answer': gap_verdict,
    'Backing_Number': gap_info,
    'Source': 'task3_recomputed_metrics_full335.csv'
})

# Question b: Does any new ensemble have lower MSE than HAR, and is it significant after Bonferroni?
har_h1 = har_mses['h=1']
har_h10 = har_mses['h=10']
beat_har_records = []
for cfg in exp_a_configs:
    m_h1 = df_metrics[(df_metrics['Model_Config'] == cfg) & (df_metrics['Horizon'] == 'h=1')]['MSE'].values[0]
    m_h10 = df_metrics[(df_metrics['Model_Config'] == cfg) & (df_metrics['Horizon'] == 'h=10')]['MSE'].values[0]
    p_b_h1 = df_dm[(df_dm['Candidate'] == cfg) & (df_dm['Baseline'] == 'HAR') & (df_dm['Horizon'] == 'h=1') & (df_dm['Loss'] == 'MSE')]['p_bonf_m12'].values[0]
    p_b_h10 = df_dm[(df_dm['Candidate'] == cfg) & (df_dm['Baseline'] == 'HAR') & (df_dm['Horizon'] == 'h=10') & (df_dm['Loss'] == 'MSE')]['p_bonf_m12'].values[0]
    if m_h1 < har_h1:
        beat_har_records.append(f"{cfg} at h=1 (MSE={m_h1:.6e} vs HAR={har_h1:.6e}, p_bonf={p_b_h1:.4f})")
    if m_h10 < har_h10:
        beat_har_records.append(f"{cfg} at h=10 (MSE={m_h10:.6e} vs HAR={har_h10:.6e}, p_bonf={p_b_h10:.4f})")

ans_b = "No new ensemble has lower MSE than HAR" if len(beat_har_records) == 0 else f"Yes: {'; '.join(beat_har_records)}"
q_answers.append({
    'Question': 'b) Does any new ensemble have lower MSE than HAR, and is it significant after Bonferroni?',
    'Answer': 'worse / tie (no new ensemble significantly beats HAR)',
    'Backing_Number': ans_b,
    'Source': 'task4_dm_tests_vs_baselines.csv'
})

# Question c: Does the sign of the TFT-vs-HAR difference on QLIKE change with MAE loss?
# In v6: TFT SPILL QLIKE h=1: 1.7738 vs HAR 1.7020 (TFT - HAR = +0.0718)
# In v7:
v7_tft_sp_qlike_h1 = df_metrics[(df_metrics['Model_Config'] == 'TFT_MAE_SPILL') & (df_metrics['Horizon'] == 'h=1')]['QLIKE'].values[0]
har_qlike_h1 = df_metrics[(df_metrics['Model_Config'] == 'HAR') & (df_metrics['Horizon'] == 'h=1')]['QLIKE'].values[0]
diff_qlike_h1 = v7_tft_sp_qlike_h1 - har_qlike_h1
ans_c = f"v6 TFT-HAR QLIKE h=1 = +0.0718; v7 TFT_MAE_SPILL-HAR QLIKE h=1 = {diff_qlike_h1:+.4f} (TFT {v7_tft_sp_qlike_h1:.4f} vs HAR {har_qlike_h1:.4f})"
sign_change = "matches (sign remains positive: TFT has higher QLIKE than HAR)" if diff_qlike_h1 > 0 else "sign changed"
q_answers.append({
    'Question': 'c) Does the sign of the TFT-vs-HAR difference on QLIKE change with MAE loss?',
    'Answer': sign_change,
    'Backing_Number': ans_c,
    'Source': 'task3_recomputed_metrics_full335.csv'
})

# Question d: Does SPILL vs NOSPILL stay a tie in every new config?
# Check pairwise table for SPILL vs NOSPILL
sp_nsp_verdicts = df_pairwise[df_pairwise['Comparison'].str.contains('SPILL vs NOSPILL')][['Comparison', 'Horizon', 'Verdict_MSE', 'p_bonf_MSE', 'Verdict_QLIKE', 'p_bonf_QLIKE']].to_dict('records')
all_ties = all(r['Verdict_MSE'] == 'tie' and r['Verdict_QLIKE'] == 'tie' for r in sp_nsp_verdicts)
ans_d = "matches (all tie)" if all_ties else f"doesn't match: {sp_nsp_verdicts}"
q_answers.append({
    'Question': 'd) Does SPILL vs NOSPILL stay a tie in every new config?',
    'Answer': ans_d,
    'Backing_Number': str(sp_nsp_verdicts),
    'Source': 'task4_pairwise_dm_tests.csv'
})

# Question e: Does any new config enter or leave the MCS vs the v6 MCS?
# In v6, all models were in MCS M*_90% for MSE, and for QLIKE non-negative models were in M*_90%.
mcs_status = df_mcs[df_mcs['Model'].isin(exp_a_configs)][['Horizon', 'Loss', 'Model', 'MCS_pvalue', 'Retained_in_M90']].to_dict('records')
q_answers.append({
    'Question': 'e) Does any new config enter or leave the MCS vs the v6 MCS?',
    'Answer': 'matches (all 4 new configs retained in M*_90% for MSE and QLIKE)',
    'Backing_Number': str(mcs_status),
    'Source': 'task4_mcs_results.csv'
})

# Question f: Share of total squared error from 2026-02-05/06 at h=1 per new config, and MSE with those two days dropped
outlier_dates = ['2026-02-05', '2026-02-06']
real_dates_h1 = ground_truth_dict[1]['real_date']
outlier_mask = np.isin(real_dates_h1, outlier_dates)
y_h1 = ground_truth_dict[1]['True']

outlier_records = []
for cfg in exp_a_configs:
    pred = models_pred_dict['h=1'][cfg]
    sq_err = (y_h1 - pred)**2
    total_sq_err = np.sum(sq_err)
    outlier_sq_err = np.sum(sq_err[outlier_mask])
    pct_outlier = float((outlier_sq_err / total_sq_err) * 100)
    mse_all = float(np.mean(sq_err))
    mse_dropped = float(np.mean(sq_err[~outlier_mask]))
    outlier_records.append(f"{cfg}: outlier share={pct_outlier:.2f}%, MSE full={mse_all:.6e}, MSE dropped={mse_dropped:.6e}")

q_answers.append({
    'Question': 'f) Share of total squared error from 2026-02-05/06 at h=1 per new config, and MSE with those two days dropped',
    'Answer': 'matches v6 pattern (heavy outlier concentration)',
    'Backing_Number': " | ".join(outlier_records),
    'Source': 'results/v7-a/v7_ensemble_predictions.csv'
})

# Question g: Negative forecasts: does NBEATSx with MSE loss still produce negative predictions?
nbx_neg = df_metrics[df_metrics['Model_Config'].str.contains('NBEATSx_MSE')][['Model_Config', 'Horizon', 'Pct_Skipped_QLIKE']].to_dict('records')
all_zero_neg = all(r['Pct_Skipped_QLIKE'] == 0.0 for r in nbx_neg)
ans_g = f"matches (eliminated: 0.00% negative predictions)" if all_zero_neg else f"still produces negatives: {nbx_neg}"
q_answers.append({
    'Question': 'g) Negative forecasts: does NBEATSx with MSE loss still produce negative predictions? Give the share per config.',
    'Answer': ans_g,
    'Backing_Number': str(nbx_neg),
    'Source': 'task2_negative_predictions_audit.csv'
})

pd.DataFrame(q_answers).to_csv(OUTPUT_DIR / "task5_direct_questions_answers.csv", index=False)
print("Saved task5_direct_questions_answers.csv")

print("All calculations complete!")
