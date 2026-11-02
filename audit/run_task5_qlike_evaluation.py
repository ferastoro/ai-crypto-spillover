#!/usr/bin/env python3
"""
Task 5: QLIKE Evaluation + DM test vs HAR + Kaggle Retraining Script
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
RESULTS_V6 = BASE_DIR / "results" / "v6"
AUDIT_OUT = BASE_DIR / "audit_outputs"
AUDIT_OUT.mkdir(exist_ok=True, parents=True)

def compute_qlike(y_true, y_pred, eps=1e-6):
    """
    QLIKE loss: y / y_hat - log(y / y_hat) - 1
    Requires y_hat > 0.
    """
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    # Clip predictions to eps to avoid division by zero or log of negative
    yp_clipped = np.maximum(yp, eps)
    # Avoid log(0) if y_true is 0 by small threshold
    yt_safe = np.maximum(yt, 1e-12)
    ratio = yt_safe / yp_clipped
    qlike_vals = ratio - np.log(ratio) - 1.0
    return qlike_vals

def dm_test_qlike(y_true, y_pred_m, y_pred_b, h, eps=1e-6, use_hln=True, m_family=12):
    """
    Diebold-Mariano test under QLIKE loss differential.
    """
    loss_m = compute_qlike(y_true, y_pred_m, eps=eps)
    loss_b = compute_qlike(y_true, y_pred_b, eps=eps)
    d = loss_m - loss_b
    T = len(d)
    md = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    gamma = [gamma0]
    lag = 0 if h == 1 else (h - 1)
    
    for l in range(1, lag + 1):
        gamma.append(float(np.cov(d[l:], d[:-l])[0, 1]))
        
    var_d = gamma[0] + sum([2.0 * (1.0 - l / (lag + 1)) * gamma[l] for l in range(1, lag + 1)])
    if var_d <= 0:
        return {'md': md, 'stat': 0.0, 'p_raw': 1.0, 'p_bonf': 1.0, 'var_d': var_d, 'lag': lag, 'hln_applied': False}
        
    stat = md / np.sqrt(var_d / T)
    hln_applied = False
    if h > 1 and use_hln:
        hln = np.sqrt((T + 1 - 2 * h + (h / T) * (h - 1)) / T)
        stat_adj = stat * hln
        hln_applied = True
    else:
        stat_adj = stat
        
    p_raw = float(2.0 * (1.0 - norm.cdf(abs(stat_adj))))
    p_bonf = min(1.0, p_raw * m_family)
    return {
        'md': md,
        'stat': stat_adj,
        'p_raw': p_raw,
        'p_bonf': p_bonf,
        'var_d': var_d,
        'lag': lag,
        'hln_applied': hln_applied
    }

def main():
    print("Running Task 5: QLIKE Evaluation...")
    # 1. Load data
    har_h1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values('Date').reset_index(drop=True)
    har_h10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values('Date').reset_index(drop=True)
    garch_df = pd.read_csv(INPUTS_V5 / "GARCH_Predictions_V4.csv")
    ens_df = pd.read_csv(RESULTS_V6 / "v6_ensemble_predictions.csv")
    
    # Extract GARCH
    garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
    garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
    garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)
    
    # Helper to get ensemble predictions
    def get_ens(cfg, h):
        sub = ens_df[ens_df['Config'] == cfg].copy()
        sub['h'] = sub.groupby('cutoff').cumcount() + 1
        return sub[sub['h'] == h].sort_values('real_date').reset_index(drop=True)['prediction'].values
        
    for h_val, har_file, garch_sub in [(1, har_h1, garch_h1), (10, har_h10, garch_h10)]:
        yt = har_file['True'].values
        y_har = har_file['HAR'].values
        
        models = {
            'HAR': y_har,
            'Hist-Exp': har_file['Hist_Exp'].values,
            'Roll-20': har_file['Roll_20'].values,
            'GARCH(1,1)': garch_sub['GARCH11'].values,
            'EGARCH(1,1)': garch_sub['EGARCH'].values,
            'TFT-SPILL': get_ens('TFT_SPILL_T0', h_val),
            'TFT-NOSPILL': get_ens('TFT_NOSPILL_T0', h_val),
            'NBEATSx-SPILL': get_ens('NBEATSx_SPILL_T0', h_val),
            'NBEATSx-NOSPILL': get_ens('NBEATSx_NOSPILL_T0', h_val)
        }
        
        records = []
        for name, pred in models.items():
            neg_count = int(np.sum(pred <= 0))
            neg_pct = float(neg_count / len(pred) * 100.0)
            
            # Clipped version (pred >= 1e-6)
            qlike_clip = compute_qlike(yt, pred, eps=1e-6)
            mean_qlike_clip = float(np.mean(qlike_clip))
            
            # DM test vs HAR (clipped)
            if name != 'HAR':
                dm_res = dm_test_qlike(yt, pred, y_har, h=h_val, eps=1e-6, m_family=12)
                dm_stat = round(dm_res['stat'], 4)
                p_raw = round(dm_res['p_raw'], 4)
                p_bonf = round(dm_res['p_bonf'], 4)
                winner = 'Model' if dm_res['stat'] < -1.96 else ('HAR' if dm_res['stat'] > 1.96 else 'Tie')
            else:
                dm_stat = 0.0
                p_raw = 1.0
                p_bonf = 1.0
                winner = 'Benchmark'
                
            # Excluded version (only where pred > 0)
            valid_mask = pred > 0
            if valid_mask.sum() > 0:
                qlike_excl = compute_qlike(yt[valid_mask], pred[valid_mask], eps=1e-6)
                mean_qlike_excl = float(np.mean(qlike_excl))
            else:
                mean_qlike_excl = np.nan
                
            records.append({
                'Horizon': f'h={h_val}',
                'Model': name,
                'Non_Positive_Count': neg_count,
                'Non_Positive_Pct': f"{neg_pct:.2f}%",
                'QLIKE_Clipped_1e-6': round(mean_qlike_clip, 6),
                'QLIKE_Excluded_Strict': round(mean_qlike_excl, 6) if not np.isnan(mean_qlike_excl) else 'N/A',
                'DM_QLIKE_Stat_vs_HAR': dm_stat,
                'DM_p_raw': p_raw,
                'DM_p_bonf': p_bonf,
                'Winner_vs_HAR': winner
            })
            
        df_h = pd.DataFrame(records)
        out_name = AUDIT_OUT / f"task5_qlike_evaluation_h{h_val}.csv"
        df_h.to_csv(out_name, index=False)
        print(f"\nResults for h={h_val}:")
        print(df_h.to_string(index=False))
        
    # Combine both horizons into one master CSV
    df_h1 = pd.read_csv(AUDIT_OUT / "task5_qlike_evaluation_h1.csv")
    df_h10 = pd.read_csv(AUDIT_OUT / "task5_qlike_evaluation_h10.csv")
    df_master = pd.concat([df_h1, df_h10], ignore_index=True)
    df_master.to_csv(AUDIT_OUT / "task5_qlike_evaluation.csv", index=False)
    print(f"\nSaved combined QLIKE evaluation to {AUDIT_OUT / 'task5_qlike_evaluation.csv'}")

if __name__ == "__main__":
    main()

