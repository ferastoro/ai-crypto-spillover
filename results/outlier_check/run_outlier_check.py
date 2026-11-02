#!/usr/bin/env python3
"""
results/outlier_check/run_outlier_check.py
Outlier sensitivity analysis pipeline for RSIC v6 Deep Learning experiments.
Evaluates model ranking and DM test robustness against extreme return squared days.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_DIR = os.path.join(ROOT_DIR, "inputs/v5")
V6_OUT_DIR = os.path.join(ROOT_DIR, "results/v6")
AUDIT_V6_DIR = os.path.join(ROOT_DIR, "audit/v6")
OUT_DIR = os.path.join(ROOT_DIR, "results/outlier_check")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(SRC_DIR)
sys.path.append(INPUT_DIR)
import v5_helpers

def md5_file(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("================================================================================")
    print("RSIC v6 OUTLIER SENSITIVITY CHECK")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # INPUT FILES AND HASHES
    # --------------------------------------------------------------------------
    input_files = {
        'v6_ensemble_predictions.csv': os.path.join(V6_OUT_DIR, "v6_ensemble_predictions.csv"),
        'CV_Predictions_V4.csv': os.path.join(INPUT_DIR, "CV_Predictions_V4.csv"),
        'dataset_deep_learning.csv': os.path.join(INPUT_DIR, "dataset_deep_learning.csv"),
        'har_preds_T0_h1_v2.csv': os.path.join(INPUT_DIR, "har_preds_T0_h1_v2.csv"),
        'har_preds_T0_h10_v2.csv': os.path.join(INPUT_DIR, "har_preds_T0_h10_v2.csv"),
        'GARCH_Predictions_V4.csv': os.path.join(INPUT_DIR, "GARCH_Predictions_V4.csv"),
        'v5_helpers.py': os.path.join(INPUT_DIR, "v5_helpers.py"),
        'audit_v6_dm_ensemble.csv': os.path.join(AUDIT_V6_DIR, "audit_v6_dm_ensemble.csv"),
    }

    print("\n[INPUT FILES AND MD5 HASHES]")
    for name, path in input_files.items():
        h = md5_file(path)
        sz = os.path.getsize(path)
        rel_path = os.path.relpath(path, ROOT_DIR)
        print(f"  {rel_path}: md5={h} ({sz} bytes)")

    # --------------------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------------------
    cv4 = pd.read_csv(input_files['CV_Predictions_V4.csv'])
    cv4['cutoff'] = pd.to_datetime(cv4['cutoff'])
    cv4['real_date'] = pd.to_datetime(cv4['real_date'])

    df_dl = pd.read_csv(input_files['dataset_deep_learning.csv'])
    df_dl['Date'] = pd.to_datetime(df_dl['Date'])
    hist_series = pd.Series(df_dl['ret_crypto_btc_idr_indodax'].values**2, index=df_dl['Date'])

    har_h1_df = pd.read_csv(input_files['har_preds_T0_h1_v2.csv'])
    har_h10_df = pd.read_csv(input_files['har_preds_T0_h10_v2.csv'])
    har_h1_df['Date'] = pd.to_datetime(har_h1_df['Date'])
    har_h10_df['Date'] = pd.to_datetime(har_h10_df['Date'])

    garch_df = pd.read_csv(input_files['GARCH_Predictions_V4.csv'])
    m_garch = v5_helpers.load_garch_merged(cv4, garch_df)
    g1 = m_garch[m_garch['h_step'] == 1].reset_index(drop=True)
    g10 = m_garch[m_garch['h_step'] == 10].reset_index(drop=True)

    ens_df = pd.read_csv(input_files['v6_ensemble_predictions.csv'])
    ens_df['cutoff'] = pd.to_datetime(ens_df['cutoff'])
    ens_df['real_date'] = pd.to_datetime(ens_df['real_date'])

    c_origins = v5_helpers.get_cutoff_origins(cv4, hist_series)
    grp_h1 = cv4.groupby('cutoff').head(1).reset_index(drop=True)
    grp_h10 = cv4.groupby('cutoff').tail(1).reset_index(drop=True)

    hexp_arr_h1, roll_arr_h1, _ = v5_helpers.compute_baselines(grp_h1, hist_series, c_origins)
    hexp_arr_h10, roll_arr_h10, _ = v5_helpers.compute_baselines(grp_h10, hist_series, c_origins)

    # --------------------------------------------------------------------------
    # BUILD UNIFIED DATASETS WITH STRICT (cutoff, real_date) MERGE
    # --------------------------------------------------------------------------
    def assemble_horizon_data(h_step_val, base_grp, har_df, hexp_arr, roll_arr, garch_sub):
        df_base = base_grp[['cutoff', 'real_date', 'y']].copy()
        
        # Merge HAR
        m_har = pd.merge(df_base, har_df[['Date', 'HAR']], left_on='real_date', right_on='Date', how='inner')
        assert len(m_har) == 335, f"HAR merge len != 335 (got {len(m_har)})"
        
        models_dict = {
            'HAR': m_har['HAR'].values,
            'Hist_Exp': hexp_arr,
            'Roll_20': roll_arr,
            'GARCH11': garch_sub['GARCH11'].values,
            'EGARCH': garch_sub['EGARCH'].values,
        }
        
        # Merge DL ensemble configs
        for cfg in ['TFT_SPILL_T0', 'TFT_NOSPILL_T0', 'NBEATSx_SPILL_T0', 'NBEATSx_NOSPILL_T0']:
            clean_name = cfg.replace('_T0', '')
            sub_ens = ens_df[ens_df['Config'] == cfg][['cutoff', 'real_date', 'prediction']]
            m_dl = pd.merge(df_base[['cutoff', 'real_date']], sub_ens, on=['cutoff', 'real_date'], how='inner')
            assert len(m_dl) == 335, f"DL {cfg} merge len != 335 (got {len(m_dl)})"
            models_dict[clean_name] = m_dl['prediction'].values
            
        return df_base, models_dict

    df_h1, models_h1 = assemble_horizon_data(1, grp_h1, har_h1_df, hexp_arr_h1, roll_arr_h1, g1)
    df_h10, models_h10 = assemble_horizon_data(10, grp_h10, har_h10_df, hexp_arr_h10, roll_arr_h10, g10)

    # --------------------------------------------------------------------------
    # LANGKAH 1: SANITY CHECK FULL-SAMPLE MSE REPRODUCTION
    # --------------------------------------------------------------------------
    print("\n[LANGKAH 1: SANITY CHECK FULL-SAMPLE MSE]")
    y_h1 = df_h1['y'].values
    y_h10 = df_h10['y'].values

    mse_har_h1 = np.mean((y_h1 - models_h1['HAR'])**2)
    mse_har_h10 = np.mean((y_h10 - models_h10['HAR'])**2)
    mse_tft_spill_h1 = np.mean((y_h1 - models_h1['TFT_SPILL'])**2)
    mse_tft_nospill_h1 = np.mean((y_h1 - models_h1['TFT_NOSPILL'])**2)
    mse_nb_spill_h1 = np.mean((y_h1 - models_h1['NBEATSx_SPILL'])**2)

    print(f"  HAR h=1 MSE:             {mse_har_h1:.12e} (expected: 2.234681e-06, diff: {abs(mse_har_h1 - 2.234681e-06):.2e})")
    print(f"  HAR h=10 MSE:            {mse_har_h10:.12e} (expected: 2.439743e-06, diff: {abs(mse_har_h10 - 2.439743e-06):.2e})")
    print(f"  TFT SPILL ens h=1 MSE:   {mse_tft_spill_h1:.12e} (expected: 2.388453e-06, diff: {abs(mse_tft_spill_h1 - 2.388453e-06):.2e})")
    print(f"  TFT NOSPILL ens h=1 MSE: {mse_tft_nospill_h1:.12e} (expected: 2.394793e-06, diff: {abs(mse_tft_nospill_h1 - 2.394793e-06):.2e})")
    print(f"  NBEATSx SPILL ens h=1:   {mse_nb_spill_h1:.12e} (expected: 2.272067e-06, diff: {abs(mse_nb_spill_h1 - 2.272067e-06):.2e})")

    tol = 1e-11
    assert abs(mse_har_h1 - 2.2346806329693047e-06) < tol, "HAR h1 mismatch"
    assert abs(mse_har_h10 - 2.4397431333520627e-06) < tol, "HAR h10 mismatch"
    assert abs(mse_tft_spill_h1 - 2.3884532468485217e-06) < tol, "TFT SPILL h1 mismatch"
    assert abs(mse_tft_nospill_h1 - 2.394793350569176e-06) < tol, "TFT NOSPILL h1 mismatch"
    assert abs(mse_nb_spill_h1 - 2.272067473140114e-06) < tol, "NBEATSx SPILL h1 mismatch"
    print("  >>> SANITY CHECK PASS: All MSE values match ground truth exactly! <<<\n")

    # --------------------------------------------------------------------------
    # LANGKAH 2: METRICS BY EXCLUSION LEVEL
    # --------------------------------------------------------------------------
    print("[LANGKAH 2: METRICS BY EXCLUSION LEVEL]")
    clip_threshold = 1e-10

    def compute_metrics_series(y_true, y_pred):
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        y_c = np.clip(y_true, clip_threshold, None)
        p_c = np.clip(y_pred, clip_threshold, None)
        ratio = p_c / y_c
        qlike = np.mean(ratio - np.log(ratio) - 1.0)
        return mse, mae, qlike

    # Check clipping count
    print("  Clipping diagnostics (< 1e-10):")
    for h_label, m_dict in [('h=1', models_h1), ('h=10', models_h10)]:
        for m_name, arr in m_dict.items():
            n_clip = int(np.sum(arr < clip_threshold))
            pct_clip = (n_clip / len(arr)) * 100.0
            if n_clip > 0:
                print(f"    {m_name} ({h_label}): {n_clip} rows clipped ({pct_clip:.2f}%)")

    metrics_records = []
    model_order = ['HAR', 'Hist_Exp', 'Roll_20', 'GARCH11', 'EGARCH', 'TFT_SPILL', 'TFT_NOSPILL', 'NBEATSx_SPILL', 'NBEATSx_NOSPILL']
    
    for h_label, df_h, m_dict in [('h=1', df_h1, models_h1), ('h=10', df_h10, models_h10)]:
        y = df_h['y'].values
        q99 = np.percentile(y, 99)
        q975 = np.percentile(y, 97.5)
        q95 = np.percentile(y, 95)
        
        exclusions = [
            ('none', np.full(len(y), True)),
            ('top_1%', y <= q99),
            ('top_2.5%', y <= q975),
            ('top_5%', y <= q95)
        ]
        
        for ex_lvl, mask in exclusions:
            y_sub = y[mask]
            n_obs = len(y_sub)
            for m_name in model_order:
                p_sub = m_dict[m_name][mask]
                mse, mae, qlike = compute_metrics_series(y_sub, p_sub)
                metrics_records.append({
                    'model': m_name,
                    'horizon': h_label,
                    'exclusion_level': ex_lvl,
                    'MSE': mse,
                    'MAE': mae,
                    'QLIKE': qlike,
                    'n': n_obs
                })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(OUT_DIR, "metrics_by_exclusion.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved: {metrics_csv_path} ({len(metrics_df)} rows)")

    # --------------------------------------------------------------------------
    # LANGKAH 3: DIEBOLD-MARIANO TESTS
    # --------------------------------------------------------------------------
    print("\n[LANGKAH 3: DIEBOLD-MARIANO TESTS ACROSS EXCLUSIONS]")
    
    def diebold_mariano_engine(y_true, y_p1, y_p2, h_val, loss_type='MSE'):
        y_c = np.clip(y_true, clip_threshold, None)
        p1_c = np.clip(y_p1, clip_threshold, None)
        p2_c = np.clip(y_p2, clip_threshold, None)
        
        if loss_type == 'MSE':
            d = (y_true - y_p1)**2 - (y_true - y_p2)**2
        elif loss_type == 'MAE':
            d = np.abs(y_true - y_p1) - np.abs(y_true - y_p2)
        elif loss_type == 'QLIKE':
            q1 = p1_c / y_c - np.log(p1_c / y_c) - 1.0
            q2 = p2_c / y_c - np.log(p2_c / y_c) - 1.0
            d = q1 - q2
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
            
        T = len(d)
        d_bar = np.mean(d)
        gamma0 = np.var(d, ddof=0)
        gamma_sum = 0.0
        lag = 0 if h_val == 1 else (h_val - 1)
        for l in range(1, lag + 1):
            weight = 1.0 - (l / (lag + 1))
            cov = np.cov(d[l:], d[:-l], ddof=0)[0, 1]
            gamma_sum += 2.0 * weight * cov
            
        var_d = gamma0 + gamma_sum
        if var_d <= 0:
            stat = 0.0
            p_raw = 1.0
        else:
            stat = d_bar / np.sqrt(var_d / T)
            if h_val > 1:
                hln = np.sqrt((T + 1 - 2*h_val + (h_val * (h_val - 1)) / T) / T)
                stat = stat * hln
            p_raw = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))
            
        p_bonf_m8 = min(1.0, p_raw * 8)
        p_bonf_m12 = min(1.0, p_raw * 12)
        return stat, p_raw, p_bonf_m8, p_bonf_m12

    dl_list = ['TFT_SPILL', 'TFT_NOSPILL', 'NBEATSx_SPILL', 'NBEATSx_NOSPILL']
    baseline_list = ['Hist_Exp', 'Roll_20', 'HAR']
    losses = ['MSE', 'MAE', 'QLIKE']

    dm_records = []
    for h_label, h_val, df_h, m_dict in [('h=1', 1, df_h1, models_h1), ('h=10', 10, df_h10, models_h10)]:
        y = df_h['y'].values
        q99 = np.percentile(y, 99)
        q975 = np.percentile(y, 97.5)
        q95 = np.percentile(y, 95)
        
        exclusions = [
            ('none', np.full(len(y), True)),
            ('top_1%', y <= q99),
            ('top_2.5%', y <= q975),
            ('top_5%', y <= q95)
        ]
        
        for ex_lvl, mask in exclusions:
            y_sub = y[mask]
            for dl_m in dl_list:
                p_dl = m_dict[dl_m][mask]
                for bl_m in baseline_list:
                    p_bl = m_dict[bl_m][mask]
                    pair_name = f"{dl_m} vs {bl_m}"
                    for loss in losses:
                        stat, p_raw, p_m8, p_m12 = diebold_mariano_engine(y_sub, p_dl, p_bl, h_val, loss_type=loss)
                        dm_records.append({
                            'pair': pair_name,
                            'horizon': h_label,
                            'loss': loss,
                            'exclusion_level': ex_lvl,
                            'DM_stat': stat,
                            'p_raw': p_raw,
                            'p_bonf_m8': p_m8,
                            'p_bonf_m12': p_m12
                        })

    dm_df = pd.DataFrame(dm_records)
    dm_csv_path = os.path.join(OUT_DIR, "dm_by_exclusion.csv")
    dm_df.to_csv(dm_csv_path, index=False)
    print(f"  Saved: {dm_csv_path} ({len(dm_df)} rows)")

    # --------------------------------------------------------------------------
    # LANGKAH 4: EXTREME DAYS AND SSE ANALYSIS
    # --------------------------------------------------------------------------
    print("\n[LANGKAH 4: EXTREME DAYS & SSE CONCENTRATION]")
    # Target dates from h=1
    y1 = df_h1['y'].values
    q95_1 = np.percentile(y1, 95)
    
    total_sse_har_h1 = np.sum((y1 - models_h1['HAR'])**2)
    total_sse_tft_sp_h1 = np.sum((y1 - models_h1['TFT_SPILL'])**2)
    total_sse_tft_nosp_h1 = np.sum((y1 - models_h1['TFT_NOSPILL'])**2)

    total_sse_har_h10 = np.sum((df_h10['y'].values - models_h10['HAR'])**2)
    total_sse_tft_sp_h10 = np.sum((df_h10['y'].values - models_h10['TFT_SPILL'])**2)
    total_sse_tft_nosp_h10 = np.sum((df_h10['y'].values - models_h10['TFT_NOSPILL'])**2)

    # Merge returns from df_dl
    dl_ret_map = dict(zip(df_dl['Date'].dt.strftime('%Y-%m-%d'), df_dl['ret_crypto_btc_idr_indodax']))

    extreme_df = pd.DataFrame({
        'cutoff_h1': df_h1['cutoff'].dt.strftime('%Y-%m-%d'),
        'cutoff_h10': df_h10['cutoff'].dt.strftime('%Y-%m-%d'),
        'real_date': df_h1['real_date'].dt.strftime('%Y-%m-%d'),
        'y': y1,
        'HAR_pred_h1': models_h1['HAR'],
        'TFT_SPILL_pred_h1': models_h1['TFT_SPILL'],
        'TFT_NOSPILL_pred_h1': models_h1['TFT_NOSPILL'],
        'HAR_pred_h10': models_h10['HAR'],
        'TFT_SPILL_pred_h10': models_h10['TFT_SPILL'],
        'TFT_NOSPILL_pred_h10': models_h10['TFT_NOSPILL'],
    })

    extreme_df['return_btc_idr'] = extreme_df['real_date'].map(dl_ret_map)
    extreme_df['abs_return'] = np.sqrt(extreme_df['y'])
    extreme_df['pct_return'] = extreme_df['return_btc_idr'] * 100.0

    extreme_df['sse_HAR_h1'] = (extreme_df['y'] - extreme_df['HAR_pred_h1'])**2
    extreme_df['sse_TFT_SPILL_h1'] = (extreme_df['y'] - extreme_df['TFT_SPILL_pred_h1'])**2
    extreme_df['sse_TFT_NOSPILL_h1'] = (extreme_df['y'] - extreme_df['TFT_NOSPILL_pred_h1'])**2

    extreme_df['pct_total_sse_HAR_h1'] = (extreme_df['sse_HAR_h1'] / total_sse_har_h1) * 100.0
    extreme_df['pct_total_sse_TFT_SPILL_h1'] = (extreme_df['sse_TFT_SPILL_h1'] / total_sse_tft_sp_h1) * 100.0
    extreme_df['pct_total_sse_TFT_NOSPILL_h1'] = (extreme_df['sse_TFT_NOSPILL_h1'] / total_sse_tft_nosp_h1) * 100.0

    # Sort descending by y
    extreme_sorted = extreme_df.sort_values('y', ascending=False).reset_index(drop=True)
    extreme_sorted['rank'] = extreme_sorted.index + 1

    top17_days = extreme_sorted.head(17).copy()
    
    # Add market notes
    notes = {
        '2026-02-05': 'Extreme crash: Bitcoin dropped ~14.85% in 1 day; massive market volatility surge',
        '2026-02-06': 'Extreme rebound: Bitcoin surged +11.75% next day; HAR autoregressive lag reacted immediately',
        '2026-08-21': 'Summer rally: Bitcoin surged +6.53%',
        '2026-06-02': 'Early June dip: Bitcoin dropped -6.11%',
        '2026-03-04': 'March breakout: Bitcoin jumped +5.71%',
        '2026-08-19': 'August rally start: Bitcoin jumped +5.64%',
        '2025-11-14': 'Mid-November pullback: Bitcoin dropped -5.63%',
        '2026-08-20': 'August continuation: Bitcoin jumped +5.48%',
        '2025-10-10': 'October correction: Bitcoin dropped -5.42%',
        '2025-05-08': 'May rally: Bitcoin jumped +5.18%',
        '2026-02-25': 'Late February rally: Bitcoin jumped +5.08%',
        '2025-11-20': 'Late November drop: Bitcoin dropped -5.00%',
        '2026-04-13': 'Mid-April rebound: Bitcoin jumped +4.93%',
        '2026-03-02': 'Early March run: Bitcoin jumped +4.86%',
        '2025-12-02': 'Early December pump: Bitcoin jumped +4.73%',
        '2026-02-23': 'Late February dip: Bitcoin dropped -4.70%',
        '2025-12-01': 'Early December drop: Bitcoin dropped -4.69%'
    }
    top17_days['event_note'] = top17_days['real_date'].map(notes)

    extreme_csv_path = os.path.join(OUT_DIR, "extreme_days.csv")
    cols_to_save = [
        'rank', 'real_date', 'cutoff_h1', 'cutoff_h10', 'y', 'return_btc_idr', 'pct_return',
        'HAR_pred_h1', 'TFT_SPILL_pred_h1', 'TFT_NOSPILL_pred_h1',
        'HAR_pred_h10', 'TFT_SPILL_pred_h10', 'TFT_NOSPILL_pred_h10',
        'pct_total_sse_HAR_h1', 'pct_total_sse_TFT_SPILL_h1', 'pct_total_sse_TFT_NOSPILL_h1',
        'event_note'
    ]
    top17_days[cols_to_save].to_csv(extreme_csv_path, index=False)
    print(f"  Saved: {extreme_csv_path} (17 rows)")

    top1_har_sse = top17_days.head(4)['pct_total_sse_HAR_h1'].sum()
    top1_tft_sse = top17_days.head(4)['pct_total_sse_TFT_SPILL_h1'].sum()
    print(f"\n  Total SSE from top 1% days (top 4 days: Feb 5, Feb 6, Aug 21, Jun 2):")
    print(f"    HAR h=1:       {top1_har_sse:.2f}% of total test SSE")
    print(f"    TFT SPILL h=1: {top1_tft_sse:.2f}% of total test SSE")

    # --------------------------------------------------------------------------
    # LANGKAH 5: KEY QUESTION INVESTIGATION (TFT vs Hist_Exp h=1)
    # --------------------------------------------------------------------------
    print("\n[LANGKAH 5: KEY RESEARCH QUESTION]")
    print("  Question: Apakah TFT SPILL/NOSPILL vs Hist_Exp h=1 masih signifikan setelah outlier dikeluarkan?")
    
    for pair in ['TFT_SPILL vs Hist_Exp', 'TFT_NOSPILL vs Hist_Exp']:
        sub = dm_df[(dm_df['pair'] == pair) & (dm_df['horizon'] == 'h=1') & (dm_df['loss'] == 'MSE')]
        print(f"\n  --- {pair} (h=1, loss=MSE) ---")
        for idx, r in sub.iterrows():
            print(f"    Exclusion: {r['exclusion_level']:8s} | DM_stat: {r['DM_stat']:8.4f} | p_raw: {r['p_raw']:.6e} | p_bonf_m8: {r['p_bonf_m8']:.6e} | p_bonf_m12: {r['p_bonf_m12']:.6e}")

    print("\n================================================================================")
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("================================================================================")

if __name__ == "__main__":
    main()
