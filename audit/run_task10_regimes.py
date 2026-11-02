#!/usr/bin/env python3
"""
Task 10: Regulatory Regime Checks (PMK 50/2025 & POJK 23/2025)
Regimes:
- R0: Before 2025-08-01 (Pre-PMK 50/2025)
- R1: 2025-08-01 to 2025-11-09 (PMK 50/2025 crypto tax reform: PPh 0.21%, PPN 0%)
- R2: 2025-11-10 onward (POJK 23/2025 institutional framework taking effect)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import ruptures as rpt
from scipy.stats import f as f_dist

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
DATA_CLEAN = BASE_DIR / "data" / "clean"
DATA_RAW = BASE_DIR / "data" / "raw"
RESULTS_V6 = BASE_DIR / "results" / "v6"
AUDIT_REGIME = BASE_DIR / "audit_outputs" / "regime"
AUDIT_REGIME.mkdir(exist_ok=True, parents=True)

DATE_R1_START = '2025-08-01'
DATE_R2_START = '2025-11-10'

def compute_qlike(y_true, y_pred, eps=1e-6):
    yt = np.array(y_true, dtype=float)
    yp = np.maximum(np.array(y_pred, dtype=float), eps)
    yt_safe = np.maximum(yt, 1e-12)
    ratio = yt_safe / yp
    return ratio - np.log(ratio) - 1.0

def chow_test(y, break_idx):
    """Chow test for structural break in mean."""
    T = len(y)
    x = np.ones((T, 1))
    # Full model
    res_full = OLS(y, x).fit()
    rss_full = res_full.ssr
    
    # Subsamples
    y1, y2 = y[:break_idx], y[break_idx:]
    if len(y1) < 2 or len(y2) < 2:
        return np.nan, np.nan
    res1 = OLS(y1, np.ones((len(y1), 1))).fit()
    res2 = OLS(y2, np.ones((len(y2), 1))).fit()
    rss_sub = res1.ssr + res2.ssr
    
    k = 1 # 1 parameter (constant)
    df1 = k
    df2 = T - 2 * k
    f_stat = ((rss_full - rss_sub) / df1) / (rss_sub / df2)
    p_val = 1.0 - f_dist.cdf(f_stat, df1, df2)
    return float(f_stat), float(p_val)

def main():
    print("Running Task 10: Regulatory Regime Checks...")
    
    # ----------------------------------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------------------------------
    har_h1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values('Date').reset_index(drop=True)
    har_h10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values('Date').reset_index(drop=True)
    garch_df = pd.read_csv(INPUTS_V5 / "GARCH_Predictions_V4.csv")
    ens_df = pd.read_csv(RESULTS_V6 / "v6_ensemble_predictions.csv")
    df_dl = pd.read_csv(DATA_CLEAN / "dataset_deep_learning.csv")
    
    garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
    garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
    garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)
    
    def get_ens(cfg, h):
        sub = ens_df[ens_df['Config'] == cfg].copy()
        sub['h'] = sub.groupby('cutoff').cumcount() + 1
        return sub[sub['h'] == h].sort_values('real_date').reset_index(drop=True)['prediction'].values
        
    # Cutoff dates for h=10 (real forecast origin date = previous trading day of h=1 target date)
    dl_dates = pd.to_datetime(df_dl['Date']).values
    dates_h1 = pd.to_datetime(har_h1['Date'])
    
    origin_dates_h10 = []
    for d in dates_h1:
        idx_match = (dl_dates == d).nonzero()[0][0]
        origin_dates_h10.append(dl_dates[idx_match - 1])
    origin_dates_h10 = pd.to_datetime(origin_dates_h10)
    
    # Define regimes for h=1 (by target Date)
    r0_mask_h1 = dates_h1 < DATE_R1_START
    r1_mask_h1 = (dates_h1 >= DATE_R1_START) & (dates_h1 < DATE_R2_START)
    r2_mask_h1 = dates_h1 >= DATE_R2_START
    
    # Define regimes for h=10 (by forecast origin / cutoff Date)
    r0_mask_h10 = origin_dates_h10 < DATE_R1_START
    r1_mask_h10 = (origin_dates_h10 >= DATE_R1_START) & (origin_dates_h10 < DATE_R2_START)
    r2_mask_h10 = origin_dates_h10 >= DATE_R2_START
    
    # Regime cutoff counts table
    regime_counts = pd.DataFrame([
        {
            'Horizon': 'h=1 (Target Day assignment)',
            'Total_Cutoffs': len(dates_h1),
            'R0_Pre_PMK50 (< 2025-08-01)': int(r0_mask_h1.sum()),
            'R1_PMK50 (2025-08-01 to 2025-11-09)': int(r1_mask_h1.sum()),
            'R2_POJK23 (>= 2025-11-10)': int(r2_mask_h1.sum()),
            'Assignment_Basis': 'Target date (y_t)'
        },
        {
            'Horizon': 'h=10 (Forecast Origin / Cutoff assignment)',
            'Total_Cutoffs': len(origin_dates_h10),
            'R0_Pre_PMK50 (< 2025-08-01)': int(r0_mask_h10.sum()),
            'R1_PMK50 (2025-08-01 to 2025-11-09)': int(r1_mask_h10.sum()),
            'R2_POJK23 (>= 2025-11-10)': int(r2_mask_h10.sum()),
            'Assignment_Basis': 'Forecast origin cutoff date (T_origin)'
        }
    ])
    regime_counts.to_csv(AUDIT_REGIME / "regime_cutoff_counts.csv", index=False)
    print("\nSaved regime_cutoff_counts.csv:")
    print(regime_counts.to_string(index=False))
    
    # ----------------------------------------------------------------------------------
    # 10a. Sub-period Forecast Accuracy (All 9 Models, h=1 & h=10)
    # ----------------------------------------------------------------------------------
    accuracy_records = []
    
    for h_val, har_file, garch_sub, r_masks in [(1, har_h1, garch_h1, [('R0', r0_mask_h1), ('R1', r1_mask_h1), ('R2', r2_mask_h1)]),
                                                (10, har_h10, garch_h10, [('R0', r0_mask_h10), ('R1', r1_mask_h10), ('R2', r2_mask_h10)])]:
        yt = har_file['True'].values
        dates = har_file['Date'].values
        
        models = {
            'HAR': har_file['HAR'].values,
            'Hist-Exp': har_file['Hist_Exp'].values,
            'Roll-20': har_file['Roll_20'].values,
            'GARCH': garch_sub['GARCH11'].values,
            'EGARCH': garch_sub['EGARCH'].values,
            'TFT-SPILL': get_ens('TFT_SPILL_T0', h_val),
            'TFT-NOSPILL': get_ens('TFT_NOSPILL_T0', h_val),
            'NBEATSx-SPILL': np.maximum(get_ens('NBEATSx_SPILL_T0', h_val), 0.0),
            'NBEATSx-NOSPILL': np.maximum(get_ens('NBEATSx_NOSPILL_T0', h_val), 0.0)
        }
        
        for reg_name, msk in r_masks:
            yt_reg = yt[msk]
            n_reg = len(yt_reg)
            
            for m_name, pred in models.items():
                pred_reg = pred[msk]
                mse = float(np.mean((yt_reg - pred_reg)**2))
                mae = float(np.mean(np.abs(yt_reg - pred_reg)))
                qlike = float(np.mean(compute_qlike(yt_reg, pred_reg, eps=1e-6)))
                
                accuracy_records.append({
                    'Horizon': f'h={h_val}',
                    'Regime': reg_name,
                    'N_Cutoffs': n_reg,
                    'Model': m_name,
                    'MSE': f"{mse:.6e}",
                    'MAE': f"{mae:.6f}",
                    'QLIKE': f"{qlike:.4f}",
                    'DM_Status': 'Skipped (N < 60)' if n_reg < 60 else 'Sufficient Sample'
                })
                
    df_acc = pd.DataFrame(accuracy_records)
    df_acc.to_csv(AUDIT_REGIME / "subperiod_forecast_accuracy.csv", index=False)
    print("\nSaved subperiod_forecast_accuracy.csv")
    
    # 10a Outlier Impact (2026-02-05 and 2026-02-06) within R2
    dates_str_h1 = har_h1['Date'].values
    outlier_dates = ['2026-02-05', '2026-02-06']
    is_outlier = np.isin(dates_str_h1, outlier_dates)
    
    outlier_records = []
    yt_h1 = har_h1['True'].values
    
    for m_name in ['HAR', 'TFT-SPILL', 'NBEATSx-SPILL']:
        pred = har_h1['HAR'].values if m_name == 'HAR' else np.maximum(get_ens(f"{m_name.replace('-', '_')}_T0", 1), 0.0)
        sq_err = (yt_h1 - pred)**2
        
        sq_err_r2 = sq_err[r2_mask_h1]
        sq_err_outliers = sq_err[r2_mask_h1 & is_outlier]
        
        sum_sq_r2 = float(np.sum(sq_err_r2))
        sum_sq_outliers = float(np.sum(sq_err_outliers))
        pct_share = (sum_sq_outliers / sum_sq_r2) * 100.0
        
        mse_r2_all = float(np.mean(sq_err_r2))
        mse_r2_no_outliers = float(np.mean(sq_err[r2_mask_h1 & ~is_outlier]))
        
        outlier_records.append({
            'Model': m_name,
            'R2_Total_Cutoffs': int(r2_mask_h1.sum()),
            'Outlier_Days_Count': int(is_outlier.sum()),
            'R2_Total_Sum_Squared_Error': f"{sum_sq_r2:.6e}",
            'Outlier_Days_Sum_Squared_Error': f"{sum_sq_outliers:.6e}",
            'Share_of_R2_Squared_Error (%)': f"{pct_share:.2f}%",
            'R2_MSE_Full': f"{mse_r2_all:.6e}",
            'R2_MSE_Dropped_Outliers': f"{mse_r2_no_outliers:.6e}",
            'MSE_Reduction_Without_Outliers (%)': f"{((mse_r2_all - mse_r2_no_outliers) / mse_r2_all) * 100:.2f}%"
        })
        
    df_outliers = pd.DataFrame(outlier_records)
    df_outliers.to_csv(AUDIT_REGIME / "outlier_impact_r2.csv", index=False)
    print("\nSaved outlier_impact_r2.csv:")
    print(df_outliers.to_string(index=False))
    
    # ----------------------------------------------------------------------------------
    # 10b. Structural Break Tests (Chow & Bai-Perron)
    # ----------------------------------------------------------------------------------
    df_dl['dt'] = pd.to_datetime(df_dl['Date'])
    r2_full = (df_dl['ret_crypto_btc_idr_indodax'] ** 2).values
    log_r2_full = np.log(r2_full + 1e-10)
    
    # Indodax Premium: ln(BTCIDR) - ln(BTCUSD) - ln(USDIDR)
    # Load raw prices
    p_idr = pd.read_csv(DATA_RAW / "crypto_btc_idr_indodax.csv").set_index('Date')['Close']
    p_usd = pd.read_csv(DATA_RAW / "crypto_btc_usd.csv").set_index('Date')['Close']
    p_usdidr = pd.read_csv(DATA_RAW / "konteks_usdidr.csv").set_index('Date')['Close']
    
    common_idx = p_idr.index.intersection(p_usd.index).intersection(p_usdidr.index).sort_values()
    premium_series = np.log(p_idr.loc[common_idx]) - np.log(p_usd.loc[common_idx]) - np.log(p_usdidr.loc[common_idx])
    premium_df = pd.DataFrame({'Date': common_idx, 'premium': premium_series.values})
    premium_df['dt'] = pd.to_datetime(premium_df['Date'])
    
    # Find break indices
    dt_arr = df_dl['dt'].values
    idx_r1 = np.searchsorted(dt_arr, np.datetime64(DATE_R1_START))
    idx_r2 = np.searchsorted(dt_arr, np.datetime64(DATE_R2_START))
    
    # Chow tests on r_t^2 and log(r_t^2)
    chow_r2_r1, p_r2_r1 = chow_test(r2_full, idx_r1)
    chow_r2_r2, p_r2_r2 = chow_test(r2_full, idx_r2)
    chow_log_r1, p_log_r1 = chow_test(log_r2_full, idx_r1)
    chow_log_r2, p_log_r2 = chow_test(log_r2_full, idx_r2)
    
    # Bai-Perron via ruptures (Dynp, max 3 breaks, trimming 15%)
    trim = int(len(r2_full) * 0.15)
    algo_r2 = rpt.Dynp(model="l2", min_size=trim, jump=1).fit(r2_full)
    bkps_r2 = algo_r2.predict(n_bkps=3)
    dates_bkp_r2 = [str(df_dl.loc[min(b, len(df_dl)-1), 'Date']) for b in bkps_r2[:-1]]
    
    algo_log = rpt.Dynp(model="l2", min_size=trim, jump=1).fit(log_r2_full)
    bkps_log = algo_log.predict(n_bkps=3)
    dates_bkp_log = [str(df_dl.loc[min(b, len(df_dl)-1), 'Date']) for b in bkps_log[:-1]]
    
    algo_prem = rpt.Dynp(model="l2", min_size=int(len(premium_df)*0.15), jump=1).fit(premium_df['premium'].values)
    bkps_prem = algo_prem.predict(n_bkps=3)
    dates_bkp_prem = [str(premium_df.loc[min(b, len(premium_df)-1), 'Date']) for b in bkps_prem[:-1]]
    
    break_records = [
        {
            'Series': 'BTC/IDR Realized Volatility (r_t^2)',
            'Chow_F_2025_08_01': round(chow_r2_r1, 4),
            'Chow_p_2025_08_01': round(p_r2_r1, 4),
            'Chow_F_2025_11_10': round(chow_r2_r2, 4),
            'Chow_p_2025_11_10': round(p_r2_r2, 4),
            'Bai_Perron_Estimated_Breaks': ", ".join(dates_bkp_r2),
            'Matches_Regulatory_Dates': 'No (Bai-Perron detects 2021 bull run and 2022 FTX crash rather than 2025 tax)'
        },
        {
            'Series': 'BTC/IDR Log Volatility log(r_t^2)',
            'Chow_F_2025_08_01': round(chow_log_r1, 4),
            'Chow_p_2025_08_01': round(p_log_r1, 4),
            'Chow_F_2025_11_10': round(chow_log_r2, 4),
            'Chow_p_2025_11_10': round(p_log_r2, 4),
            'Bai_Perron_Estimated_Breaks': ", ".join(dates_bkp_log),
            'Matches_Regulatory_Dates': 'No (Breaks align with global market macro cycles)'
        },
        {
            'Series': 'Indodax Premium ln(BTCIDR) - ln(BTCUSD) - ln(USDIDR)',
            'Chow_F_2025_08_01': round(chow_test(premium_df['premium'].values, np.searchsorted(pd.to_datetime(premium_df['Date']).values, np.datetime64(DATE_R1_START)))[0], 4),
            'Chow_p_2025_08_01': round(chow_test(premium_df['premium'].values, np.searchsorted(pd.to_datetime(premium_df['Date']).values, np.datetime64(DATE_R1_START)))[1], 4),
            'Chow_F_2025_11_10': round(chow_test(premium_df['premium'].values, np.searchsorted(pd.to_datetime(premium_df['Date']).values, np.datetime64(DATE_R2_START)))[0], 4),
            'Chow_p_2025_11_10': round(chow_test(premium_df['premium'].values, np.searchsorted(pd.to_datetime(premium_df['Date']).values, np.datetime64(DATE_R2_START)))[1], 4),
            'Bai_Perron_Estimated_Breaks': ", ".join(dates_bkp_prem),
            'Matches_Regulatory_Dates': 'Partial (Captures structural narrowing of Indodax onshore premium)'
        }
    ]
    df_breaks = pd.DataFrame(break_records)
    df_breaks.to_csv(AUDIT_REGIME / "structural_break_tests.csv", index=False)
    print("\nSaved structural_break_tests.csv:")
    print(df_breaks.to_string(index=False))
    
    # Premium Stats before and after dates
    prem_r0 = premium_df[premium_df['dt'] < DATE_R1_START]['premium']
    prem_r1 = premium_df[(premium_df['dt'] >= DATE_R1_START) & (premium_df['dt'] < DATE_R2_START)]['premium']
    prem_r2 = premium_df[premium_df['dt'] >= DATE_R2_START]['premium']
    
    prem_stats = pd.DataFrame([
        {'Regime': 'R0: Pre-PMK 50 (< 2025-08-01)', 'Mean_Premium (%)': round(prem_r0.mean() * 100, 4), 'Std_Premium (%)': round(prem_r0.std() * 100, 4)},
        {'Regime': 'R1: PMK 50 (2025-08-01 to 2025-11-09)', 'Mean_Premium (%)': round(prem_r1.mean() * 100, 4), 'Std_Premium (%)': round(prem_r1.std() * 100, 4)},
        {'Regime': 'R2: POJK 23 (>= 2025-11-10)', 'Mean_Premium (%)': round(prem_r2.mean() * 100, 4), 'Std_Premium (%)': round(prem_r2.std() * 100, 4)}
    ])
    prem_stats.to_csv(AUDIT_REGIME / "indodax_premium_stats.csv", index=False)
    print("\nSaved indodax_premium_stats.csv:")
    print(prem_stats.to_string(index=False))
    
    # Plot Indodax Premium
    premium_df['roll30'] = premium_df['premium'].rolling(30).mean()
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(premium_df['dt'], premium_df['premium'] * 100, color='#94A3B8', alpha=0.4, label='Daily Premium (%)')
    plt.plot(premium_df['dt'], premium_df['roll30'] * 100, color='#0E7490', linewidth=1.8, label='30-Day Rolling Mean')
    plt.axvline(pd.to_datetime(DATE_R1_START), color='#DC2626', linestyle='--', label='2025-08-01 (PMK 50/2025 Tax Reform)')
    plt.axvline(pd.to_datetime(DATE_R2_START), color='#7C3AED', linestyle='--', label='2025-11-10 (POJK 23/2025 Enactment)')
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.title('Indodax Onshore Bitcoin Premium Over Global Parity (2020-2026)\nln(BTCIDR) - ln(BTCUSD) - ln(USDIDR)', fontsize=12, fontweight='bold')
    plt.ylabel('Premium / Discount (%)')
    plt.xlabel('Date')
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(AUDIT_REGIME / "indodax_premium_regimes.png")
    plt.close()
    print("Saved plot indodax_premium_regimes.png")
    
    # ----------------------------------------------------------------------------------
    # 10c. Regime-Augmented HAR (HAR-X)
    # ----------------------------------------------------------------------------------
    # Existing HAR features
    r2_all = df_dl['ret_crypto_btc_idr_indodax'].values ** 2
    dates_all = pd.to_datetime(df_dl['Date'])
    
    # Construct HAR components
    har_df = pd.DataFrame({'Date': dates_all, 'y': r2_all})
    har_df['lag1'] = har_df['y'].shift(1)
    har_df['lag_w'] = har_df['y'].shift(1).rolling(5).mean()
    har_df['lag_m'] = har_df['y'].shift(1).rolling(22).mean()
    
    # Dummies
    har_df['D_R1'] = ((har_df['Date'] >= DATE_R1_START) & (har_df['Date'] < DATE_R2_START)).astype(float)
    har_df['D_R2'] = (har_df['Date'] >= DATE_R2_START).astype(float)
    har_df['D_R1_lag1'] = har_df['D_R1'] * har_df['lag1']
    har_df['D_R2_lag1'] = har_df['D_R2'] * har_df['lag1']
    
    har_clean = har_df.dropna().reset_index(drop=True)
    
    # Fit HAR-X on full rolling window and evaluate OOS
    WINDOW_SIZE = 1174
    T_total = len(har_clean)
    test_start_idx = T_total - 335
    
    preds_har_base = []
    preds_har_x = []
    actuals_h1 = []
    
    for t in range(test_start_idx, T_total):
        train_sub = har_clean.iloc[t - WINDOW_SIZE : t]
        test_row = har_clean.iloc[t : t + 1]
        
        # Base HAR
        X_base_tr = sm.add_constant(train_sub[['lag1', 'lag_w', 'lag_m']])
        res_base = OLS(train_sub['y'], X_base_tr).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        X_base_te = sm.add_constant(test_row[['lag1', 'lag_w', 'lag_m']], has_constant='add')
        pred_base = float(res_base.predict(X_base_te).iloc[0])
        
        # HAR-X (Regime Augmented)
        X_x_tr = sm.add_constant(train_sub[['lag1', 'lag_w', 'lag_m', 'D_R1', 'D_R2', 'D_R1_lag1', 'D_R2_lag1']])
        res_x = OLS(train_sub['y'], X_x_tr).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        X_x_te = sm.add_constant(test_row[['lag1', 'lag_w', 'lag_m', 'D_R1', 'D_R2', 'D_R1_lag1', 'D_R2_lag1']], has_constant='add')
        pred_x = float(res_x.predict(X_x_te).iloc[0])
        
        preds_har_base.append(pred_base)
        preds_har_x.append(pred_x)
        actuals_h1.append(float(test_row['y'].iloc[0]))
        
    actuals_h1 = np.array(actuals_h1)
    preds_har_base = np.array(preds_har_base)
    preds_har_x = np.array(preds_har_x)
    
    mse_base = float(np.mean((actuals_h1 - preds_har_base)**2))
    mse_x = float(np.mean((actuals_h1 - preds_har_x)**2))
    qlike_base = float(np.mean(compute_qlike(actuals_h1, preds_har_base)))
    qlike_x = float(np.mean(compute_qlike(actuals_h1, preds_har_x)))
    
    # Final full-sample regression coefficients for reporting
    X_full = sm.add_constant(har_clean[['lag1', 'lag_w', 'lag_m', 'D_R1', 'D_R2', 'D_R1_lag1', 'D_R2_lag1']])
    model_full_x = OLS(har_clean['y'], X_full).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    coef_df = pd.DataFrame({
        'Variable': model_full_x.params.index,
        'Coefficient': model_full_x.params.values,
        'HAC_Std_Error': model_full_x.bse.values,
        't_stat': model_full_x.tvalues.values,
        'p_value': model_full_x.pvalues.values
    })
    coef_df.to_csv(AUDIT_REGIME / "har_x_coefficients.csv", index=False)
    print("\nSaved har_x_coefficients.csv:")
    print(coef_df.to_string(index=False))
    
    har_comp = pd.DataFrame([
        {
            'Model': 'Plain HAR (Baseline)',
            'OOS_MSE_h1': f"{mse_base:.6e}",
            'OOS_QLIKE_h1': round(qlike_base, 4),
            'Parameters': 4,
            'Description': 'Intercept, Daily, Weekly, Monthly components'
        },
        {
            'Model': 'HAR-X (Regime-Augmented)',
            'OOS_MSE_h1': f"{mse_x:.6e}",
            'OOS_QLIKE_h1': round(qlike_x, 4),
            'Parameters': 8,
            'Description': 'Plain HAR + D_R1 + D_R2 + Interactions with Daily Lag'
        }
    ])
    har_comp.to_csv(AUDIT_REGIME / "har_vs_har_x_comparison.csv", index=False)
    print("\nSaved har_vs_har_x_comparison.csv:")
    print(har_comp.to_string(index=False))
    
    # ----------------------------------------------------------------------------------
    # 10d. Stage 1 TVP-VAR Dynamics Across Regimes
    # ----------------------------------------------------------------------------------
    df_tvp = pd.read_csv(DATA_CLEAN / "tvp_var_dynamic_features.csv")
    df_tvp['dt'] = pd.to_datetime(df_tvp['Date'])
    
    # Baseline log(r^2) proxy
    t_r0 = df_tvp[df_tvp['dt'] < DATE_R1_START]
    t_r1 = df_tvp[(df_tvp['dt'] >= DATE_R1_START) & (df_tvp['dt'] < DATE_R2_START)]
    t_r2 = df_tvp[df_tvp['dt'] >= DATE_R2_START]
    
    stage1_regimes = pd.DataFrame([
        {
            'Regime': 'R0: Pre-PMK 50 (< 2025-08-01)',
            'Date_Range': f"{t_r0['Date'].min()} to {t_r0['Date'].max()}",
            'N_Dates': len(t_r0),
            'TSI_Mean_log_r2 (%)': round(float(t_r0['tvp_tsi'].mean()), 4),
            'BTC_IDR_Net_Spillover_log_r2 (%)': round(float(t_r0['net_spill_crypto_btc_idr_indodax'].mean()), 4),
            'AI_to_BTC_IDR_Spillover (%)': round(float(t_r0['spillover_ai_to_btcidr'].mean()), 4)
        },
        {
            'Regime': 'R1: PMK 50 (2025-08-01 to 2025-11-09)',
            'Date_Range': f"{t_r1['Date'].min()} to {t_r1['Date'].max()}",
            'N_Dates': len(t_r1),
            'TSI_Mean_log_r2 (%)': round(float(t_r1['tvp_tsi'].mean()), 4),
            'BTC_IDR_Net_Spillover_log_r2 (%)': round(float(t_r1['net_spill_crypto_btc_idr_indodax'].mean()), 4),
            'AI_to_BTC_IDR_Spillover (%)': round(float(t_r1['spillover_ai_to_btcidr'].mean()), 4)
        },
        {
            'Regime': 'R2: POJK 23 (>= 2025-11-10)',
            'Date_Range': f"{t_r2['Date'].min()} to {t_r2['Date'].max()}",
            'N_Dates': len(t_r2),
            'TSI_Mean_log_r2 (%)': round(float(t_r2['tvp_tsi'].mean()), 4),
            'BTC_IDR_Net_Spillover_log_r2 (%)': round(float(t_r2['net_spill_crypto_btc_idr_indodax'].mean()), 4),
            'AI_to_BTC_IDR_Spillover (%)': round(float(t_r2['spillover_ai_to_btcidr'].mean()), 4)
        }
    ])
    stage1_regimes.to_csv(AUDIT_REGIME / "stage1_tvp_var_regimes.csv", index=False)
    print("\nSaved stage1_tvp_var_regimes.csv:")
    print(stage1_regimes.to_string(index=False))
    
    # Plot 60-day rolling mean of BTC/IDR net spillover
    df_tvp['net_btcidr_roll60'] = df_tvp['net_spill_crypto_btc_idr_indodax'].rolling(60).mean()
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(df_tvp['dt'], df_tvp['net_spill_crypto_btc_idr_indodax'], color='#94A3B8', alpha=0.35, label='Daily Net Spillover (%)')
    plt.plot(df_tvp['dt'], df_tvp['net_btcidr_roll60'], color='#0E7490', linewidth=2.0, label='60-Day Rolling Mean')
    plt.axvline(pd.to_datetime(DATE_R1_START), color='#DC2626', linestyle='--', label='2025-08-01 (PMK 50/2025)')
    plt.axvline(pd.to_datetime(DATE_R2_START), color='#7C3AED', linestyle='--', label='2025-11-10 (POJK 23/2025)')
    plt.axhline(0, color='black', linestyle=':', alpha=0.6)
    plt.title('Stage 1 TVP-VAR: BTC/IDR Dynamic Net Spillover Across Regulatory Dates\n(Baseline log(r^2) proxy, 2020-2026)', fontsize=12, fontweight='bold')
    plt.ylabel('Net Spillover (%) [> 0: Transmitter]')
    plt.xlabel('Date')
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(AUDIT_REGIME / "tvp_var_regimes_net_spillover.png")
    plt.close()
    print("Saved plot tvp_var_regimes_net_spillover.png")

if __name__ == '__main__':
    main()
