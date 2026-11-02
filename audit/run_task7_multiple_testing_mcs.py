#!/usr/bin/env python3
"""
Task 7: Model Confidence Set (MCS) & Bonferroni Family Audit
Hansen, Lunde, & Nason (2011) Model Confidence Set via arch.bootstrap.MCS
MSE and QLIKE loss, alpha=0.10, bootstrap=5000, 9 models, h=1 and h=10.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
RESULTS_V6 = BASE_DIR / "results" / "v6"
AUDIT_OUT = BASE_DIR / "audit_outputs"
AUDIT_OUT.mkdir(exist_ok=True, parents=True)

def compute_qlike(y_true, y_pred, eps=1e-6):
    yt = np.array(y_true, dtype=float)
    yp = np.maximum(np.array(y_pred, dtype=float), eps)
    yt_safe = np.maximum(yt, 1e-12)
    ratio = yt_safe / yp
    return ratio - np.log(ratio) - 1.0

def main():
    print("Running Task 7: Model Confidence Set (MCS)...")
    har_h1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values('Date').reset_index(drop=True)
    har_h10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values('Date').reset_index(drop=True)
    garch_df = pd.read_csv(INPUTS_V5 / "GARCH_Predictions_V4.csv")
    ens_df = pd.read_csv(RESULTS_V6 / "v6_ensemble_predictions.csv")
    
    garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
    garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
    garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)
    
    def get_ens(cfg, h):
        sub = ens_df[ens_df['Config'] == cfg].copy()
        sub['h'] = sub.groupby('cutoff').cumcount() + 1
        return sub[sub['h'] == h].sort_values('real_date').reset_index(drop=True)['prediction'].values
        
    mcs_records = []
    
    for h_val, har_file, garch_sub in [(1, har_h1, garch_h1), (10, har_h10, garch_h10)]:
        yt = har_file['True'].values
        
        # 9 Models (NBEATSx predictions clipped >= 0 for fair loss computation)
        models = {
            'HAR': har_file['HAR'].values,
            'Hist-Exp': har_file['Hist_Exp'].values,
            'Roll-20': har_file['Roll_20'].values,
            'GARCH(1,1)': garch_sub['GARCH11'].values,
            'EGARCH(1,1)': garch_sub['EGARCH'].values,
            'TFT-SPILL': get_ens('TFT_SPILL_T0', h_val),
            'TFT-NOSPILL': get_ens('TFT_NOSPILL_T0', h_val),
            'NBEATSx-SPILL': np.maximum(get_ens('NBEATSx_SPILL_T0', h_val), 0.0),
            'NBEATSx-NOSPILL': np.maximum(get_ens('NBEATSx_NOSPILL_T0', h_val), 0.0)
        }
        
        # 1. MSE Losses
        losses_mse = pd.DataFrame({m: (yt - pred)**2 for m, pred in models.items()})
        print(f"\nComputing MCS for h={h_val} (MSE loss, 5000 bootstrap)...")
        mcs_mse = MCS(losses_mse, size=0.10, reps=5000, seed=42)
        mcs_mse.compute()
        pvals_mse = mcs_mse.pvalues
        
        # 2. QLIKE Losses
        losses_qlike = pd.DataFrame({m: compute_qlike(yt, pred, eps=1e-6) for m, pred in models.items()})
        print(f"Computing MCS for h={h_val} (QLIKE loss, 5000 bootstrap)...")
        mcs_qlike = MCS(losses_qlike, size=0.10, reps=5000, seed=42)
        mcs_qlike.compute()
        pvals_qlike = mcs_qlike.pvalues
        
        for m in models.keys():
            p_mse = float(pvals_mse.loc[m, 'Pvalue']) if m in pvals_mse.index else 0.0
            in_mcs_mse = 'Yes' if p_mse >= 0.10 else 'No'
            
            p_qlike = float(pvals_qlike.loc[m, 'Pvalue']) if m in pvals_qlike.index else 0.0
            in_mcs_qlike = 'Yes' if p_qlike >= 0.10 else 'No'
            
            mean_mse = float(np.mean(losses_mse[m]))
            mean_qlike = float(np.mean(losses_qlike[m]))
            
            mcs_records.append({
                'Horizon': f'h={h_val}',
                'Model': m,
                'Mean_MSE': f"{mean_mse:.6e}",
                'MCS_pvalue_MSE': round(p_mse, 4),
                'In_MCS_MSE_90pct': in_mcs_mse,
                'Mean_QLIKE': round(mean_qlike, 6),
                'MCS_pvalue_QLIKE': round(p_qlike, 4),
                'In_MCS_QLIKE_90pct': in_mcs_qlike
            })
            
    df_mcs = pd.DataFrame(mcs_records)
    out_mcs = AUDIT_OUT / "task7_mcs_results.csv"
    df_mcs.to_csv(out_mcs, index=False)
    print(f"\nSaved task7_mcs_results.csv:")
    print(df_mcs.to_string(index=False))
    
    # Document Bonferroni families
    bonf_records = [
        {
            'Panel': 'Current Panel A: DL Models vs Classical Baselines',
            'm_size': 12,
            'Hypotheses_Tested': '4 DL models (TFT-SPILL, TFT-NOSPILL, NBX-SPILL, NBX-NOSPILL) x 3 Baselines (Hist-Exp, Roll-20, HAR)',
            'Alpha_Nominal': 0.05,
            'Alpha_Adjusted': round(0.05 / 12, 6),
            'Critique': 'Ad hoc partitioning: separates ablation from baseline benchmarks'
        },
        {
            'Panel': 'Current Panel B: Ablation SPILL vs NOSPILL',
            'm_size': 2,
            'Hypotheses_Tested': '2 comparisons (TFT-SPILL vs TFT-NOSPILL, NBX-SPILL vs NBX-NOSPILL) per horizon',
            'Alpha_Nominal': 0.05,
            'Alpha_Adjusted': round(0.05 / 2, 6),
            'Critique': 'Restricted family artificially boosts power for ablation test relative to Panel A'
        },
        {
            'Panel': 'Unified Benchmark Family (All vs HAR Benchmark)',
            'm_size': 8,
            'Hypotheses_Tested': '8 candidate models vs HAR benchmark per horizon',
            'Alpha_Nominal': 0.05,
            'Alpha_Adjusted': round(0.05 / 8, 6),
            'Critique': 'Standard Dunnett/many-to-one benchmark family'
        },
        {
            'Panel': 'Full Pairwise Family (All-Pairs)',
            'm_size': 36,
            'Hypotheses_Tested': 'All 9x8/2 = 36 pairwise model comparisons',
            'Alpha_Nominal': 0.05,
            'Alpha_Adjusted': round(0.05 / 36, 6),
            'Critique': 'Conservative Bonferroni for all pairs; overcome by Model Confidence Set (MCS)'
        }
    ]
    df_bonf = pd.DataFrame(bonf_records)
    out_bonf = AUDIT_OUT / "task7_bonferroni_families.csv"
    df_bonf.to_csv(out_bonf, index=False)
    print(f"\nSaved task7_bonferroni_families.csv")

if __name__ == '__main__':
    main()

