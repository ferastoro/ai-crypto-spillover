"""
========================================================================================
retrain_loss_matched_kaggle.py
========================================================================================
Tujuan:
Re-training loss-matched ablation experiment pada Kaggle GPU environment:
- TFT dilatih dengan MAE loss (sebelumnya MSE)
- N-BEATSx dilatih dengan MSE loss (sebelumnya MAE)
Untuk mengisolasi efek arsitektur model dari fungsi loss (loss-architecture unconfounding).

5 Seeds: [42, 7, 13, 21, 99]
2 Feature Sets: NOSPILL (11 fitur), SPILL (20 fitur)
Rolling evaluation: 335 cutoffs, h=1 dan h=10.
Dataset path di Kaggle: /kaggle/input/datasets/ferastoro/ai-stocks-crypto-and-ihsg/data_clean
========================================================================================
"""

import os
import sys
import time
import json
import random
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT, NBEATSx
from neuralforecast.losses.pytorch import MAE, MSE
from pytorch_lightning.callbacks import Callback

# Deteksi direktori Kaggle vs Lokal
KAGGLE_DIR = Path("/kaggle/input/datasets/ferastoro/ai-stocks-crypto-and-ihsg/data_clean")
LOCAL_DIR = Path("./data/clean")
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else LOCAL_DIR
OUTPUT_DIR = Path("./audit_outputs/kaggle_retrain_results") if not Path("/kaggle/working").exists() else Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99]
HORIZON = 10
N_CUTOFFS = 335

NOSPILL_COLS = [
    'ret_saham_nvda', 'ret_saham_msft', 'ret_saham_tsla', 'ret_saham_googl', 'ret_saham_meta',
    'ret_crypto_btc_usd', 'ret_crypto_eth_usd', 'crypto_btc_usd_weekend_return',
    'gap_days', 'is_monday', 'is_ihsg_holiday'
]

SPILL_COLS = NOSPILL_COLS + [
    'spillover_ai_to_btcidr', 'tvp_tsi',
    'net_spill_saham_nvda', 'net_spill_saham_msft', 'net_spill_saham_tsla', 'net_spill_saham_googl', 'net_spill_saham_meta',
    'net_spill_crypto_btc_usd', 'net_spill_crypto_eth_usd'
]

class RestoreBestWeightsCallback(Callback):
    """Menyimpan weights terbaik pada validasi dan memulihkannya setelah fit selesai."""
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.best_step = 0
        
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss', metrics.get('ptl/val_loss', None))
        if val_loss is not None:
            val_loss = float(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_step = trainer.global_step
                self.best_state_dict = copy.deepcopy(pl_module.state_dict())
                
    def on_train_end(self, trainer, pl_module):
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)

def prepare_data():
    df_dl = pd.read_csv(DATA_DIR / "dataset_deep_learning.csv")
    df_tvp = pd.read_csv(DATA_DIR / "tvp_var_dynamic_features.csv")
    
    df_merged = pd.merge(df_dl, df_tvp, on='Date', how='left').sort_values('Date').reset_index(drop=True)
    df_merged = df_merged.ffill().bfill()
    
    # Target r_t^2
    df_merged['y'] = df_merged['ret_crypto_btc_idr_indodax'] ** 2
    df_merged['unique_id'] = 'BTC_IDR'
    df_merged['ds'] = pd.to_datetime(df_merged['Date'])
    
    return df_merged

def run_experiment():
    print("=" * 80)
    print("STARTING LOSS-MATCHED RE-TRAINING (TFT with MAE, NBEATSx with MSE)")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    df = prepare_data()
    all_dates = df['ds'].values
    total_len = len(df)
    
    # 1165 train + 167 val = 1332 fit window
    train_val_len = 1332
    test_len = total_len - train_val_len
    
    results_records = []
    
    # Model Configurations (Loss matched: TFT -> MAE, NBEATSx -> MSE)
    configs = [
        ('TFT', 'NOSPILL', 'MAE', {'input_size': 50, 'learning_rate': 0.001, 'hidden_size': 64}, NOSPILL_COLS),
        ('TFT', 'SPILL', 'MAE', {'input_size': 50, 'learning_rate': 0.001, 'hidden_size': 64}, SPILL_COLS),
        ('NBEATSx', 'NOSPILL', 'MSE', {'input_size': 20, 'learning_rate': 0.0003, 'mlp_units': [[256, 256], [256, 256], [256, 256]]}, NOSPILL_COLS),
        ('NBEATSx', 'SPILL', 'MSE', {'input_size': 20, 'learning_rate': 0.0003, 'mlp_units': [[256, 256], [256, 256], [256, 256]]}, SPILL_COLS)
    ]
    
    for model_name, fset_name, loss_name, hp, feat_cols in configs:
        print(f"\n>>> Running Config: {model_name} | {fset_name} | Inverted Loss: {loss_name}")
        loss_fn = MAE() if loss_name == 'MAE' else MSE()
        
        seed_predictions = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            cb = RestoreBestWeightsCallback()
            
            if model_name == 'TFT':
                model = TFT(
                    h=HORIZON,
                    input_size=hp['input_size'],
                    learning_rate=hp['learning_rate'],
                    hidden_size=hp['hidden_size'],
                    loss=loss_fn,
                    max_steps=1000,
                    val_check_steps=25,
                    early_stop_patience_steps=5,
                    futr_exog_list=feat_cols,
                    callbacks=[cb],
                    random_seed=seed
                )
            else:
                model = NBEATSx(
                    h=HORIZON,
                    input_size=hp['input_size'],
                    learning_rate=hp['learning_rate'],
                    mlp_units=hp['mlp_units'],
                    loss=loss_fn,
                    max_steps=1000,
                    val_check_steps=25,
                    early_stop_patience_steps=5,
                    futr_exog_list=feat_cols,
                    callbacks=[cb],
                    random_seed=seed
                )
                
            nf = NeuralForecast(models=[model], freq='D')
            
            # Cross validation over 335 cutoffs
            t0 = time.time()
            cv_df = nf.cross_validation(
                df=df[['unique_id', 'ds', 'y'] + feat_cols],
                n_windows=N_CUTOFFS,
                step_size=1,
                val_size=167
            )
            elapsed = time.time() - t0
            print(f"  Seed {seed} finished in {elapsed:.1f}s")
            
            pred_col = [c for c in cv_df.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']][0]
            cv_df = cv_df.rename(columns={pred_col: f'pred_s{seed}'})
            seed_predictions.append(cv_df[['unique_id', 'ds', 'cutoff', 'y', f'pred_s{seed}']])
            
        # Ensemble across 5 seeds
        ens_df = seed_predictions[0].copy()
        for i in range(1, len(SEEDS)):
            ens_df = pd.merge(ens_df, seed_predictions[i], on=['unique_id', 'ds', 'cutoff', 'y'])
            
        pred_cols = [f'pred_s{s}' for s in SEEDS]
        ens_df['ensemble_pred'] = ens_df[pred_cols].mean(axis=1)
        ens_df['h'] = ens_df.groupby('cutoff').cumcount() + 1
        
        # Save predictions
        ens_df.to_csv(OUTPUT_DIR / f"loss_matched_{model_name}_{fset_name}_{loss_name}_preds.csv", index=False)
        
        for h_eval in [1, 10]:
            sub = ens_df[ens_df['h'] == h_eval]
            yt = sub['y'].values
            yp = sub['ensemble_pred'].values
            
            mse = float(np.mean((yt - yp) ** 2))
            mae = float(np.mean(np.abs(yt - yp)))
            
            # Hist-Exp benchmark for R2_OOS
            # (hist_exp MSE: h=1 is 2.689570e-6, h=10 is 2.686277e-6)
            mse_bench = 2.689570e-6 if h_eval == 1 else 2.686277e-6
            r2_oos = 1.0 - (mse / mse_bench)
            neg_pct = float((yp < 0).mean() * 100.0)
            
            results_records.append({
                'Model': model_name,
                'FeatureSet': fset_name,
                'Inverted_Loss': loss_name,
                'Horizon': f'h={h_eval}',
                'MSE': mse,
                'MAE': mae,
                'R2_OOS': r2_oos,
                'Negative_Pct': f"{neg_pct:.2f}%"
            })
            
    df_res = pd.DataFrame(results_records)
    df_res.to_csv(OUTPUT_DIR / "loss_matched_evaluation_metrics.csv", index=False)
    print("\n" + "=" * 80)
    print("FINAL LOSS-MATCHED EVALUATION RESULTS:")
    print(df_res.to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    run_experiment()

