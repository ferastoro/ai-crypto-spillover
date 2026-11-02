"""
========================================================================================
kaggle_refit_60days_expanding.py
========================================================================================
Tujuan:
Eksekusi 60-Day Expanding Window Periodic Refitting untuk model Deep Learning:
- TFT (input_size=50, lr=0.001, hidden_size=64, loss=MSE)
- N-BEATSx (input_size=20, lr=0.0003, mlp_units=3x[256, 256], loss=MAE)
- Feature set: SPILL (20 fitur) & NOSPILL (11 fitur)
- Seed: 42 saja (sesuai instruksi audit)
- Horizons: h=1 dan h=10 across 335 test cutoffs
- Evaluasi metrik MSE, MAE, QLIKE vs benchmark HAR

Mekanisme Expanding Window Refitting:
- Cutoff 0 s.d. 59: Model dilatih pada data awal (Train 1,165 + Val 167 = 1,332 hari).
- Cutoff 60 s.d. 119: Window training bertambah +60 hari (1,392 hari), model di-refit.
- Cutoff 120 s.d. 179: Window training bertambah +60 hari (1,452 hari), model di-refit.
- Cutoff 180 s.d. 239: Window training bertambah +60 hari (1,512 hari), model di-refit.
- Cutoff 240 s.d. 299: Window training bertambah +60 hari (1,572 hari), model di-refit.
- Cutoff 300 s.d. 334: Window training bertambah +60 hari (1,632 hari), model di-refit.

Jalankan di Kaggle GPU (P100 / T4 x2) dengan dataset:
/kaggle/input/datasets/ferastoro/ai-stocks-crypto-and-ihsg/data_clean
========================================================================================
"""

import os
import sys
import time
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
OUTPUT_DIR = Path("./audit_outputs/followup") if not Path("/kaggle/working").exists() else Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HORIZON = 10
N_CUTOFFS = 335
REFIT_STEP = 60

# Fitur
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

def compute_qlike(y_true, y_pred, eps=1e-6):
    yt = np.array(y_true, dtype=float)
    yp = np.maximum(np.array(y_pred, dtype=float), eps)
    yt_safe = np.maximum(yt, 1e-12)
    ratio = yt_safe / yp
    return ratio - np.log(ratio) - 1.0

def prepare_data():
    df_dl = pd.read_csv(DATA_DIR / "dataset_deep_learning.csv")
    df_tvp = pd.read_csv(DATA_DIR / "tvp_var_dynamic_features.csv")
    df_merged = pd.merge(df_dl, df_tvp, on='Date', how='left').sort_values('Date').reset_index(drop=True)
    df_merged = df_merged.ffill().bfill()
    df_merged['y'] = df_merged['ret_crypto_btc_idr_indodax'] ** 2
    df_merged['unique_id'] = 'BTC_IDR'
    df_merged['ds'] = pd.to_datetime(df_merged['Date'])
    return df_merged

def build_model(model_name: str, feature_set: str, futr_cols: list):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    cb = RestoreBestWeightsCallback()
    
    if model_name == 'TFT':
        return TFT(
            h=HORIZON,
            input_size=50,
            hidden_size=64,
            learning_rate=0.001,
            loss=MSE(),
            futr_exog_list=futr_cols,
            max_steps=1000,
            val_check_steps=50,
            early_stop_patience_steps=10,
            callbacks=[cb],
            random_seed=SEED
        )
    elif model_name == 'NBEATSx':
        return NBEATSx(
            h=HORIZON,
            input_size=20,
            learning_rate=0.0003,
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            loss=MAE(),
            futr_exog_list=futr_cols,
            max_steps=1000,
            val_check_steps=50,
            early_stop_patience_steps=10,
            callbacks=[cb],
            random_seed=SEED
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_expanding_refit():
    print("=" * 80)
    print("RUNNING 60-DAY EXPANDING WINDOW REFITTING EXPERIMENT")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    df = prepare_data()
    T_total = len(df)
    # T0 initial split: 1165 train + 167 val = 1332 days
    base_split = 1332
    test_len = 335
    
    # Block definitions: [0..59], [60..119], [120..179], [180..239], [240..299], [300..334]
    blocks = []
    for b_idx, start_c in enumerate(range(0, test_len, REFIT_STEP)):
        end_c = min(start_c + REFIT_STEP, test_len)
        blocks.append((b_idx, start_c, end_c))
        
    print(f"Total expanding blocks: {len(blocks)}")
    for b_idx, start_c, end_c in blocks:
        print(f"  Block {b_idx + 1}: Cutoffs {start_c} to {end_c - 1} (N = {end_c - start_c}) | Train size = {base_split + start_c} days")
        
    model_configs = [
        ('TFT', 'SPILL', SPILL_COLS),
        ('TFT', 'NOSPILL', NOSPILL_COLS),
        ('NBEATSx', 'SPILL', SPILL_COLS),
        ('NBEATSx', 'NOSPILL', NOSPILL_COLS)
    ]
    
    all_predictions = []
    
    for model_name, feat_name, exog_cols in model_configs:
        cfg_label = f"{model_name}_{feat_name}_REFIT60"
        print(f"\n>>> Running Config: {cfg_label} <<<")
        
        for b_idx, start_c, end_c in blocks:
            current_train_end = base_split + start_c
            df_train_current = df.iloc[:current_train_end].copy()
            
            val_size = 167
            train_size = len(df_train_current) - val_size
            
            print(f"  -> Fitting Block {b_idx + 1} (Cutoffs {start_c}..{end_c-1}) with {len(df_train_current)} historical days...")
            model = build_model(model_name, feat_name, exog_cols)
            nf = NeuralForecast(models=[model], freq='D')
            
            # Fit with validation
            nf.fit(df=df_train_current, val_size=val_size)
            
            # Predict for cutoffs in this block
            for c in range(start_c, end_c):
                cutoff_idx = base_split + c
                hist_window = df.iloc[:cutoff_idx].copy()
                
                # Exogenous future features for 10 steps ahead
                future_df = df.iloc[cutoff_idx : cutoff_idx + HORIZON].copy()
                
                fcst = nf.predict(df=hist_window)
                model_col = model_name
                preds = fcst[model_col].values[:HORIZON]
                
                for step_h in range(1, HORIZON + 1):
                    target_idx = cutoff_idx + step_h - 1
                    target_row = df.iloc[target_idx]
                    all_predictions.append({
                        'Config': cfg_label,
                        'Model': model_name,
                        'Feature_Set': feat_name,
                        'Block': b_idx + 1,
                        'Cutoff': c,
                        'h': step_h,
                        'Cutoff_Date': df.iloc[cutoff_idx - 1]['Date'],
                        'Target_Date': target_row['Date'],
                        'True_y': float(target_row['y']),
                        'Prediction': float(preds[step_h - 1])
                    })
                    
    df_all_preds = pd.DataFrame(all_predictions)
    df_all_preds.to_csv(OUTPUT_DIR / "refit_60days_expanding_predictions.csv", index=False)
    print("\nSaved refit_60days_expanding_predictions.csv")
    
    # Evaluation vs HAR
    # Load HAR baseline
    try:
        har_h1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values('Date').reset_index(drop=True)
        har_h10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values('Date').reset_index(drop=True)
        y_true_h1 = har_h1['True'].values
        y_true_h10 = har_h10['True'].values
        y_har_h1 = har_h1['HAR'].values
        y_har_h10 = har_h10['HAR'].values
        
        eval_records = []
        # Add HAR
        eval_records.append({
            'Config': 'HAR_Benchmark',
            'Horizon': 'h=1',
            'MSE': float(np.mean((y_true_h1 - y_har_h1)**2)),
            'MAE': float(np.mean(np.abs(y_true_h1 - y_har_h1))),
            'QLIKE': float(np.mean(compute_qlike(y_true_h1, y_har_h1)))
        })
        eval_records.append({
            'Config': 'HAR_Benchmark',
            'Horizon': 'h=10',
            'MSE': float(np.mean((y_true_h10 - y_har_h10)**2)),
            'MAE': float(np.mean(np.abs(y_true_h10 - y_har_h10))),
            'QLIKE': float(np.mean(compute_qlike(y_true_h10, y_har_h10)))
        })
        
        for cfg in df_all_preds['Config'].unique():
            sub = df_all_preds[df_all_preds['Config'] == cfg]
            for h_val in [1, 10]:
                sub_h = sub[sub['h'] == h_val].sort_values('Cutoff').reset_index(drop=True)
                yt = sub_h['True_y'].values
                yp = np.maximum(sub_h['Prediction'].values, 0.0)
                
                eval_records.append({
                    'Config': cfg,
                    'Horizon': f'h={h_val}',
                    'MSE': float(np.mean((yt - yp)**2)),
                    'MAE': float(np.mean(np.abs(yt - yp))),
                    'QLIKE': float(np.mean(compute_qlike(yt, yp)))
                })
                
        df_eval = pd.DataFrame(eval_records)
        df_eval.to_csv(OUTPUT_DIR / "refit_60days_expanding_vs_har.csv", index=False)
        print("\nSaved refit_60days_expanding_vs_har.csv:")
        print(df_eval.to_string(index=False))
    except Exception as e:
        print(f"Notice: HAR evaluation on Kaggle: {e}")

if __name__ == '__main__':
    run_expanding_refit()

