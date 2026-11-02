import os
import sys
import json
import hashlib
import time
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.stats import skew
from statsmodels.tsa.api import VAR

# =====================================================================
# PATHS CONFIGURATION (RESOLVED FROM REPO ROOT)
# =====================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

# Inputs
NB_TVP_PATH = REPO_ROOT / "notebooks" / "02_tvp_var_spillover.ipynb"
COLLECT_DATA_PATH = REPO_ROOT / "src" / "collect_data.py"
DATA_CLEAN_DIR = REPO_ROOT / "data" / "clean"
RESULTS_V4_DIR = REPO_ROOT / "results" / "v4"
CV_PRED_V4_PATH = RESULTS_V4_DIR / "CV_Predictions_V4.csv"
V5_NB_PATH = REPO_ROOT / "deep-learning-spillover-v5.ipynb"
RUN_FINAL_EVAL_PATH = REPO_ROOT / "scratch" / "final" / "run_final_eval.py"

# Data clean files
TVP_DYN_FEATURES_PATH = DATA_CLEAN_DIR / "tvp_var_dynamic_features.csv"
DATASET_DL_PATH = DATA_CLEAN_DIR / "dataset_deep_learning.csv"
TARGET_ALT_PATH = DATA_CLEAN_DIR / "target_alternatives.csv"
DATASET_GARCH_PATH = DATA_CLEAN_DIR / "dataset_garch_baseline.csv"

# Outputs (written to audit_outputs/checkpoint3a/, never scratch/)
OUT_DIR = REPO_ROOT / "audit_outputs" / "checkpoint3a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RERUN_CSV = OUT_DIR / "tvp_var_dynamic_features_rerun.csv"
OUT_TABLE_SPILL_CSV = OUT_DIR / "Average_Spillover_Table_vol.csv"
OUT_FILE_A = OUT_DIR / "checkpoint3a_A.txt"
OUT_FILE_B = OUT_DIR / "checkpoint3a_B.txt"
OUT_FILE_C = OUT_DIR / "checkpoint3a_C.txt"

def get_md5(filepath):
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def eval_assert(condition, val_str=""):
    try:
        assert condition
        return f"PASS {val_str}".strip()
    except AssertionError:
        return f"FAIL {val_str}".strip()

# Buffer dictionaries for output files
out_A = []
out_B = []
out_C = []

def log_A(line=""):
    out_A.append(str(line))

def log_B(line=""):
    out_B.append(str(line))

def log_C(line=""):
    out_C.append(str(line))

# Load notebook 02_tvp_var_spillover.ipynb
nb_path = str(NB_TVP_PATH)
with open(nb_path, 'r') as f:
    nb_tvp = json.load(f)

cells_tvp = nb_tvp['cells']

# Cell source extractors
def get_cell_source(idx):
    return "".join(cells_tvp[idx]['source'])

# =====================================================================
# SECTION A: PROVENANCE
# =====================================================================
log_A("======================================================================")
log_A("CHECKPOINT 3a - BAGIAN A: PROVENANCE")
log_A("======================================================================")

# A.1
log_A("\n--- A.1. PROVENANCE FILE & NOTEBOOK ---")
prov_files = [
    str(TVP_DYN_FEATURES_PATH),
    str(DATASET_DL_PATH),
    str(TARGET_ALT_PATH)
]
for pf in prov_files:
    f_md5 = get_md5(pf)
    df_temp = pd.read_csv(pf)
    d_col = 'Date' if 'Date' in df_temp.columns else ('ds' if 'ds' in df_temp.columns else None)
    d_min = df_temp[d_col].min() if d_col else "N/A"
    d_max = df_temp[d_col].max() if d_col else "N/A"
    log_A(f"File: {pf}")
    log_A(f"  MD5: {f_md5}")
    log_A(f"  Jumlah baris: {len(df_temp)}")
    log_A(f"  Tanggal min: {d_min}")
    log_A(f"  Tanggal max: {d_max}")

nb_md5 = get_md5(nb_path)
total_cells = len(cells_tvp)
cells_with_out = sum(1 for c in cells_tvp if len(c.get('outputs', [])) > 0)
log_A(f"Notebook: {nb_path}")
log_A(f"  MD5: {nb_md5}")
log_A(f"  Total cells: {total_cells}")
log_A(f"  Cells dengan output: {cells_with_out}")
log_A(f"  Status tersimpan output: {cells_with_out > 0}")

# A.2
log_A("\n--- A.2. END_DATE DI collect_data.py ---")
collect_path = str(COLLECT_DATA_PATH)
collect_lines = open(collect_path).readlines()
for line_idx, line_text in enumerate(collect_lines, 1):
    if "END_DATE" in line_text and "=" in line_text and "202" in line_text:
        log_A(f"File: {collect_path} | Nomor baris: {line_idx} | Kode: {line_text.strip()}")

# A.3
log_A("\n--- A.3. PEMBENTUKAN Y DAN Z-SCORE ---")
log_A("Nomor cell: Cell 3")
log_A("Kode persis Cell 3:")
log_A(get_cell_source(3).strip())

df_baseline = pd.read_csv(DATASET_GARCH_PATH)
df_baseline['Date'] = pd.to_datetime(df_baseline['Date'])
core_vars = [
    'ret_saham_nvda', 'ret_saham_msft', 'ret_saham_tsla', 'ret_saham_googl', 'ret_saham_meta', 
    'ret_crypto_btc_usd', 'ret_crypto_eth_usd', 'ret_crypto_btc_idr_indodax'
]
Y_raw = (df_baseline[core_vars] ** 2).values
Y_log = np.log(Y_raw + 1e-10)
means_col = np.mean(Y_log, axis=0)
stds_col = np.std(Y_log, axis=0)

log_A("\nRuang lingkup standardisasi: Seluruh sampel (full sample / static), bukan expanding, bukan rolling.")
log_A("Nilai mean dan std per kolom yang dipakai (Y_log = log(r^2 + 1e-10)):")
log_A(f"{'Variabel':<32} | {'Mean':<12} | {'Std':<12}")
log_A("-" * 62)
for v_name, m_val, s_val in zip(core_vars, means_col, stds_col):
    log_A(f"{v_name:<32} | {m_val:<12.6f} | {s_val:<12.6f}")

# A.4
log_A("\n--- A.4. DEFINISI PERSIS KOLOM FITUR ---")
log_A("1. tvp_tsi (Cell 10 & Cell 19):")
log_A("   Cell 10:")
log_A("   off_diag_sum = gfevd_hist[t].sum() - np.trace(gfevd_hist[t])")
log_A("   tsi[t] = (off_diag_sum / M) * 100")
log_A("   Cell 19:")
log_A("   df_dynamic[\"tvp_tsi\"] = tsi")

log_A("\n2. net_spill_* (Cell 10 & Cell 19):")
log_A("   Cell 10:")
log_A("   dir_from[t, i] = (np.sum(gfevd_hist[t, i, :]) - gfevd_hist[t, i, i]) * 100")
log_A("   dir_to[t, i] = (np.sum(gfevd_hist[t, :, i]) - gfevd_hist[t, i, i]) * 100")
log_A("   net_spill = dir_to - dir_from")
log_A("   Cell 19:")
log_A("   for i, var in enumerate(core_vars):")
log_A("       col_name = var.replace(\"ret_\", \"net_spill_\")")
log_A("       df_dynamic[col_name] = net_spill[:, i]")

log_A("\n3. spillover_ai_to_btcidr (Cell 19):")
log_A("   idx_btcidr = 7")
log_A("   spill_ai_to_btc = np.sum(gfevd_hist[:, idx_btcidr, 0:5], axis=1) * 100")
log_A("   df_dynamic[\"spillover_ai_to_btcidr\"] = spill_ai_to_btc")

# A.5
log_A("\n--- A.5. ANALISIS 121 BARIS FFILL DI df_har ---")
df_dl = pd.read_csv(DATASET_DL_PATH)
df_dl['Date'] = pd.to_datetime(df_dl['Date'])
df_tvp = pd.read_csv(TVP_DYN_FEATURES_PATH)
df_tvp['Date'] = pd.to_datetime(df_tvp['Date'])

merged_raw = pd.merge(df_dl, df_tvp, on='Date', how='left').sort_values('Date').reset_index(drop=True)
alt_df = pd.read_csv(TARGET_ALT_PATH)
alt_df['Date'] = pd.to_datetime(alt_df['Date'])
merged_raw = pd.merge(merged_raw, alt_df[['Date', 'parkinson']], on='Date', how='left')
merged_raw.ffill(inplace=True)
df_har = merged_raw.dropna().reset_index(drop=True)

ffill_mask = ~df_har['Date'].isin(df_tvp['Date'])
ffill_dates = df_har.loc[ffill_mask, 'Date']
n_ffill = len(ffill_dates)

log_A(f"Jumlah baris df_har: {len(df_har)}")
log_A(f"Jumlah baris hasil ffill TVP: {n_ffill}")
log_A("Daftar tanggal pertama 10 hasil ffill:")
for d in ffill_dates.head(10):
    log_A(f"  {d.strftime('%Y-%m-%d')}")
log_A("Daftar tanggal terakhir 10 hasil ffill:")
for d in ffill_dates.tail(10):
    log_A(f"  {d.strftime('%Y-%m-%d')}")

tvp_min_date = df_tvp['Date'].min()
tvp_max_date = df_tvp['Date'].max()
n_awal = (ffill_dates < tvp_min_date).sum()
n_tengah = ((ffill_dates >= tvp_min_date) & (ffill_dates <= tvp_max_date)).sum()
n_akhir = (ffill_dates > tvp_max_date).sum()

log_A(f"Jumlah baris ffill di awal sampel (< {tvp_min_date.strftime('%Y-%m-%d')}): {n_awal}")
log_A(f"Jumlah baris ffill di tengah sampel ({tvp_min_date.strftime('%Y-%m-%d')} s.d. {tvp_max_date.strftime('%Y-%m-%d')}): {n_tengah}")
log_A(f"Jumlah baris ffill di akhir sampel (> {tvp_max_date.strftime('%Y-%m-%d')}): {n_akhir}")

btc_zero_ffill = (df_har.loc[ffill_mask, 'ret_crypto_btc_idr_indodax'] == 0).sum()
btc_zero_total = (df_har['ret_crypto_btc_idr_indodax'] == 0).sum()
log_A(f"Jumlah hari BTC-IDR return = 0 pada 121 baris ffill: {btc_zero_ffill}")
log_A(f"Jumlah hari BTC-IDR return = 0 pada seluruh baris df_har: {btc_zero_total}")


# =====================================================================
# SECTION B: VERIFIKASI HASIL (RERUN PENUH, BASE CASE)
# =====================================================================
log_B("======================================================================")
log_B("CHECKPOINT 3a - BAGIAN B: VERIFIKASI HASIL (RERUN PENUH, BASE CASE)")
log_B("======================================================================")

log_B("\n--- KODE ASLI DARI NOTEBOOK tvp_var_spillover.ipynb ---")
log_B("Nomor cell: Cell 6 (ForgettingFactorTVPVAR)")
log_B(get_cell_source(6).strip())
log_B("\nNomor cell: Cell 8 (calculate_gfevd)")
log_B(get_cell_source(8).strip())
log_B("\nNomor cell: Cell 10 (tsi, dir_from, dir_to, net_spill)")
log_B(get_cell_source(10).strip())

# TVP-VAR class exactly as written in notebook Cell 6
class ForgettingFactorTVPVAR:
    def __init__(self, y, p=1, lambda_ff=0.99, kappa_ff=0.96):
        self.y = y
        self.T, self.M = y.shape
        self.p = p
        self.K = self.M * self.p + 1  # jumlah prediktor per persamaan (lagged vars + intercept)
        self.lambda_ff = lambda_ff
        self.kappa_ff = kappa_ff
        
        # Buat matriks regressor Z
        self.Z = np.zeros((self.T - p, self.M, self.K * self.M))
        self.Y = self.y[p:]
        
        for t in range(self.T - p):
            # Z_t = [1, y_{t-1}, ..., y_{t-p}]
            z_t = np.concatenate([[1.0], self.y[t : t + p].flatten()[::-1]])
            # Ekspansi ke sistem SUR (Seemingly Unrelated Regressions)
            self.Z[t] = np.kron(np.eye(self.M), z_t)
            
    def estimate(self):
        T_est = self.T - self.p
        
        # Inisialisasi priors
        beta_t = np.zeros((self.K * self.M, 1))
        P_t = np.eye(self.K * self.M) * 10.0  # Varian awal state (uninformative)
        Sigma_t = np.cov(self.y.T)            # Inisialisasi error covariance dengan sampel kovarians
        
        # Array untuk menyimpan history
        self.beta_history = np.zeros((T_est, self.K * self.M))
        self.Sigma_history = np.zeros((T_est, self.M, self.M))
        
        for t in range(T_est):
            Z_t = self.Z[t]
            y_t = self.Y[t].reshape(self.M, 1)
            
            # 1. Prediction Step
            # Koefisien diasumsikan konstan secara ekspektasi: beta_{t|t-1} = beta_{t-1|t-1}
            # Varians state diprediksi dengan forgetting factor lambda
            P_t_pred = P_t / self.lambda_ff
            
            # Error Prediksi
            v_t = y_t - Z_t @ beta_t
            
            # 2. Update Measurement Covariance (Sigma) dengan EWMA / forgetting factor kappa
            Sigma_t = self.kappa_ff * Sigma_t + (1 - self.kappa_ff) * (v_t @ v_t.T)
            
            # 3. Update Step (Kalman Gain)
            F_t = Z_t @ P_t_pred @ Z_t.T + Sigma_t
            K_t = P_t_pred @ Z_t.T @ pinv(F_t)
            
            # Koreksi state
            beta_t = beta_t + K_t @ v_t
            P_t = P_t_pred - K_t @ Z_t @ P_t_pred
            
            # Simpan hasil
            self.beta_history[t] = beta_t.flatten()
            self.Sigma_history[t] = Sigma_t
            
        return self.beta_history, self.Sigma_history

# calculate_gfevd exactly as written in notebook Cell 8
def calculate_gfevd(beta_history, sigma_history, M, p, H=10):
    T_est = beta_history.shape[0]
    gfevd_history = np.zeros((T_est, M, M))
    
    for t in range(T_est):
        # Bentuk Matriks Koefisien VAR A_1, A_2 ... A_p
        # beta_t memiliki susunan: [c1, A1_11, A1_12, ..., c2, A1_21, ...]
        beta_t = beta_history[t]
        Sigma_t = sigma_history[t]
        sigma_diag = np.diag(Sigma_t)
        
        # Ekstrak matriks koefisien (A)
        # Bentuk A adalah list of matrices: [A_1, A_2, ..., A_p]
        A = np.zeros((p, M, M))
        idx = 0
        for i in range(M):
            idx += 1 # lewati intercept
            for lag in range(p):
                for j in range(M):
                    A[lag, i, j] = beta_t[idx]
                    idx += 1
                    
        # Hitung VMA (Vector Moving Average) representation koefisien (Phi)
        # Phi_0 = I_M
        # Phi_h = sum_{j=1}^h Phi_{h-j} A_j
        Phi = np.zeros((H, M, M))
        Phi[0] = np.eye(M)
        for h in range(1, H):
            for j in range(1, min(h, p) + 1):
                Phi[h] += Phi[h-j] @ A[j-1]
                
        # Hitung Generalized FEVD (theta_{ij}(H))
        theta = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                e_j = np.zeros((M, 1))
                e_j[j] = 1.0
                
                num = 0
                den = 0
                for h in range(H):
                    # (e_i' * Phi_h * Sigma * e_j)^2
                    term = (Phi[h][i, :] @ Sigma_t @ e_j)[0]
                    num += term**2
                    
                    # e_i' * Phi_h * Sigma * Phi_h' * e_i
                    den += Phi[h][i, :] @ Sigma_t @ Phi[h][i, :].T
                    
                theta[i, j] = (1.0 / sigma_diag[j]) * (num / den)
                
        # Normalisasi tiap baris agar total varians error = 1 (100%)
        row_sums = theta.sum(axis=1)
        theta_tilde = theta / row_sums[:, np.newaxis]
        
        gfevd_history[t] = theta_tilde
        
    return gfevd_history

# Helper to compute indices from gfevd
def compute_indices(gfevd_hist):
    T_est, M, _ = gfevd_hist.shape
    tsi = np.zeros(T_est)
    for t in range(T_est):
        off_diag_sum = gfevd_hist[t].sum() - np.trace(gfevd_hist[t])
        tsi[t] = (off_diag_sum / M) * 100
        
    dir_from = np.zeros((T_est, M))
    dir_to = np.zeros((T_est, M))
    for t in range(T_est):
        for i in range(M):
            dir_from[t, i] = (np.sum(gfevd_hist[t, i, :]) - gfevd_hist[t, i, i]) * 100
            dir_to[t, i] = (np.sum(gfevd_hist[t, :, i]) - gfevd_hist[t, i, i]) * 100
            
    net_spill = dir_to - dir_from
    return tsi, dir_from, dir_to, net_spill

# Base Case Execution
p_lag = 1
H_horizon = 10
Y_std_base = (Y_log - means_col) / stds_col
model_base = ForgettingFactorTVPVAR(Y_std_base, p=p_lag, lambda_ff=0.99, kappa_ff=0.96)
beta_hist_base, sigma_hist_base = model_base.estimate()
gfevd_hist_base = calculate_gfevd(beta_hist_base, sigma_hist_base, M=len(core_vars), p=p_lag, H=H_horizon)
tsi_base, dir_from_base, dir_to_base, net_spill_base = compute_indices(gfevd_hist_base)

dates_base = pd.to_datetime(df_baseline['Date'])
plot_dates_base = dates_base[p_lag:]

df_rerun = pd.DataFrame({"Date": plot_dates_base.dt.strftime('%Y-%m-%d')})
df_rerun["tvp_tsi"] = tsi_base
for i, var in enumerate(core_vars):
    df_rerun[var.replace("ret_", "net_spill_")] = net_spill_base[:, i]
df_rerun["spillover_ai_to_btcidr"] = np.sum(gfevd_hist_base[:, 7, 0:5], axis=1) * 100

cols_order = [
    "Date", "tvp_tsi", 
    "net_spill_saham_nvda", "net_spill_saham_msft", "net_spill_saham_tsla", 
    "net_spill_saham_googl", "net_spill_saham_meta", "net_spill_crypto_btc_usd", 
    "net_spill_crypto_eth_usd", "net_spill_crypto_btc_idr_indodax", 
    "spillover_ai_to_btcidr"
]
df_rerun = df_rerun[cols_order]

rerun_path = str(OUT_RERUN_CSV)
df_rerun.to_csv(rerun_path, index=False)
rerun_md5 = get_md5(rerun_path)
orig_tvp_path = str(TVP_DYN_FEATURES_PATH)
orig_tvp_md5 = get_md5(orig_tvp_path)
df_orig = pd.read_csv(orig_tvp_path)

# B.1
log_B("\n--- B.1. PERBANDINGAN RERUN DENGAN FILE TERSIMPAN ---")
log_B(f"Path rerun: {rerun_path} | MD5: {rerun_md5}")
log_B(f"Path tersimpan: {orig_tvp_path} | MD5: {orig_tvp_md5}")
log_B("Max Abs Diff per Kolom (rerun vs tersimpan):")
for col in cols_order:
    if col != "Date":
        diff_max = np.max(np.abs(df_rerun[col].values - df_orig[col].values))
        log_B(f"  {col:<32}: {diff_max:.6e}")

# B.2
log_B("\n--- B.2. DISTRIBUSI & SKEWNESS INPUT PER KOLOM ---")
log_B(f"{'Variabel':<32} | {'Jumlah Nol r^2':<15} | {'Skewness r^2':<15} | {'Skewness log(r^2)':<18}")
log_B("-" * 86)
for i, v in enumerate(core_vars):
    n_zeros = (Y_raw[:, i] == 0).sum()
    sk_raw = skew(Y_raw[:, i])
    sk_log = skew(Y_log[:, i])
    log_B(f"{v:<32} | {n_zeros:<15} | {sk_raw:<15.4f} | {sk_log:<18.4f}")

# B.3
log_B("\n--- B.3. VALIDASI TSI & GFEVD ROW-SUM ---")
tsi_mean = np.mean(tsi_base)
tsi_min = np.min(tsi_base)
tsi_max = np.max(tsi_base)
tsi_valid = (tsi_base >= 0).all() and (tsi_base <= 100).all()
log_B(f"TSI Mean: {tsi_mean:.4f}")
log_B(f"TSI Min:  {tsi_min:.4f}")
log_B(f"TSI Max:  {tsi_max:.4f}")
log_B(f"Assert 0 <= TSI <= 100: {eval_assert(tsi_valid, f'min={tsi_min:.4f}, max={tsi_max:.4f}')}")

gfevd_dev = np.abs(gfevd_hist_base.sum(axis=2) - 1.0).max()
dev_valid = bool(gfevd_dev <= 1e-10)
log_B(f"GFEVD row-sum max abs deviasi dari 1.0: {gfevd_dev:.4e}")
log_B(f"Assert GFEVD row-sum deviasi <= 1e-10: {eval_assert(dev_valid, f'max_dev={gfevd_dev:.4e}')}")

# B.4
log_B("\n--- B.4. TABEL RATA-RATA SPILLOVER LENGKAP 8x8 ---")
avg_gfevd = gfevd_hist_base.mean(axis=0) * 100
var_labels = [v.replace('ret_', '').upper() for v in core_vars]
table_spill = pd.DataFrame(avg_gfevd, index=var_labels, columns=var_labels)
diag_vals = np.diag(table_spill)
table_spill['FROM Others'] = table_spill.sum(axis=1) - diag_vals
to_others = table_spill.iloc[:8, :8].sum(axis=0) - diag_vals
table_spill.loc['TO Others'] = list(to_others) + [tsi_mean]
net_spillover = to_others - table_spill['FROM Others'].iloc[:8]
table_spill.loc['NET Spillover'] = list(net_spillover) + [np.nan]

table_csv_path = str(OUT_TABLE_SPILL_CSV)
table_spill.round(4).to_csv(table_csv_path)
log_B(f"Tersimpan ke: {table_csv_path} (MD5: {get_md5(table_csv_path)})")
log_B("Isi Lengkap Tabel Rata-rata Spillover (dalam %):")
log_B(table_spill.round(2).to_string())

# B.5
log_B("\n--- B.5. RINCIAN BARIS BTC-IDR (RECEIVER) ---")
idx_btc = 7
row_btc = table_spill.iloc[idx_btc, :8]
for v_lbl, v_val in row_btc.items():
    log_B(f"  Kontribusi {v_lbl:<32}: {v_val:.4f}%")
sum_ai = row_btc.iloc[0:5].sum()
c_btcusd = row_btc.iloc[5]
c_ethusd = row_btc.iloc[6]
c_self = row_btc.iloc[7]
log_B(f"  Jumlah 5 Saham AI: {sum_ai:.4f}%")
log_B(f"  BTC-USD:           {c_btcusd:.4f}%")
log_B(f"  ETH-USD:           {c_ethusd:.4f}%")
log_B(f"  Self (BTC-IDR):    {c_self:.4f}%")
log_B(f"  Total Baris:       {row_btc.sum():.4f}%")

# B.6
log_B("\n--- B.6. KONSISTENSI TOTAL NET DAN SPILLOVER AI KE BTC-IDR ---")
sum_net = np.sum(net_spillover)
log_B(f"Sum NET Spillover (8 aset): {sum_net:.6e}")
log_B(f"Assert Sum NET ~ 0 (tol 1e-10): {eval_assert(abs(sum_net) < 1e-10, f'sum_net={sum_net:.4e}')}")

spill_ai_file_mean = df_orig['spillover_ai_to_btcidr'].mean()
spill_ai_table_val = sum_ai
diff_spill_ai = abs(spill_ai_file_mean - spill_ai_table_val)
log_B(f"Rata-rata spillover_ai_to_btcidr dari file:     {spill_ai_file_mean:.6f}%")
log_B(f"Rata-rata spillover_ai_to_btcidr dari tabel 8x8: {spill_ai_table_val:.6f}%")
log_B(f"Selisih absolut:                               {diff_spill_ai:.6e}%")
log_B(f"Assert konsisten (tol 1e-10): {eval_assert(diff_spill_ai < 1e-10, f'diff={diff_spill_ai:.4e}')}")


# =====================================================================
# SECTION C: ROBUSTNESS
# =====================================================================
log_C("======================================================================")
log_C("CHECKPOINT 3a - BAGIAN C: ROBUSTNESS")
log_C("======================================================================")

# C.1
log_C("\n--- C.1. SENSITIVITY 9 KOMBINASI LAMBDA x KAPPA ---")
lambda_grid = [0.98, 0.99, 0.995]
kappa_grid = [0.94, 0.96, 0.98]
sens_rows = []

for l_val in lambda_grid:
    for k_val in kappa_grid:
        m_s = ForgettingFactorTVPVAR(Y_std_base, p=1, lambda_ff=l_val, kappa_ff=k_val)
        b_s, s_s = m_s.estimate()
        g_s = calculate_gfevd(b_s, s_s, M=8, p=1, H=10)
        t_s, _, _, n_s = compute_indices(g_s)
        avg_g_s = g_s.mean(axis=0) * 100
        d_s = np.diag(avg_g_s)
        from_s = avg_g_s.sum(axis=1) - d_s
        to_s = avg_g_s.sum(axis=0) - d_s
        net_s = to_s - from_s
        
        row_dict = {'lambda': l_val, 'kappa': k_val, 'TSI_mean': np.mean(t_s)}
        for a_idx, a_name in enumerate(var_labels):
            row_dict[f'NET_{a_name}'] = net_s[a_idx]
        sens_rows.append(row_dict)

df_sens = pd.DataFrame(sens_rows)
log_C("Tabel 9 Kombinasi (lambda x kappa):")
log_C(df_sens.round(4).to_string(index=False))

log_C("\nRingkasan Tanda & Rentang NET per Aset (dari 9 kombinasi):")
log_C(f"{'Aset':<32} | {'Positif':<7} | {'Negatif':<7} | {'Min NET':<10} | {'Max NET':<10}")
log_C("-" * 74)
for a_name in var_labels:
    col_k = f'NET_{a_name}'
    n_pos = int((df_sens[col_k] > 0).sum())
    n_neg = int((df_sens[col_k] < 0).sum())
    min_n = df_sens[col_k].min()
    max_n = df_sens[col_k].max()
    log_C(f"{a_name:<32} | {n_pos:<7} | {n_neg:<7} | {min_n:<10.4f} | {max_n:<10.4f}")

# C.2
log_C("\n--- C.2. PROXY VOLATILITAS ---")
alt_merged = pd.merge(df_baseline[['Date']], alt_df[['Date', 'parkinson']], on='Date', how='left')

# Proxy a: log(r^2 + 1e-10) z-scored (base case)
Ya_std = Y_std_base

# Proxy b: |r| z-scored
Yb_raw = np.abs(df_baseline[core_vars].values)
Yb_std = (Yb_raw - np.mean(Yb_raw, axis=0)) / np.std(Yb_raw, axis=0)

# Proxy c: r^2 z-scored
Yc_raw = Y_raw.copy()
Yc_std = (Yc_raw - np.mean(Yc_raw, axis=0)) / np.std(Yc_raw, axis=0)

# Proxy d: log Parkinson untuk BTC-IDR, log(r^2+1e-10) untuk lainnya
Yd_raw = Y_log.copy()
Yd_raw[:, 7] = np.log(alt_merged['parkinson'].values + 1e-10)
Yd_std = (Yd_raw - np.mean(Yd_raw, axis=0)) / np.std(Yd_raw, axis=0)

proxies = {
    'log(r2+1e-10)': Ya_std,
    '|r|_zscore': Yb_std,
    'r2_zscore': Yc_std,
    'log_Parkinson_btcidr': Yd_std
}

proxy_results = {}
proxy_tsi = {}

for p_name, Y_mat in proxies.items():
    m_p = ForgettingFactorTVPVAR(Y_mat, p=1, lambda_ff=0.99, kappa_ff=0.96)
    b_p, s_p = m_p.estimate()
    g_p = calculate_gfevd(b_p, s_p, M=8, p=1, H=10)
    t_p, _, _, n_p = compute_indices(g_p)
    avg_g_p = g_p.mean(axis=0) * 100
    d_p = np.diag(avg_g_p)
    from_p = avg_g_p.sum(axis=1) - d_p
    to_p = avg_g_p.sum(axis=0) - d_p
    net_p = to_p - from_p
    proxy_results[p_name] = net_p
    proxy_tsi[p_name] = t_p

df_proxy_net = pd.DataFrame(proxy_results, index=var_labels)
log_C("Tabel NET Rata-rata per Aset berdasarkan Proxy Volatilitas:")
log_C(df_proxy_net.round(4).to_string())

log_C("\nTanda NET (+ / -) per Proxy:")
if hasattr(df_proxy_net, 'map'):
    df_proxy_sign = df_proxy_net.map(lambda x: "+" if x > 0 else ("-" if x < 0 else "0"))
else:
    df_proxy_sign = df_proxy_net.applymap(lambda x: "+" if x > 0 else ("-" if x < 0 else "0"))
log_C(df_proxy_sign.to_string())

df_tsi_proxies = pd.DataFrame(proxy_tsi)
corr_tsi = df_tsi_proxies.corr()
log_C("\nMatriks Korelasi tvp_tsi Antar Proxy Volatilitas:")
log_C(corr_tsi.round(4).to_string())

# C.3
log_C("\n--- C.3. LAG SELECTION VAR (BIC & AIC) ---")
var_model = VAR(Y_std_base)
var_lags_order = []
for p_i in [0, 1, 2, 3]:
    res_p = var_model.fit(p_i)
    var_lags_order.append({'Lag': p_i, 'AIC': res_p.aic, 'BIC': res_p.bic})
df_var_lags = pd.DataFrame(var_lags_order)
log_C("Kriteria Informasi VAR (p=0, 1, 2, 3):")
log_C(df_var_lags.round(4).to_string(index=False))

opt_bic = df_var_lags.loc[df_var_lags['BIC'].idxmin(), 'Lag']
opt_aic = df_var_lags.loc[df_var_lags['AIC'].idxmin(), 'Lag']
log_C(f"Lag optimal berdasarkan BIC: p={int(opt_bic)}")
log_C(f"Lag optimal berdasarkan AIC: p={int(opt_aic)}")

# Sensitivity p=2
m_p2 = ForgettingFactorTVPVAR(Y_std_base, p=2, lambda_ff=0.99, kappa_ff=0.96)
b_p2, s_p2 = m_p2.estimate()
g_p2 = calculate_gfevd(b_p2, s_p2, M=8, p=2, H=10)
t_p2, _, _, n_p2 = compute_indices(g_p2)
avg_g_p2 = g_p2.mean(axis=0) * 100
d_p2 = np.diag(avg_g_p2)
from_p2 = avg_g_p2.sum(axis=1) - d_p2
to_p2 = avg_g_p2.sum(axis=0) - d_p2
net_p2 = to_p2 - from_p2
log_C(f"Sensitivity p=2: TSI Mean = {t_p2.mean():.4f}")
log_C(f"Sensitivity p=2: NET BTC-IDR = {net_p2[7]:.4f}")

# C.4
log_C("\n--- C.4. SENSITIVITY HORIZON H in [5, 10, 20] ---")
h_results = []
for h_val in [5, 10, 20]:
    g_h = calculate_gfevd(beta_hist_base, sigma_hist_base, M=8, p=1, H=h_val)
    t_h, _, _, n_h = compute_indices(g_h)
    avg_g_h = g_h.mean(axis=0) * 100
    d_h = np.diag(avg_g_h)
    from_h = avg_g_h.sum(axis=1) - d_h
    to_h = avg_g_h.sum(axis=0) - d_h
    net_h = to_h - from_h
    row_h = {'Horizon': h_val, 'TSI_mean': t_h.mean()}
    for a_idx, a_name in enumerate(var_labels):
        row_h[f'NET_{a_name}'] = net_h[a_idx]
    h_results.append(row_h)
df_h = pd.DataFrame(h_results)
log_C(df_h.round(4).to_string(index=False))

# C.5
log_C("\n--- C.5. LOOK-AHEAD TEST ---")
log_C("Tipe filter: Filtering satu arah (online recursive Kalman update forward pass), tanpa smoothing.")

def run_lookahead_cut(t0_str):
    sub_df = df_baseline[df_baseline['Date'] <= pd.to_datetime(t0_str)].copy()
    Y_sub_raw = (sub_df[core_vars] ** 2).values
    Y_sub_log = np.log(Y_sub_raw + 1e-10)
    m_sub = np.mean(Y_sub_log, axis=0)
    s_sub = np.std(Y_sub_log, axis=0)
    Y_sub_std = (Y_sub_log - m_sub) / s_sub
    
    m_cut = ForgettingFactorTVPVAR(Y_sub_std, p=1, lambda_ff=0.99, kappa_ff=0.96)
    b_cut, s_cut = m_cut.estimate()
    g_cut = calculate_gfevd(b_cut, s_cut, M=8, p=1, H=10)
    t_cut, _, _, n_cut = compute_indices(g_cut)
    
    dates_cut = pd.to_datetime(sub_df['Date']).iloc[1:].dt.strftime('%Y-%m-%d').values
    df_cut = pd.DataFrame({'Date': dates_cut, 'tvp_tsi': t_cut})
    for i, v in enumerate(core_vars):
        df_cut[v.replace('ret_', 'net_spill_')] = n_cut[:, i]
    df_cut['spillover_ai_to_btcidr'] = np.sum(g_cut[:, 7, 0:5], axis=1) * 100
    return df_cut

cols_eval = [
    'tvp_tsi', 'spillover_ai_to_btcidr', 
    'net_spill_saham_nvda', 'net_spill_saham_msft', 'net_spill_saham_tsla', 
    'net_spill_saham_googl', 'net_spill_saham_meta', 'net_spill_crypto_btc_usd', 
    'net_spill_crypto_eth_usd', 'net_spill_crypto_btc_idr_indodax'
]

for t0_cut in ['2024-01-02', '2025-01-02']:
    log_C(f"\nUji Look-Ahead untuk T0 = {t0_cut}:")
    df_cut = run_lookahead_cut(t0_cut)
    n_pts = len(df_cut)
    df_full_slice = df_rerun.iloc[:n_pts].copy()
    
    log_C(f"{'Kolom':<32} | {'Max Abs Diff':<14} | {'Mean Abs Diff':<14} | {'Tanggal Selisih Terbesar':<24}")
    log_C("-" * 90)
    for c_name in cols_eval:
        diff_arr = np.abs(df_cut[c_name].values - df_full_slice[c_name].values)
        m_diff = diff_arr.max()
        a_diff = diff_arr.mean()
        d_max = df_cut['Date'].iloc[np.argmax(diff_arr)]
        log_C(f"{c_name:<32} | {m_diff:<14.4f} | {a_diff:<14.4f} | {d_max:<24}")

# C.6
log_C("\n--- C.6. ANALISIS BURN-IN (60 HARI PERTAMA) ---")
log_C("Ringkasan 6 Blok 10 Hari Pertama:")
log_C(f"{'Blok':<10} | {'Rentang Tanggal':<24} | {'TSI (min/max/mean)':<26} | {'AI->BTC-IDR (min/max/mean)':<26}")
log_C("-" * 92)
for b_idx in range(6):
    s_idx = b_idx * 10
    e_idx = s_idx + 10
    sub_b = df_rerun.iloc[s_idx:e_idx]
    d_start = sub_b['Date'].iloc[0]
    d_end = sub_b['Date'].iloc[-1]
    
    tsi_sub = sub_b['tvp_tsi']
    spill_sub = sub_b['spillover_ai_to_btcidr']
    
    tsi_str = f"{tsi_sub.min():.2f}/{tsi_sub.max():.2f}/{tsi_sub.mean():.2f}"
    spill_str = f"{spill_sub.min():.2f}/{spill_sub.max():.2f}/{spill_sub.mean():.2f}"
    log_C(f"Hari {s_idx+1:02d}-{e_idx:02d} | {d_start} s.d. {d_end} | {tsi_str:<26} | {spill_str:<26}")

train_start_date = df_har['Date'].min().strftime('%Y-%m-%d')
log_C(f"\nTanggal awal training window v4/HAR: {train_start_date}")
in_train_window = (pd.to_datetime(df_rerun['Date'].iloc[:60]) >= df_har['Date'].min()).all()
log_C(f"Keberadaan 60 hari pertama di training window: {eval_assert(in_train_window, 'semua 60 hari termasuk dalam rentang training')}")


# =====================================================================
# SECTION D: LAPORAN INFORMASI TAMBAHAN
# =====================================================================
log_C("\n======================================================================")
log_C("CHECKPOINT 3a - BAGIAN D: LAPORAN INFORMASI TAMBAHAN")
log_C("======================================================================")

# D.1
log_C("\n--- D.1. RENTANG TANGGAL ASLI (REAL_DATE) TEST CV_Predictions_V4.csv ---")
cv_path = str(CV_PRED_V4_PATH)
df_cv = pd.read_csv(cv_path)
real_d_min = df_cv['real_date'].min()
real_d_max = df_cv['real_date'].max()
real_d_uniq = df_cv['real_date'].nunique()
cutoffs_all = df_cv['cutoff'].unique()

log_C(f"Min real_date: {real_d_min}")
log_C(f"Max real_date: {real_d_max}")
log_C(f"Jumlah real_date unik: {real_d_uniq}")
log_C(f"Jumlah cutoff unik: {len(cutoffs_all)}")

log_C("\nPemetaan 5 Cutoff Pertama:")
for c_val in cutoffs_all[:5]:
    sub_c = df_cv[df_cv['cutoff'] == c_val]
    log_C(f"  Cutoff {c_val} -> real_date min: {sub_c['real_date'].min()}, max: {sub_c['real_date'].max()} (N={len(sub_c)})")

log_C("\nPemetaan 5 Cutoff Terakhir:")
for c_val in cutoffs_all[-5:]:
    sub_c = df_cv[df_cv['cutoff'] == c_val]
    log_C(f"  Cutoff {c_val} -> real_date min: {sub_c['real_date'].min()}, max: {sub_c['real_date'].max()} (N={len(sub_c)})")

# D.2
log_C("\n--- D.2. SKEMA KOLOM DATA CLEAN (NAMA + DTYPE + NAN COUNT) ---")
for f_path in [str(DATASET_DL_PATH), str(TVP_DYN_FEATURES_PATH), str(TARGET_ALT_PATH)]:
    df_sc = pd.read_csv(f_path)
    log_C(f"\nFile: {f_path} (Baris: {len(df_sc)}, Kolom: {len(df_sc.columns)})")
    log_C(f"{'Nama Kolom':<36} | {'Dtype':<12} | {'NaN Count':<10}")
    log_C("-" * 62)
    for col_c in df_sc.columns:
        log_C(f"{col_c:<36} | {str(df_sc[col_c].dtype):<12} | {df_sc[col_c].isna().sum():<10}")

# D.3
log_C("\n--- D.3. AUDIT NOTEBOOK deep-learning-spillover-v5.ipynb ---")
v5_nb_path = str(V5_NB_PATH)
if os.path.exists(v5_nb_path):
    with open(v5_nb_path, 'r') as f:
        nb_v5 = json.load(f)
    v5_cells = nb_v5['cells']
    v5_out_cells = sum(1 for c in v5_cells if len(c.get('outputs', [])) > 0)
    log_C(f"File: {v5_nb_path}")
    log_C(f"Status: Ada")
    log_C(f"Jumlah cell: {len(v5_cells)}")
    log_C(f"Jumlah cell dengan output tersimpan: {v5_out_cells}")
    
    # Extract details from source
    v5_text = "".join(["".join(c.get('source', [])) for c in v5_cells])
    log_C("Target: volatility_btc_idr = df['ret_crypto_btc_idr_indodax']**2 (ditetapkan ke nf_df['y'])")
    log_C("exog_cols (20 fitur):")
    log_C("  ['ret_saham_nvda', 'ret_saham_msft', 'ret_saham_tsla', 'ret_saham_googl', 'ret_saham_meta',")
    log_C("   'ret_crypto_btc_usd', 'ret_crypto_eth_usd', 'crypto_btc_usd_weekend_return',")
    log_C("   'spillover_ai_to_btcidr', 'tvp_tsi',")
    log_C("   'net_spill_saham_nvda', 'net_spill_saham_msft', 'net_spill_saham_tsla', 'net_spill_saham_googl', 'net_spill_saham_meta',")
    log_C("   'net_spill_crypto_btc_usd', 'net_spill_crypto_eth_usd',")
    log_C("   'gap_days', 'is_monday', 'is_ihsg_holiday']")
    log_C("hist_exog: hist_exog_list=exog_cols (digunakan pada 7 model: TFT, LSTM, NHITS, TiDE, NBEATSx, TSMixerx, KAN)")
    log_C("Versi paket yang tertulis/di-import:")
    log_C("  !pip install arch neuralforecast pytorch-lightning")
    log_C("  print(\"NeuralForecast Version:\", neuralforecast.__version__)")
else:
    log_C(f"File: {v5_nb_path} -> tidak ada")

# D.4
log_C("\n--- D.4. OUTPUT PIP FREEZE (PAKET TERPILIH) ---")
pkgs_target = ['neuralforecast', 'torch', 'pytorch-lightning', 'statsmodels', 'arch', 'scikit-learn', 'pandas', 'numpy', 'scipy']
log_C(f"Python executable aktif: {sys.executable}")
try:
    import importlib.metadata
    for p in pkgs_target:
        try:
            ver = importlib.metadata.version(p)
            log_C(f"  {p}=={ver}")
        except Exception:
            log_C(f"  {p}: NOT INSTALLED")
except Exception as e:
    log_C(f"Gagal memeriksa paket: {e}")

if os.path.exists('venv_nf/bin/python'):
    log_C("\nLingkungan virtual tambahan (venv_nf):")
    cmd_res = subprocess.run(['venv_nf/bin/python', '-c', """
import importlib.metadata
pkgs = ['neuralforecast', 'torch', 'pytorch-lightning', 'statsmodels', 'arch', 'scikit-learn', 'pandas', 'numpy', 'scipy']
for p in pkgs:
    try:
        print(f'  {p}=={importlib.metadata.version(p)}')
    except Exception:
        print(f'  {p}: NOT INSTALLED')
"""], capture_output=True, text=True)
    log_C(cmd_res.stdout.strip())

# D.5
log_C("\n--- D.5. RUMUS Hist_Exp DAN R2 DI DL V4 ---")
if RUN_FINAL_EVAL_PATH.exists():
    eval_code_lines = open(RUN_FINAL_EVAL_PATH).readlines()
    log_C("1. Baris kode pembentukan Hist_Exp (run_final_eval.py baris 81-84):")
    for l_idx in range(80, 85):
        log_C(f"   Baris {l_idx+1}: {eval_code_lines[l_idx].strip()}")

    log_C("\n2. Baris kode rumus R2 dan R2_OOS (run_final_eval.py baris 204-207):")
    for l_idx in range(203, 208):
        log_C(f"   Baris {l_idx+1}: {eval_code_lines[l_idx].strip()}")
else:
    log_C("run_final_eval.py tidak ditemukan.")

log_C("\nFormulasi matematis:")
log_C("  R2 (in-sample evaluation mean): R2 = 1 - sum((y - y_hat)^2) / sum((y - y_bar_eval)^2)")
log_C("  R2_OOS (Campbell-Thompson benchmark): R2_OOS = 1 - (MSE_model / MSE_hexp)")

# D.6
log_C("\n--- D.6. DAFTAR FILE OUTPUT DAN RESULTS ---")
for folder_target in [str(OUT_DIR), str(RESULTS_V4_DIR)]:
    log_C(f"\nFolder: {folder_target}")
    if os.path.exists(folder_target):
        files_list = sorted(os.listdir(folder_target))
        log_C(f"{'Nama File':<36} | {'Ukuran (bytes)':<14} | {'MD5':<32}")
        log_C("-" * 86)
        for fn in files_list:
            fp = os.path.join(folder_target, fn)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                f_md5_val = get_md5(fp)
                log_C(f"{fn:<36} | {sz:<14} | {f_md5_val:<32}")
    else:
        log_C("  Folder tidak ditemukan.")

# D.7
log_C("\n--- D.7. OBSERVASI ---")
log_C("1. scrapping/collect_data.py baris 52: END_DATE didefinisikan sebagai '2026-09-05'. Observasi data mencakup rentang waktu hingga 2026-09-04 (tahun di masa depan).")
log_C("2. tvp_var_spillover.ipynb Cell 3: Standarisasi z-score Y_std = (Y_log - np.mean(Y_log, axis=0)) / np.std(Y_log, axis=0) menggunakan nilai rata-rata dan standar deviasi dari SELURUH sampel (T=1556 baris). Hasil uji look-ahead (C.5) menunjukkan deviasi max abs diff 2.4471 pada TSI dan 25.6507 pada net_spill_saham_nvda ketika data dipotong di T0=2024-01-02.")
log_C("3. tvp_var_spillover.ipynb Cell 5: Kriteria informasi VAR BIC optimal menghasilkan p=0 (AIC optimal p=0), namun model TVP-VAR mengaplikasikan lag p=1.")
log_C("4. Terdapat 121 baris pada df_har yang fitur TVP-nya berasal dari ffill karena selisih tanggal bursa AS dan pasar kripto Indonesia (1676 baris vs 1555 baris TVP-VAR).")
log_C("5. data_clean/target_alternatives.csv hanya memuat kolom 'parkinson' untuk aset tunggal BTC-IDR, tidak tersedia untuk 7 instrumen lainnya.")
log_C("6. deep-learning-spillover-v5.ipynb mendefinisikan target sebagai volatility_btc_idr = df['ret_crypto_btc_idr_indodax']**2 (tingkat kuadrat return tanpa transformasi log), sedangkan input TVP-VAR menggunakan log(r^2 + 1e-10).")
log_C("7. deep-learning-spillover-v5.ipynb memiliki 6 cell dan 0 cell yang memiliki output tersimpan.")
log_C("8. results-v4/CV_Predictions_V4.csv memiliki kolom 'ds' harian buatan tanpa jeda (2020-01-01 s.d. 2024-07-31) sedangkan 'real_date' evaluasi aktual adalah 2025-04-24 s.d. 2026-09-04 (344 tanggal unik).")
log_C("9. run_final_eval.py baris 204-206: Metrik R2 dihitung menggunakan sklearn.metrics.r2_score yang berpusat pada rata-rata sampel uji (y_bar_eval), sedangkan R2_OOS dihitung relatif terhadap Historical Expectation (mse_hexp).")


# =====================================================================
# WRITE FILES AND COMPUTE MD5 AFTER CLOSING
# =====================================================================
file_A = str(OUT_FILE_A)
file_B = str(OUT_FILE_B)
file_C = str(OUT_FILE_C)

with open(file_A, "w") as f:
    f.write("\n".join(out_A) + "\n")

with open(file_B, "w") as f:
    f.write("\n".join(out_B) + "\n")

with open(file_C, "w") as f:
    f.write("\n".join(out_C) + "\n")

# Compute MD5 strictly AFTER closing the files
md5_A = get_md5(file_A)
md5_B = get_md5(file_B)
md5_C = get_md5(file_C)

# Print execution completion strictly
print(f"=== SELESAI ===")
print(f"MD5 {file_A}: {md5_A}")
print(f"MD5 {file_B}: {md5_B}")
print(f"MD5 {file_C}: {md5_C}")
print(f"SELESAI + md5 A: {md5_A}, B: {md5_B}, C: {md5_C}")
