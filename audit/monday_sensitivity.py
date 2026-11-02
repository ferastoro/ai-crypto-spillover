#!/usr/bin/env python3
"""
audit/monday_sensitivity.py
========================================================================================
Tujuan:
Uji sensitivitas model TVP-VAR (RQ1) terhadap:
1. Hari Senin dan hari pasca-libur bursa (jarak kalender > 1 hari).
2. Perbedaan definisi return cryptocurrency antara dataset RQ1 (dataset_garch_baseline.csv)
   dan dataset Deep Learning (dataset_deep_learning.csv).

Aturan & Batasan:
- READ-ONLY terhadap seluruh file existing.
- Tanpa GPU, tanpa API eksternal.
- Toleransi verifikasi baseline terhadap Average_Spillover_Table_vol.csv: 1e-3.
- Kriteria perubahan material (ditetapkan ex-ante):
  1. TSI rata-rata berubah > 3 poin persentase (pp).
  2. Tanda net spillover BTC-IDR atau BTC-USD berbalik.
  3. Transmisi 5 saham AI -> BTC-IDR berubah > 2 poin persentase (pp).
========================================================================================
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.linalg import pinv

# Direktori proyek
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CLEAN = BASE_DIR / "data" / "clean"
DATA_RAW = BASE_DIR / "data" / "raw"
SCRATCH_FINAL = BASE_DIR / "scratch" / "final"
AUDIT_DIR = BASE_DIR / "audit"

CORE_VARS = [
    'ret_saham_nvda', 'ret_saham_msft', 'ret_saham_tsla', 'ret_saham_googl', 'ret_saham_meta', 
    'ret_crypto_btc_usd', 'ret_crypto_eth_usd', 'ret_crypto_btc_idr_indodax'
]

VAR_LABELS = [v.replace('ret_', '').upper() for v in CORE_VARS]
AI_VARS = ['SAHAM_NVDA', 'SAHAM_MSFT', 'SAHAM_TSLA', 'SAHAM_GOOGL', 'SAHAM_META']

# ======================================================================================
# MODEL TVP-VAR & GFEVD (Sesuai persis tvp_var_spillover.ipynb Cell 6 & 8)
# ======================================================================================

class ForgettingFactorTVPVAR:
    def __init__(self, y: np.ndarray, p: int = 1, lambda_ff: float = 0.99, kappa_ff: float = 0.96):
        self.y = y
        self.T, self.M = y.shape
        self.p = p
        self.K = self.M * self.p + 1  # lagged vars + intercept
        self.lambda_ff = lambda_ff
        self.kappa_ff = kappa_ff
        
        self.Z = np.zeros((self.T - p, self.M, self.K * self.M))
        self.Y = self.y[p:]
        
        for t in range(self.T - p):
            z_t = np.concatenate([[1.0], self.y[t : t + p].flatten()[::-1]])
            self.Z[t] = np.kron(np.eye(self.M), z_t)
            
    def estimate(self):
        T_est = self.T - self.p
        beta_t = np.zeros((self.K * self.M, 1))
        P_t = np.eye(self.K * self.M) * 10.0
        Sigma_t = np.cov(self.y.T)
        
        self.beta_history = np.zeros((T_est, self.K * self.M))
        self.Sigma_history = np.zeros((T_est, self.M, self.M))
        
        for t in range(T_est):
            Z_t = self.Z[t]
            y_t = self.Y[t].reshape(self.M, 1)
            
            P_t_pred = P_t / self.lambda_ff
            v_t = y_t - Z_t @ beta_t
            
            Sigma_t = self.kappa_ff * Sigma_t + (1 - self.kappa_ff) * (v_t @ v_t.T)
            
            F_t = Z_t @ P_t_pred @ Z_t.T + Sigma_t
            K_t = P_t_pred @ Z_t.T @ pinv(F_t)
            
            beta_t = beta_t + K_t @ v_t
            P_t = P_t_pred - K_t @ Z_t @ P_t_pred
            
            self.beta_history[t] = beta_t.flatten()
            self.Sigma_history[t] = Sigma_t
            
        return self.beta_history, self.Sigma_history


def calculate_gfevd(beta_history: np.ndarray, sigma_history: np.ndarray, M: int, p: int = 1, H: int = 10):
    T_est = beta_history.shape[0]
    gfevd_history = np.zeros((T_est, M, M))
    
    for t in range(T_est):
        beta_t = beta_history[t]
        Sigma_t = sigma_history[t]
        sigma_diag = np.diag(Sigma_t)
        
        A = np.zeros((p, M, M))
        idx = 0
        for i in range(M):
            idx += 1  # lewati intercept
            for lag in range(p):
                for j in range(M):
                    A[lag, i, j] = beta_t[idx]
                    idx += 1
                    
        Phi = np.zeros((H, M, M))
        Phi[0] = np.eye(M)
        for h in range(1, H):
            for j in range(1, min(h, p) + 1):
                Phi[h] += Phi[h-j] @ A[j-1]
                
        theta = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                e_j = np.zeros((M, 1))
                e_j[j] = 1.0
                
                num = 0.0
                den = 0.0
                for h in range(H):
                    term = (Phi[h][i, :] @ Sigma_t @ e_j)[0]
                    num += term**2
                    den += (Phi[h][i, :] @ Sigma_t @ Phi[h][i, :].T)
                    
                theta[i, j] = (1.0 / sigma_diag[j]) * (num / den)
                
        row_sums = theta.sum(axis=1)
        theta_tilde = theta / row_sums[:, np.newaxis]
        gfevd_history[t] = theta_tilde
        
    return gfevd_history


def compute_spillover_metrics(df_input: pd.DataFrame, vars_list: list = CORE_VARS, p: int = 1, H: int = 10, lambda_ff: float = 0.99, kappa_ff: float = 0.96):
    Y = (df_input[vars_list] ** 2).values
    Y_log = np.log(Y + 1e-10)
    Y_std = (Y_log - np.mean(Y_log, axis=0)) / np.std(Y_log, axis=0)
    
    M = len(vars_list)
    tvp_var = ForgettingFactorTVPVAR(Y_std, p=p, lambda_ff=lambda_ff, kappa_ff=kappa_ff)
    beta_hist, sigma_hist = tvp_var.estimate()
    gfevd_hist = calculate_gfevd(beta_hist, sigma_hist, M=M, p=p, H=H)
    
    T_est = gfevd_hist.shape[0]
    tsi = np.zeros(T_est)
    for t in range(T_est):
        off_diag_sum = gfevd_hist[t].sum() - np.trace(gfevd_hist[t])
        tsi[t] = (off_diag_sum / M) * 100.0
    mean_tsi = float(np.mean(tsi))
    
    avg_gfevd = gfevd_hist.mean(axis=0) * 100.0
    labels = [v.replace('ret_', '').upper() for v in vars_list]
    table = pd.DataFrame(avg_gfevd, index=labels, columns=labels)
    diag_vals = np.diag(table)
    table['FROM Others'] = table.sum(axis=1) - diag_vals
    to_others = table.iloc[:M, :M].sum(axis=0) - diag_vals
    table.loc['TO Others'] = list(to_others) + [mean_tsi]
    net_spillover = to_others - table['FROM Others'].iloc[:M]
    table.loc['NET Spillover'] = list(net_spillover) + [np.nan]
    
    ai_to_btcidr = float(table.loc['CRYPTO_BTC_IDR_INDODAX', AI_VARS].sum())
    ai_to_btcusd = float(table.loc['CRYPTO_BTC_USD', AI_VARS].sum())
    btcusd_to_btcidr = float(table.loc['CRYPTO_BTC_IDR_INDODAX', 'CRYPTO_BTC_USD'])
    btcidr_to_btcusd = float(table.loc['CRYPTO_BTC_USD', 'CRYPTO_BTC_IDR_INDODAX'])
    
    return {
        'N': len(df_input),
        'mean_tsi': mean_tsi,
        'net_spillover': net_spillover,
        'ai_to_btcidr': ai_to_btcidr,
        'ai_to_btcusd': ai_to_btcusd,
        'btcusd_to_btcidr': btcusd_to_btcidr,
        'btcidr_to_btcusd': btcidr_to_btcusd,
        'table': table
    }


def compute_md5(filepath: Path) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


# ======================================================================================
# EKSEKUSI UTAMA
# ======================================================================================

def main():
    print("=" * 80)
    print("AUDIT SENSITIVITAS RQ1 (TVP-VAR): HARI SENIN & DEFINISI RETURN CRYPTO")
    print("=" * 80)

    # ----------------------------------------------------------------------------------
    # LANGKAH 0: Reproduksi Baseline (Wajib Lolos)
    # ----------------------------------------------------------------------------------
    print("\n[LANGKAH 0] Memeriksa Reproduksi Baseline dari data/clean/dataset_garch_baseline.csv...")
    df_base = pd.read_csv(DATA_CLEAN / "dataset_garch_baseline.csv")
    res_v0 = compute_spillover_metrics(df_base)
    
    ref_table_path = SCRATCH_FINAL / "Average_Spillover_Table_vol.csv"
    if not ref_table_path.exists():
        print(f"ERROR: File referensi tidak ditemukan di {ref_table_path}")
        sys.exit(1)
        
    ref_table = pd.read_csv(ref_table_path, index_col=0)
    
    target_tsi = 38.7415
    target_net_btcidr = 5.4546
    target_net_btcusd = 11.7985
    
    diff_tsi = abs(res_v0['mean_tsi'] - target_tsi)
    diff_btcidr = abs(res_v0['net_spillover']['CRYPTO_BTC_IDR_INDODAX'] - target_net_btcidr)
    diff_btcusd = abs(res_v0['net_spillover']['CRYPTO_BTC_USD'] - target_net_btcusd)
    
    print(f"  TSI Model       : {res_v0['mean_tsi']:.4f} (Target: {target_tsi:.4f}, Selisih: {diff_tsi:.6f})")
    print(f"  NET BTC-IDR     : {res_v0['net_spillover']['CRYPTO_BTC_IDR_INDODAX']:.4f} (Target: {target_net_btcidr:.4f}, Selisih: {diff_btcidr:.6f})")
    print(f"  NET BTC-USD     : {res_v0['net_spillover']['CRYPTO_BTC_USD']:.4f} (Target: {target_net_btcusd:.4f}, Selisih: {diff_btcusd:.6f})")
    
    tol = 1e-3
    if diff_tsi > tol or diff_btcidr > tol or diff_btcusd > tol:
        print("\nFATAL ERROR: Hasil estimasi berbeda dari target baseline melebihi toleransi 1e-3!")
        print("BERHENTI sesuai instruksi Langkah 0.")
        sys.exit(1)
    else:
        print("  -> STATUS LANGKAH 0: LOLOS (Toleransi terpenuhi persis). Melanjutkan ke Langkah berikutnya.")

    # ----------------------------------------------------------------------------------
    # LANGKAH 1: Deskriptif Hari Senin (Tanpa Model)
    # ----------------------------------------------------------------------------------
    print("\n[LANGKAH 1] Menghitung Statistik Deskriptif Hari Senin & Hari Pasca-Libur...")
    dates = pd.to_datetime(df_base['Date'])
    gap_days = (dates - dates.shift(1)).dt.days
    is_gap = gap_days > 1  # 383 baris
    is_other = ~is_gap     # 1173 baris (termasuk baris 0 dengan gap NaN)
    
    count_gap = int(is_gap.sum())
    count_other = int(is_other.sum())
    count_eq1 = int((gap_days == 1).sum())
    pct_gap = (count_gap / len(df_base)) * 100.0
    
    print(f"  Total baris dataset: {len(df_base)}")
    print(f"  Baris jarak kalender > 1 hari : {count_gap} ({pct_gap:.2f}%)")
    print(f"  Baris lainnya (jarak <= 1 hari): {count_other} ({100 - pct_gap:.2f}%) [gap==1: {count_eq1}]")
    
    # Hitung z-score log-vol dan raw r^2
    Y_base = (df_base[CORE_VARS] ** 2).values
    Y_log_base = np.log(Y_base + 1e-10)
    Y_std_base = (Y_log_base - np.mean(Y_log_base, axis=0)) / np.std(Y_log_base, axis=0)
    
    df_std = pd.DataFrame(Y_std_base, columns=CORE_VARS)
    df_r2 = pd.DataFrame(Y_base, columns=CORE_VARS)
    
    desc_rows = []
    for v in CORE_VARS:
        label = v.replace('ret_', '').upper()
        m_z_gap = float(df_std.loc[is_gap, v].mean())
        m_z_other = float(df_std.loc[is_other, v].mean())
        diff_z = m_z_gap - m_z_other
        
        m_r2_gap = float(df_r2.loc[is_gap, v].mean())
        m_r2_other = float(df_r2.loc[is_other, v].mean())
        diff_r2 = m_r2_gap - m_r2_other
        ratio_r2 = m_r2_gap / m_r2_other if m_r2_other != 0 else np.nan
        
        var_ret_gap = float(df_base.loc[is_gap, v].var())
        var_ret_other = float(df_base.loc[is_other, v].var())
        ratio_var = var_ret_gap / var_ret_other if var_ret_other != 0 else np.nan
        
        desc_rows.append({
            'variable': label,
            'mean_z_gap': m_z_gap,
            'mean_z_other': m_z_other,
            'diff_z': diff_z,
            'mean_r2_gap': m_r2_gap,
            'mean_r2_other': m_r2_other,
            'diff_r2': diff_r2,
            'ratio_r2': ratio_r2,
            'var_ret_gap': var_ret_gap,
            'var_ret_other': var_ret_other,
            'ratio_var': ratio_var
        })
        
    df_desc = pd.DataFrame(desc_rows)
    print("\nTabel Ringkasan Deskriptif Gap > 1 Hari vs Hari Lain:")
    print(df_desc[['variable', 'diff_z', 'mean_r2_gap', 'mean_r2_other', 'diff_r2', 'ratio_r2', 'ratio_var']].to_string(index=False))

    # ----------------------------------------------------------------------------------
    # LANGKAH 2: Tiga Varian TVP-VAR
    # ----------------------------------------------------------------------------------
    print("\n[LANGKAH 2] Menjalankan 3 Varian TVP-VAR...")
    
    # V0: Baseline
    # res_v0 sudah dihitung di Langkah 0
    
    # V1: Buang semua baris jarak > 1 hari
    # Menguji V1 dengan ~is_gap (N=1173)
    df_v1 = df_base.loc[~is_gap].reset_index(drop=True)
    res_v1 = compute_spillover_metrics(df_v1)
    
    # Sebagai robustness deskriptif: V1 hanya gap == 1 (N=1172)
    df_v1_1172 = df_base.loc[gap_days == 1].reset_index(drop=True)
    res_v1_1172 = compute_spillover_metrics(df_v1_1172)
    
    # V2: Return crypto dari dataset_deep_learning.csv pada irisan tanggal
    df_dl = pd.read_csv(DATA_CLEAN / "dataset_deep_learning.csv")
    df_v2 = pd.merge(df_base[['Date']], df_dl[['Date'] + CORE_VARS], on='Date', how='inner')
    res_v2 = compute_spillover_metrics(df_v2)
    
    # Kriteria Perubahan Material:
    # 1. TSI rata-rata berubah > 3 poin persentase (|delta_TSI| > 3.0)
    # 2. Tanda net spillover BTC-IDR atau BTC-USD berbalik
    # 3. Transmisi 5 saham AI -> BTC-IDR berubah > 2 poin persentase (|delta_transmisi| > 2.0)
    
    def evaluate_material_change(res_variant, res_baseline):
        c1 = abs(res_variant['mean_tsi'] - res_baseline['mean_tsi']) > 3.0
        
        sign_btcidr_base = np.sign(res_baseline['net_spillover']['CRYPTO_BTC_IDR_INDODAX'])
        sign_btcidr_var = np.sign(res_variant['net_spillover']['CRYPTO_BTC_IDR_INDODAX'])
        sign_btcusd_base = np.sign(res_baseline['net_spillover']['CRYPTO_BTC_USD'])
        sign_btcusd_var = np.sign(res_variant['net_spillover']['CRYPTO_BTC_USD'])
        c2 = (sign_btcidr_base != sign_btcidr_var) or (sign_btcusd_base != sign_btcusd_var)
        
        c3 = abs(res_variant['ai_to_btcidr'] - res_baseline['ai_to_btcidr']) > 2.0
        
        return {
            'c1_tsi_gt_3pp': c1,
            'c2_sign_flip': c2,
            'c3_ai_trans_gt_2pp': c3,
            'is_material': c1 or c2 or c3
        }
        
    crit_v1 = evaluate_material_change(res_v1, res_v0)
    crit_v2 = evaluate_material_change(res_v2, res_v0)
    
    # Tabel Komparasi V0, V1, V2
    variants = [
        ('V0: Baseline', res_v0, {'c1_tsi_gt_3pp': False, 'c2_sign_flip': False, 'c3_ai_trans_gt_2pp': False, 'is_material': False}),
        ('V1: Drop Gap>1 (N=1173)', res_v1, crit_v1),
        ('V2: DL Crypto Ret (N=1556)', res_v2, crit_v2)
    ]
    
    comp_rows = []
    for name, res, crit in variants:
        comp_rows.append({
            'variant': name,
            'N': res['N'],
            'mean_tsi': res['mean_tsi'],
            'net_nvda': res['net_spillover']['SAHAM_NVDA'],
            'net_msft': res['net_spillover']['SAHAM_MSFT'],
            'net_tsla': res['net_spillover']['SAHAM_TSLA'],
            'net_googl': res['net_spillover']['SAHAM_GOOGL'],
            'net_meta': res['net_spillover']['SAHAM_META'],
            'net_btcusd': res['net_spillover']['CRYPTO_BTC_USD'],
            'net_ethusd': res['net_spillover']['CRYPTO_ETH_USD'],
            'net_btcidr': res['net_spillover']['CRYPTO_BTC_IDR_INDODAX'],
            'sign_btcusd': '+' if res['net_spillover']['CRYPTO_BTC_USD'] > 0 else '-',
            'sign_btcidr': '+' if res['net_spillover']['CRYPTO_BTC_IDR_INDODAX'] > 0 else '-',
            'ai_to_btcidr': res['ai_to_btcidr'],
            'ai_to_btcusd': res['ai_to_btcusd'],
            'btcusd_to_btcidr': res['btcusd_to_btcidr'],
            'btcidr_to_btcusd': res['btcidr_to_btcusd'],
            'crit_tsi_gt_3pp': "Ya" if crit['c1_tsi_gt_3pp'] else "Tidak",
            'crit_sign_flip': "Ya" if crit['c2_sign_flip'] else "Tidak",
            'crit_ai_gt_2pp': "Ya" if crit['c3_ai_trans_gt_2pp'] else "Tidak",
            'perubahan_material': "Ya" if crit['is_material'] else "Tidak"
        })
        
    df_comp = pd.DataFrame(comp_rows)
    print("\nTabel Perbandingan 3 Varian TVP-VAR:")
    print(df_comp[['variant', 'N', 'mean_tsi', 'net_btcusd', 'net_btcidr', 'ai_to_btcidr', 'perubahan_material']].to_string(index=False))

    # ----------------------------------------------------------------------------------
    # LANGKAH 3: Verifikasi Dua Angka Audit Sebelumnya
    # ----------------------------------------------------------------------------------
    print("\n[LANGKAH 3] Verifikasi Dua Angka dari Audit Sebelumnya...")
    
    # 3a. Korelasi return log BTC-IDR vs BTC-USD lag 0 dan lag -1
    col_idr = 'ret_crypto_btc_idr_indodax'
    col_usd = 'ret_crypto_btc_usd'
    
    s_idr = df_base[col_idr]
    s_usd = df_base[col_usd]
    
    corr_lag0 = float(s_idr.corr(s_usd))
    # lag -1: USD memimpin IDR 1 hari (corr(IDR_t, USD_{t-1}))
    corr_lag_minus1 = float(s_idr.corr(s_usd.shift(1)))
    
    print(f"  3a. Korelasi Return BTC-IDR vs BTC-USD:")
    print(f"      Lag 0  : {corr_lag0:.4f} (Target audit: 0.9811, Selisih: {abs(corr_lag0 - 0.9811):.6f})")
    print(f"      Lag -1 : {corr_lag_minus1:.4f} (Target audit: -0.0046, Selisih: {abs(corr_lag_minus1 - (-0.0046)):.6f})")
    status_3a = "VERIFIED" if (abs(corr_lag0 - 0.9811) < 1e-4 and abs(corr_lag_minus1 - (-0.0046)) < 1e-4) else "UNVERIFIED"
    print(f"      Status 3a: {status_3a}")
    
    # 3b. Breakdown selisih ~120 hari
    # Mengamati tanggal raw untuk 11 instrumen
    instruments_files = {
        'saham_nvda': 'saham_nvda.csv',
        'saham_msft': 'saham_msft.csv',
        'saham_tsla': 'saham_tsla.csv',
        'saham_googl': 'saham_googl.csv',
        'saham_meta': 'saham_meta.csv',
        'crypto_btc_usd': 'crypto_btc_usd.csv',
        'crypto_eth_usd': 'crypto_eth_usd.csv',
        'crypto_btc_idr_indodax': 'crypto_btc_idr_indodax.csv',
        'konteks_ihsg': 'konteks_ihsg.csv',
        'konteks_vix': 'konteks_vix.csv',
        'konteks_usdidr': 'konteks_usdidr.csv',
    }
    
    raw_dates = {}
    for k, f in instruments_files.items():
        df_inst = pd.read_csv(DATA_RAW / f)
        raw_dates[k] = set(pd.to_datetime(df_inst['Date']).dt.strftime('%Y-%m-%d'))
        
    dl_dates = set(df_dl['Date'])
    base_dates = set(df_base['Date'])
    missing_121 = dl_dates - base_dates
    
    drop_breakdown = []
    for k in instruments_files.keys():
        missing_count = len(dl_dates - raw_dates[k])
        drop_breakdown.append({
            'instrument': k,
            'us_dates_missing': missing_count,
            'pct_of_121_gap': (missing_count / len(missing_121)) * 100.0 if len(missing_121) > 0 else 0.0
        })
    df_drop = pd.DataFrame(drop_breakdown)
    print(f"\n  3b. Breakdown 121 Tanggal yang Hilang dari Kalender Saham AS (N_DL={len(dl_dates)} vs N_Base={len(base_dates)}):")
    print(df_drop.to_string(index=False))

    # ----------------------------------------------------------------------------------
    # EXPORT TABEL KE CSV
    # ----------------------------------------------------------------------------------
    # Simpan tabel gabungan ke audit/monday_sensitivity_tables.csv
    tables_csv_path = AUDIT_DIR / "monday_sensitivity_tables.csv"
    
    with open(tables_csv_path, 'w') as f:
        f.write("# TABEL 1: DESKRIPTIF HARI SENIN & PASCA-LIBUR (GAP > 1 HARI)\n")
        df_desc.to_csv(f, index=False)
        f.write("\n# TABEL 2: PERBANDINGAN TIGA VARIAN TVP-VAR & EVALUASI KRITERIA MATERIAL\n")
        df_comp.to_csv(f, index=False)
        f.write("\n# TABEL 3: BREAKDOWN 121 TANGGAL YANG TERBUANG DALAM LISTWISE DELETION\n")
        df_drop.to_csv(f, index=False)
        f.write("\n# TABEL 4: MATRIX GFEVD RATA-RATA V0 BASELINE\n")
        res_v0['table'].to_csv(f)
        f.write("\n# TABEL 5: MATRIX GFEVD RATA-RATA V1 DROP GAP\n")
        res_v1['table'].to_csv(f)
        f.write("\n# TABEL 6: MATRIX GFEVD RATA-RATA V2 DL DEFINITION\n")
        res_v2['table'].to_csv(f)
        
    print(f"\n[OUTPUT] Seluruh tabel berhasil diekspor ke: {tables_csv_path}")

    # MD5 Hashes
    md5_files = [
        DATA_CLEAN / "dataset_garch_baseline.csv",
        DATA_CLEAN / "dataset_deep_learning.csv",
        SCRATCH_FINAL / "Average_Spillover_Table_vol.csv",
        BASE_DIR / "scrapping" / "preprocess_data.py",
        BASE_DIR / "tvp_var_spillover.ipynb"
    ]
    print("\nMD5 Hashes Input Files:")
    for mf in md5_files:
        print(f"  {mf.name:35}: {compute_md5(mf)}")


if __name__ == "__main__":
    main()
