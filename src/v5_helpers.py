"""
========================================================================================
v5_helpers.py - Modul Pembantu Standar Evaluasi Deep Learning V5 (RSIC)
========================================================================================
File ini diekstrak langsung dari repositori lokal untuk menjamin konsistensi evaluasi
antara benchmark ekonometrika (HAR, GARCH) dan model Deep Learning (TFT, TiDE, dll.).

SUMBER EKSTRAKSI & NOMOR BARIS FILE ASAL:
1. get_cutoff_origins():
   - File Asal: scratch/final/run_final_eval.py (baris 64-70)
   - Deskripsi: Menentukan tanggal observasi historis nyata terakhir sebelum real_date h=1
2. compute_baselines():
   - File Asal: scratch/final/run_final_eval.py (baris 74-85)
   - Deskripsi: Menghitung Hist_Exp, Roll_20, dan Persistence secara rekursif per cutoff
3. prepare_t0_target() & prepare_t2_target():
   - File Asal: scratch/run_checkpoint2i.py (baris 170-196)
   - Deskripsi: Definisi target kuadrat return T0 (r2_1) dan Parkinson volatility T2
4. fit_predict_har_ols():
   - File Asal: scratch/run_checkpoint2i.py (baris 327-333)
   - Deskripsi: Estimasi OLS regresi HAR multi-horizon dengan intercept
5. diebold_mariano_test():
   - File Asal: scratch/run_checkpoint2i.py (baris 440-466) & scratch/final/run_final_eval.py (baris 101-133)
   - Deskripsi: Uji komparasi akurasi ramalan Diebold-Mariano (Newey-West + koreksi HLN + Bonferroni)
6. evaluate_metrics():
   - File Asal: scratch/final/run_final_eval.py (baris 204-210) & Checkpoint 2i
   - Deskripsi: Perhitungan metrik MSE, MAE, R2_OOS vs Hist_Exp & Roll_20, std pred, corr
========================================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# --------------------------------------------------------------------------------------
# 1. DEFINISI TARGET T0 & T2
# File Asal: scratch/run_checkpoint2i.py (baris 170-196)
# --------------------------------------------------------------------------------------
def prepare_t0_target(df):
    """
    Ekstrak target T0: kuadrat return BTC-IDR (r2_1).
    scratch/run_checkpoint2i.py baris 176:
    df_har['r2_1'] = df_har['ret_crypto_btc_idr_indodax']**2
    """
    if 'ret_crypto_btc_idr_indodax' not in df.columns:
        raise ValueError("Kolom 'ret_crypto_btc_idr_indodax' tidak ditemukan di DataFrame!")
    return df['ret_crypto_btc_idr_indodax'] ** 2

def prepare_t2_target(df, alt_df=None):
    """
    Ekstrak target T2: Parkinson volatility dengan penanganan nilai nol.
    scratch/run_checkpoint2i.py baris 170-174:
    p_zeros = (df_har['parkinson'] == 0).sum()
    if p_zeros > 0:
        min_p_nz = df_har.loc[df_har['parkinson'] > 0, 'parkinson'].min()
        df_har.loc[df_har['parkinson'] == 0, 'parkinson'] = min_p_nz / 2
    """
    if 'parkinson' in df.columns:
        park_series = df['parkinson'].copy()
    elif alt_df is not None and 'parkinson' in alt_df.columns:
        merged = pd.merge(df[['Date']], alt_df[['Date', 'parkinson']], on='Date', how='left')
        park_series = merged['parkinson'].copy()
    else:
        raise ValueError("Kolom 'parkinson' tidak ditemukan di DataFrame maupun alt_df!")

    p_zeros = (park_series == 0).sum()
    if p_zeros > 0:
        min_p_nz = park_series[park_series > 0].min()
        park_series[park_series == 0] = min_p_nz / 2.0
    return park_series

# --------------------------------------------------------------------------------------
# 2. DEFINISI ORIGIN PER CUTOFF
# File Asal: scratch/final/run_final_eval.py (baris 64-70)
# --------------------------------------------------------------------------------------
def get_cutoff_origins(cv_df, hist_series):
    """
    Memetakan tiap cutoff ke tanggal observasi historis nyata terakhir sebelum real_date h=1.
    scratch/final/run_final_eval.py baris 64-70:
    for c in cutoffs:
        h1_real_date = cv_df.loc[cv_df['cutoff'] == c, 'real_date'].min()
        past_dates = hist_series.index[hist_series.index < h1_real_date]
        if len(past_dates) > 0:
            cutoff_obs_date[c] = past_dates.max()
    """
    cutoffs = cv_df['cutoff'].unique()
    cutoff_obs_date = {}
    for c in cutoffs:
        h1_real_date = cv_df.loc[cv_df['cutoff'] == c, 'real_date'].min()
        past_dates = hist_series.index[hist_series.index < h1_real_date]
        if len(past_dates) > 0:
            cutoff_obs_date[c] = past_dates.max()
    return cutoff_obs_date

# --------------------------------------------------------------------------------------
# 3. BASELINES: Hist_Exp, Roll_20, Persistence
# File Asal: scratch/final/run_final_eval.py (baris 74-85)
# --------------------------------------------------------------------------------------
def compute_baselines(cv_df, hist_series, cutoff_obs_date=None):
    """
    Menghitung prediksi benchmark Hist_Exp, Roll_20, dan Persistence secara murni kausal.
    scratch/final/run_final_eval.py baris 74-85:
    for idx, row in cv_df.iterrows():
        c_date = row['cutoff_obs_date']
        past_y = hist_series[hist_series.index <= c_date]
        y_persist.append(past_y.iloc[-1] if len(past_y) > 0 else np.nan)
        y_hist_exp.append(past_y.mean() if len(past_y) > 0 else np.nan)
        y_roll_20.append(past_y.tail(20).mean() if len(past_y) >= 20 else np.nan)
    """
    if cutoff_obs_date is None:
        cutoff_obs_date = get_cutoff_origins(cv_df, hist_series)

    y_persist = []
    y_hist_exp = []
    y_roll_20 = []

    for idx, row in cv_df.iterrows():
        c = row['cutoff']
        c_date = cutoff_obs_date.get(c, np.nan)
        if pd.isna(c_date):
            y_persist.append(np.nan)
            y_hist_exp.append(np.nan)
            y_roll_20.append(np.nan)
            continue
        past_y = hist_series[hist_series.index <= c_date]
        if len(past_y) > 0:
            y_persist.append(past_y.iloc[-1])
            y_hist_exp.append(past_y.mean())
            y_roll_20.append(past_y.tail(20).mean() if len(past_y) >= 20 else np.nan)
        else:
            y_persist.append(np.nan)
            y_hist_exp.append(np.nan)
            y_roll_20.append(np.nan)

    return np.array(y_hist_exp), np.array(y_roll_20), np.array(y_persist)

# --------------------------------------------------------------------------------------
# 3B. GARCH & EGARCH DIRECT MERGE
# File Asal: Evaluasi Standar Checkpoint 2g/2i/v4
# --------------------------------------------------------------------------------------
def load_garch_merged(cv_df, garch_df):
    """
    Menggabungkan prediksi GARCH_Predictions_V4.csv dengan CV_Predictions_V4.csv
    secara ketat pada pasangan (cutoff, real_date) dengan horizon h = cumcount + 1.
    Menjalankan assert total 3350 baris, N=335 per h, dan assert MSE GARCH11 h=1 & h=10.
    """
    cv_sub = cv_df[['cutoff', 'real_date', 'y']].copy()
    g_sub = garch_df[['cutoff', 'real_date', 'GARCH11', 'EGARCH']].copy()
    cv_sub['cutoff'] = pd.to_datetime(cv_sub['cutoff'])
    cv_sub['real_date'] = pd.to_datetime(cv_sub['real_date'])
    g_sub['cutoff'] = pd.to_datetime(g_sub['cutoff'])
    g_sub['real_date'] = pd.to_datetime(g_sub['real_date'])

    m_direct = pd.merge(
        cv_sub,
        g_sub,
        on=['cutoff', 'real_date'],
        how='inner'
    )
    m_direct['h_step'] = m_direct.groupby('cutoff').cumcount() + 1
    
    assert len(m_direct) == 3350, f"Total baris GARCH merged harus 3350, ditemukan {len(m_direct)}"
    for h_val in [1, 10]:
        sub_h = m_direct[m_direct['h_step'] == h_val]
        assert len(sub_h) == 335, f"Jumlah baris h={h_val} harus 335, ditemukan {len(sub_h)}"
    
    sub_1 = m_direct[m_direct['h_step'] == 1]
    sub_10 = m_direct[m_direct['h_step'] == 10]
    mse_g11_1 = mean_squared_error(sub_1['y'], sub_1['GARCH11'])
    mse_g11_10 = mean_squared_error(sub_10['y'], sub_10['GARCH11'])
    
    assert abs(mse_g11_1 - 2.313042e-06) < 1e-10, f"MSE GARCH11 h=1 {mse_g11_1:.6e} != 2.313042e-06"
    assert abs(mse_g11_10 - 2.478223e-06) < 1e-10, f"MSE GARCH11 h=10 {mse_g11_10:.6e} != 2.478223e-06"
    
    return m_direct

# --------------------------------------------------------------------------------------
# 4. FIT & PREDICT HAR OLS
# File Asal: scratch/run_checkpoint2i.py (baris 327-333)
# --------------------------------------------------------------------------------------
def fit_predict_har_ols(X_train, Y_train, x_test):
    """
    Estimasi parameter HAR OLS dengan intercept dan hasilkan prediksi point test.
    scratch/run_checkpoint2i.py baris 327-333:
    X_train_c = sm.add_constant(X_train)
    model = sm.OLS(Y_train, X_train_c).fit(disp=False)
    x_test_c = np.insert(x_test, 0, 1.0)
    pred = model.predict(x_test_c)[0]
    """
    X_train_c = sm.add_constant(X_train, has_constant='add')
    model = sm.OLS(Y_train, X_train_c).fit(disp=False)
    x_test_c = np.insert(x_test, 0, 1.0)
    return model.predict(x_test_c)[0]

# --------------------------------------------------------------------------------------
# 5. METRIK EVALUASI MULTI-DIMENSI
# File Asal: scratch/final/run_final_eval.py (baris 204-210) & Checkpoint 2i
# --------------------------------------------------------------------------------------
def evaluate_metrics(y_true, y_pred, y_hexp=None, y_roll20=None):
    """
    Menghitung matriks metrik evaluasi lengkap:
    - MSE, RMSE, MAE
    - R2_OOS vs Hist_Exp (Campbell-Thompson)
    - R2_OOS vs Roll_20
    - Pred std (variabilitas peramalan)
    - Pred corr vs y_true
    - Pred const MSE (MSE bila model hanya memprediksi nilai rata-rata prediksinya)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = np.array(y_true)[mask]
    yp = np.array(y_pred)[mask]

    if len(yt) < 2:
        return {
            'N': len(yt), 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan,
            'R2_OOS_HistExp': np.nan, 'R2_OOS_Roll20': np.nan,
            'Pred_Std': np.nan, 'Pred_Corr': np.nan, 'Pred_Const_MSE': np.nan
        }

    mse = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    pred_std = np.std(yp)
    corr_val = np.corrcoef(yt, yp)[0, 1] if pred_std > 0 and np.std(yt) > 0 else np.nan
    pred_const_mse = mean_squared_error(yt, np.full_like(yt, np.mean(yp)))

    r2_oos_hexp = np.nan
    if y_hexp is not None:
        yh = np.array(y_hexp)[mask]
        mse_hexp = mean_squared_error(yt, yh)
        if mse_hexp > 0:
            r2_oos_hexp = 1.0 - (mse / mse_hexp)

    r2_oos_roll = np.nan
    if y_roll20 is not None:
        yr = np.array(y_roll20)[mask]
        mse_roll = mean_squared_error(yt, yr)
        if mse_roll > 0:
            r2_oos_roll = 1.0 - (mse / mse_roll)

    return {
        'N': len(yt),
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_OOS_HistExp': r2_oos_hexp,
        'R2_OOS_Roll20': r2_oos_roll,
        'Pred_Std': pred_std,
        'Pred_Corr': corr_val,
        'Pred_Const_MSE': pred_const_mse
    }

# --------------------------------------------------------------------------------------
# 6. UJI DIEBOLD-MARIANO (NEWEY-WEST + HLN + BONFERRONI)
# File Asal: scratch/run_checkpoint2i.py (baris 440-466) & scratch/final/run_final_eval.py (baris 101-133)
# --------------------------------------------------------------------------------------
def diebold_mariano_test(y_true, y_pred_m, y_pred_b, h, m_family=1, use_hln=True):
    """
    Uji komparasi presisi ramalan Diebold-Mariano dengan kuadrat selisih error (MSE loss differential).
    scratch/run_checkpoint2i.py baris 440-466:
    d = (yt - ym)**2 - (yt - yb)**2
    Kovarians Newey-West: lag 0 untuk h=1, lag 9 untuk h=10.
    Koreksi Harvey-Leybourne-Newbold (HLN) untuk sampel kecil (opsional via use_hln).
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_m) | np.isnan(y_pred_b))
    yt = np.array(y_true)[mask]
    ym = np.array(y_pred_m)[mask]
    yb = np.array(y_pred_b)[mask]

    T = len(yt)
    if T < max(3, h + 2):
        return {'md_MSE': 0.0, 'stat': 0.0, 'p_raw': 1.0, 'p_bonf': 1.0, 'N': T, 'hln_applied': False, 'lag': 0}

    # d_t: loss differensial (e_m^2 - e_b^2). md < 0 berarti model m mengungguli model b.
    d = (yt - ym)**2 - (yt - yb)**2
    md = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma = [gamma0]
    lag = 0 if h == 1 else (h - 1)

    for l in range(1, lag + 1):
        gamma.append(np.cov(d[l:], d[:-l])[0, 1])

    var_d = gamma[0] + sum([2 * (1.0 - l / (lag + 1)) * gamma[l] for l in range(1, lag + 1)])
    if var_d <= 0:
        return {'md_MSE': md, 'stat': 0.0, 'p_raw': 1.0, 'p_bonf': 1.0, 'N': T, 'hln_applied': False, 'lag': lag}

    stat = md / np.sqrt(var_d / T)
    # Harvey, Leybourne, Newbold (1997) adjustment (h > 1 dan use_hln=True)
    hln_applied = False
    if h > 1 and use_hln:
        hln = np.sqrt((T + 1 - 2 * h + (h / T) * (h - 1)) / T)
        stat_adj = stat * hln
        hln_applied = True
    else:
        stat_adj = stat

    p_raw = 2.0 * (1.0 - norm.cdf(abs(stat_adj)))
    p_bonf = min(1.0, p_raw * m_family)

    return {
        'md_MSE': md,
        'stat': stat_adj,
        'p_raw': p_raw,
        'p_bonf': p_bonf,
        'N': T,
        'hln_applied': hln_applied,
        'lag': lag
    }
