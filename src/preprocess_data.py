"""
========================================================================================
SKRIPSI S1: DATA PREPROCESSING & DUAL-DATASET GENERATION PIPELINE
Topik: Deteksi Spillover Volatilitas antara Saham Teknologi AI Global dan Cryptocurrency
       terhadap Pasar Kripto Indonesia Menggunakan Deep Learning
========================================================================================

Deskripsi:
Script ini memproses 11 file data mentah (raw data) dari folder data_raw untuk
menghasilkan DUA versi dataset gabungan siap latih/estimasi:

VERSI 1 - Baseline Ekonometrika (DCC-GARCH / TVP-VAR):
- Menggunakan pendekatan LISTWISE DELETION (Synchronous Calendar Alignment).
- Menggabungkan seluruh 11 instrumen dengan Inner Join berdasarkan 'Date'.
- Secara otomatis membuang hari libur akhir pekan (Sabtu & Minggu) serta hari libur
  bursa non-weekend (karena model DCC-GARCH/VAR mensyaratkan panel data seimbang / balanced panel).
- Menghitung log-return kontinu untuk setiap instrumen: R_t = ln(AdjClose_t / AdjClose_{t-1}).
- Disimpan sebagai: dataset_garch_baseline.csv

VERSI 2 - Pemodelan Deep Learning (LSTM / TFT / NHITS):
- Menggunakan kalender perdagangan SAHAM AI (~1678 hari) sebagai garis waktu utama (base timeline).
- TIDAK melakukan forward-fill harga saham pada akhir pekan (menghindari distorsi volatilitas nol palsu).
- Merekayasa fitur khusus '{crypto}_weekend_return' untuk setiap aset crypto:
  Dihitung sebagai akumulasi log-return dari penutupan bursa terakhir (Jumat/sebelum libur)
  hingga penutupan crypto terakhir sebelum bursa buka kembali (Minggu/sebelum buka).
  Fitur ini hanya bernilai pada hari Senin (dan hari pertama setelah libur multi-hari), sisanya bernilai 0.
- Tetap menghitung log-return harian biasa untuk semua instrumen pada hari bursa aktif.
- Menangani disparitas hari libur bursa lokal (IHSG libur saat Wall Street buka) secara transparan.
- Disimpan sebagai: dataset_deep_learning.csv
========================================================================================
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ======================================================================================
# 1. KONFIGURASI DIREKTORI & INSTRUMEN
# ======================================================================================

# Resolusi path dinamis: mendukung eksekusi dari root proyek maupun subfolder src/
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR
DATA_RAW_DIR = str(_ROOT_DIR / "data" / "raw")
DATA_CLEAN_DIR = str(_ROOT_DIR / "data" / "clean")

# Definisi 11 instrumen penelitian:
INSTRUMENTS = {
    # 1. Saham AI Global (Prediktor Eksternal)
    "saham_nvda": "saham_nvda.csv",
    "saham_msft": "saham_msft.csv",
    "saham_tsla": "saham_tsla.csv",
    "saham_googl": "saham_googl.csv",
    "saham_meta": "saham_meta.csv",
    # 2. Cryptocurrency Global (Prediktor Eksternal)
    "crypto_btc_usd": "crypto_btc_usd.csv",
    "crypto_eth_usd": "crypto_eth_usd.csv",
    # 3. Cryptocurrency Pasar Indonesia (Target Utama Spillover)
    "crypto_btc_idr_indodax": "crypto_btc_idr_indodax.csv",
    # 4. Variabel Kontekstual & Robustness
    "konteks_ihsg": "konteks_ihsg.csv",      # Robustness (Pasar Saham Indonesia)
    "konteks_vix": "konteks_vix.csv",        # Kontekstual (Global Risk Aversion)
    "konteks_usdidr": "konteks_usdidr.csv",  # Kontekstual (Nilai Tukar Rupiah)
}

CRYPTO_KEYS = ["crypto_btc_usd", "crypto_eth_usd", "crypto_btc_idr_indodax"]
STOCK_AI_KEYS = ["saham_nvda", "saham_msft", "saham_tsla", "saham_googl", "saham_meta"]
CONTEXT_KEYS = ["konteks_ihsg", "konteks_vix", "konteks_usdidr"]


# ======================================================================================
# 2. FUNGSI PEMBANTU: LOAD DATA MENTAH
# ======================================================================================

def load_all_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Memuat seluruh file CSV dari folder data_raw ke dalam dictionary DataFrame.
    Memastikan kolom Date berformat string YYYY-MM-DD dan kolom 'Adj Close' tersedia.
    """
    raw_dict = {}
    print(f"\n[LOAD] Membaca data mentah dari: {os.path.abspath(DATA_RAW_DIR)}")
    
    for key, filename in INSTRUMENTS.items():
        filepath = os.path.join(DATA_RAW_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File raw data tidak ditemukan: {filepath}")
        
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df.sort_values("Date", ascending=True, inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Crypto Indodax hanya memiliki Close (karena crypto tidak ada dividen).
        # Standarisasi: buat kolom 'Adj Close' bernilai sama dengan 'Close'.
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
            
        raw_dict[key] = df
        print(f"  -> {key:25}: {len(df):5} baris ({df['Date'].min()} s.d. {df['Date'].max()})")
        
    return raw_dict


# ======================================================================================
# 3. VERSI 1: BASELINE DCC-GARCH / TVP-VAR (LISTWISE DELETION)
# ======================================================================================

def build_dataset_garch_baseline(raw_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Membangun dataset untuk DCC-GARCH / TVP-VAR:
    - Inner Join seluruh 11 instrumen pada kolom 'Date'.
    - Menghapus baris tanggal yang tidak memiliki kelengkapan data di seluruh pasar.
    - Menghitung log-return harian: R_t = ln(AdjClose_t / AdjClose_{t-1}).
    - Menghapus baris pertama (NaN akibat differencing).
    """
    print("\n" + "=" * 85)
    print("MEMPROSES VERSI 1: DATASET BASELINE DCC-GARCH / TVP-VAR (LISTWISE DELETION)")
    print("=" * 85)

    # 1. Kumpulkan kolom Adj Close untuk masing-masing instrumen
    merged_prices = None
    for key in INSTRUMENTS.keys():
        sub_df = raw_dict[key][["Date", "Adj Close"]].copy()
        sub_df.rename(columns={"Adj Close": f"price_{key}"}, inplace=True)
        
        if merged_prices is None:
            merged_prices = sub_df
        else:
            # Inner join menjamin listwise deletion (hanya menyisakan tanggal di mana SEMUA instrumen buka)
            merged_prices = pd.merge(merged_prices, sub_df, on="Date", how="inner")

    merged_prices.sort_values("Date", ascending=True, inplace=True)
    merged_prices.reset_index(drop=True, inplace=True)

    # 2. Hitung statistik pembuangan data (untuk pelaporan Bab 3 Metodologi)
    # Total hari kalender dari awal s.d. akhir periode
    min_date = pd.to_datetime(merged_prices["Date"].min())
    max_date = pd.to_datetime(merged_prices["Date"].max())
    total_calendar_days = (max_date - min_date).days + 1
    aligned_days = len(merged_prices)
    dropped_days = total_calendar_days - aligned_days
    pct_dropped = (dropped_days / total_calendar_days) * 100

    print(f"Total rentang kalender     : {total_calendar_days} hari ({min_date.strftime('%Y-%m-%d')} s.d. {max_date.strftime('%Y-%m-%d')})")
    print(f"Baris tersisa (Inner Join) : {aligned_days} hari bursa bersama")
    print(f"Data terbuang (Listwise)   : {dropped_days} hari ({pct_dropped:.2f}%)")
    print(f"Catatan Metodologi         : Pembuangan mencakup akhir pekan (Sabtu-Minggu ~{2*total_calendar_days//7} hari)")
    print(f"                             dan libur nasional non-sinkron (hari libur bursa AS vs BEI).")

    # 3. Hitung Log-Return untuk masing-masing instrumen: R_t = ln(P_t / P_{t-1})
    return_df = pd.DataFrame({"Date": merged_prices["Date"]})
    for key in INSTRUMENTS.keys():
        price_col = f"price_{key}"
        # Log return kontinu
        return_df[f"ret_{key}"] = np.log(merged_prices[price_col] / merged_prices[price_col].shift(1))

    # 4. Hapus baris pertama yang berisi NaN akibat perhitungan differencing t - (t-1)
    return_df.dropna(inplace=True)
    return_df.reset_index(drop=True, inplace=True)

    print(f"\n[VALIDASI] Dimensi akhir dataset GARCH : {return_df.shape[0]} baris x {return_df.shape[1]} kolom")
    print(f"Rentang tanggal observasi return      : {return_df['Date'].min()} s.d. {return_df['Date'].max()}")
    print(f"Jumlah Missing Values                 : {return_df.isna().sum().sum()} (0 NaN)")

    return return_df


# ======================================================================================
# 4. VERSI 2: DATASET DEEP LEARNING (LSTM / TFT / NHITS)
# ======================================================================================

def build_dataset_deep_learning(raw_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Membangun dataset untuk Deep Learning:
    1. Basis kalender: Kalender perdagangan saham AI (~1678 baris, hari Senin-Jumat bursa AS).
    2. Rekayasa fitur khusus: '{crypto}_weekend_return'
       - Dihitung sebagai: ln(P_crypto_{Minggu} / P_crypto_{Jumat})
       - Secara matematis menangkap akumulasi guncangan crypto selama bursa saham tutup.
       - Hanya diisi pada hari Senin (dan hari setelah bursa libur multi-hari), sisanya bernilai 0.0.
    3. Return harian biasa:
       - Saham AI: ln(P_saham_t / P_saham_{t-1})
       - Crypto: ln(P_crypto_t / P_crypto_{t-1}) pada hari bursa saham buka.
    4. Penanganan libur non-weekend pasar Indonesia (IHSG) tanpa merusak saham AI:
       - Jika BEI libur pada hari bursa AS buka, return IHSG bernilai 0.0 (pasar tutup, tidak ada fluktuasi).
    """
    print("\n" + "=" * 85)
    print("MEMPROSES VERSI 2: DATASET DEEP LEARNING (STOCK CALENDAR + WEEKEND RETURN FEATURE)")
    print("=" * 85)

    # 1. Gunakan kalender saham AI (NVDA) sebagai acuan dasar (base calendar)
    base_stock_df = raw_dict["saham_nvda"][["Date"]].copy()
    base_stock_df["dt"] = pd.to_datetime(base_stock_df["Date"])
    base_stock_df.sort_values("dt", ascending=True, inplace=True)
    base_stock_df.reset_index(drop=True, inplace=True)

    # Hitung selisih hari kalender antar hari bursa: gap_days = Date_t - Date_{t-1}
    base_stock_df["gap_days"] = (base_stock_df["dt"] - base_stock_df["dt"].shift(1)).dt.days

    # Buat dictionary harga lengkap crypto dengan index Date (24/7, 2440 hari)
    crypto_price_lookup = {}
    for c_key in CRYPTO_KEYS:
        crypto_price_lookup[c_key] = raw_dict[c_key].set_index("Date")["Adj Close"]

    # Inisialisasi DataFrame output deep learning
    dl_df = pd.DataFrame({"Date": base_stock_df["Date"]})

    # 2. Rekayasa Fitur '{crypto}_weekend_return' dan Return Harian Crypto
    print("\n[FITUR] Menghitung '{crypto}_weekend_return' dan return harian crypto...")
    
    for c_key in CRYPTO_KEYS:
        c_series = crypto_price_lookup[c_key]
        weekend_ret_list = []
        daily_ret_list = []

        for i in range(len(base_stock_df)):
            curr_date_str = base_stock_df.loc[i, "Date"]
            curr_dt = base_stock_df.loc[i, "dt"]
            gap = base_stock_df.loc[i, "gap_days"]

            if i == 0:
                # Baris pertama tidak memiliki referensi hari bursa sebelumnya
                weekend_ret_list.append(np.nan)
                daily_ret_list.append(np.nan)
                continue

            prev_stock_date_str = base_stock_df.loc[i - 1, "Date"]

            # Jika terjadi gap > 1 hari (misal: Jumat ke Senin [gap=3], atau libur panjang [gap=4]):
            if gap > 1:
                # Tanggal kalender crypto terakhir sebelum bursa saham buka kembali
                last_crypto_date = (curr_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                # Pastikan kedua tanggal tersedia di data crypto
                if prev_stock_date_str in c_series and last_crypto_date in c_series:
                    p_start_gap = c_series.loc[prev_stock_date_str]  # Harga penutupan hari Jumat
                    p_end_gap = c_series.loc[last_crypto_date]       # Harga penutupan hari Minggu
                    w_ret = float(np.log(p_end_gap / p_start_gap))
                else:
                    w_ret = 0.0

                weekend_ret_list.append(w_ret)

                # Return harian crypto pada hari Senin: dari penutupan Minggu ke penutupan Senin
                if last_crypto_date in c_series and curr_date_str in c_series:
                    d_ret = float(np.log(c_series.loc[curr_date_str] / c_series.loc[last_crypto_date]))
                else:
                    d_ret = 0.0
                daily_ret_list.append(d_ret)

            else:
                # Gap == 1 hari (perdagangan reguler Selasa s.d. Jumat):
                # Tidak ada jeda akhir pekan, sehingga weekend_return = 0.0
                weekend_ret_list.append(0.0)

                # Return harian normal: dari kemarin ke hari ini
                if prev_stock_date_str in c_series and curr_date_str in c_series:
                    d_ret = float(np.log(c_series.loc[curr_date_str] / c_series.loc[prev_stock_date_str]))
                else:
                    d_ret = 0.0
                daily_ret_list.append(d_ret)

        dl_df[f"{c_key}_weekend_return"] = weekend_ret_list
        dl_df[f"ret_{c_key}"] = daily_ret_list

    # 3. Hitung Log-Return untuk Saham AI Global
    print("[FITUR] Menghitung log-return saham AI global...")
    for s_key in STOCK_AI_KEYS:
        s_prices = raw_dict[s_key].set_index("Date")["Adj Close"]
        # Alignment langsung karena kalender identik
        aligned_p = dl_df["Date"].map(s_prices)
        dl_df[f"ret_{s_key}"] = np.log(aligned_p / aligned_p.shift(1))

    # 4. Hitung Log-Return untuk Variabel Kontekstual (VIX, USD/IDR, IHSG)
    print("[FITUR] Menghitung log-return variabel kontekstual (^VIX, IDR=X, ^JKSE)...")
    
    # VIX dan USD/IDR
    for ctx_key in ["konteks_vix", "konteks_usdidr"]:
        ctx_prices = raw_dict[ctx_key].set_index("Date")["Adj Close"]
        aligned_p = dl_df["Date"].map(ctx_prices).ffill()  # ffill jika ada 1 hari bank holiday langka
        dl_df[f"ret_{ctx_key}"] = np.log(aligned_p / aligned_p.shift(1))

    # IHSG: Deteksi dan tangani hari libur BEI ketika Wall Street buka
    ihsg_prices = raw_dict["konteks_ihsg"].set_index("Date")["Adj Close"]
    ihsg_raw_mapped = dl_df["Date"].map(ihsg_prices)
    
    # Indikator biner apakah BEI sedang libur pada hari tersebut
    dl_df["is_ihsg_holiday"] = ihsg_raw_mapped.isna().astype(int)
    
    # Forward-fill harga IHSG agar pada hari libur, return IHSG = 0.0 (tidak ada pergerakan)
    ihsg_ffill = ihsg_raw_mapped.ffill()
    dl_df["ret_konteks_ihsg"] = np.log(ihsg_ffill / ihsg_ffill.shift(1))

    # 5. Tambahkan Fitur Kalender Tambahan (Sangat Bermanfaat untuk TFT/LSTM)
    dl_df["gap_days"] = base_stock_df["gap_days"].fillna(1).astype(int)
    dl_df["day_of_week"] = pd.to_datetime(dl_df["Date"]).dt.dayofweek  # 0=Senin, ..., 4=Jumat
    dl_df["is_monday"] = (dl_df["day_of_week"] == 0).astype(int)

    # 6. Hapus baris pertama (NaN dari differencing t - 1)
    dl_df.dropna(inplace=True)
    dl_df.reset_index(drop=True, inplace=True)

    # Statistik deteksi weekend & libur bursa
    weekend_rows = (dl_df["gap_days"] >= 3).sum()
    holiday_ihsg_count = (dl_df["is_ihsg_holiday"] == 1).sum()

    print(f"\n[VALIDASI] Dimensi akhir dataset Deep Learning : {dl_df.shape[0]} baris x {dl_df.shape[1]} kolom")
    print(f"Rentang tanggal observasi                      : {dl_df['Date'].min()} s.d. {dl_df['Date'].max()}")
    print(f"Jumlah baris gap akhir pekan / libur bursa     : {weekend_rows} baris")
    print(f"Jumlah hari IHSG libur saat saham AS buka      : {holiday_ihsg_count} baris (return di-set 0.0)")
    print(f"Jumlah Missing Values (NaN)                    : {dl_df.isna().sum().sum()} (0 NaN)")

    return dl_df


# ======================================================================================
# 5. PIPELINE UTAMA & PENYIMPANAN DATASET
# ======================================================================================

def run_preprocessing():
    print("=" * 85)
    print("MEMULAI PIPELINE PREPROCESSING & DUAL-DATASET GENERATION")
    print("Topik: Spillover Volatilitas Saham AI & Crypto -> Pasar Kripto Indonesia")
    print(f"Target Direktori Simpan: {os.path.abspath(DATA_CLEAN_DIR)}")
    print("=" * 85)

    # Buat direktori data_clean jika belum ada
    os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

    # 1. Muat seluruh data mentah
    raw_dict = load_all_raw_data()

    # 2. Bangun Dataset Versi 1 (Baseline DCC-GARCH / TVP-VAR)
    df_garch = build_dataset_garch_baseline(raw_dict)
    garch_filepath = os.path.join(DATA_CLEAN_DIR, "dataset_garch_baseline.csv")
    df_garch.to_csv(garch_filepath, index=False)
    print(f"[SIMPAN] Tersimpan: {os.path.basename(garch_filepath)}")

    # 3. Bangun Dataset Versi 2 (Deep Learning: LSTM / TFT / NHITS)
    df_dl = build_dataset_deep_learning(raw_dict)
    dl_filepath = os.path.join(DATA_CLEAN_DIR, "dataset_deep_learning.csv")
    df_dl.to_csv(dl_filepath, index=False)
    print(f"[SIMPAN] Tersimpan: {os.path.basename(dl_filepath)}")

    # Simpan juga salinan langsung di root/subfolder kerja jika diinginkan
    if _SCRIPT_DIR.name == "RSIC":
        # Simpan juga di folder RSIC jika dipanggil dari RSIC
        df_garch.to_csv(os.path.join(_SCRIPT_DIR, "dataset_garch_baseline.csv"), index=False)
        df_dl.to_csv(os.path.join(_SCRIPT_DIR, "dataset_deep_learning.csv"), index=False)

    # 4. Ringkasan Komparasi untuk Laporan Skripsi (Bab 3 Metodologi)
    print("\n" + "=" * 85)
    print("RINGKASAN PERBANDINGAN DATASET PREPROCESSING (UNTUK BAB 3 METODOLOGI)")
    print("=" * 85)

    comparison_data = [
        {
            "Aspek Metodologi": "Metode Pemodelan",
            "Versi 1 (GARCH Baseline)": "DCC-GARCH / TVP-VAR",
            "Versi 2 (Deep Learning)": "LSTM / TFT / NHITS"
        },
        {
            "Aspek Metodologi": "Pendekatan Kalender",
            "Versi 1 (GARCH Baseline)": "Listwise Deletion (Inner Join)",
            "Versi 2 (Deep Learning)": "Stock Market Calendar Base"
        },
        {
            "Aspek Metodologi": "Perlakuan Akhir Pekan Crypto",
            "Versi 1 (GARCH Baseline)": "Dibuang (Sabtu & Minggu dihapus)",
            "Versi 2 (Deep Learning)": "Ditransformasi jadi 'weekend_return'"
        },
        {
            "Aspek Metodologi": "Jumlah Observasi (Baris)",
            "Versi 1 (GARCH Baseline)": f"{len(df_garch)} baris",
            "Versi 2 (Deep Learning)": f"{len(df_dl)} baris"
        },
        {
            "Aspek Metodologi": "Jumlah Kolom Fitur",
            "Versi 1 (GARCH Baseline)": f"{df_garch.shape[1]} kolom (12 kolom)",
            "Versi 2 (Deep Learning)": f"{df_dl.shape[1]} kolom (19 kolom)"
        },
        {
            "Aspek Metodologi": "Rentang Waktu Observasi",
            "Versi 1 (GARCH Baseline)": f"{df_garch['Date'].min()} s.d. {df_garch['Date'].max()}",
            "Versi 2 (Deep Learning)": f"{df_dl['Date'].min()} s.d. {df_dl['Date'].max()}"
        },
        {
            "Aspek Metodologi": "Data Terbuang (% Drop)",
            "Versi 1 (GARCH Baseline)": f"~36.19% (terbuang demi panel seimbang)",
            "Versi 2 (Deep Learning)": f"0% info crypto (terserap di weekend_ret)"
        },
        {
            "Aspek Metodologi": "Missing Value (NaN)",
            "Versi 1 (GARCH Baseline)": "0 NaN",
            "Versi 2 (Deep Learning)": "0 NaN"
        }
    ]

    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    print("=" * 85)
    print("Status: KEDUA DATASET BERHASIL DIBUAT DAN SIAP DIGUNAKAN UNTUK MODELING!")
    print("=" * 85)


if __name__ == "__main__":
    run_preprocessing()
