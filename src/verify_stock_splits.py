"""
========================================================================================
AUDIT & VERIFIKASI STOCK SPLIT & INTEGRITAS DATA HARGA
Topik: Deteksi Spillover Volatilitas Saham AI Global & Crypto -> Pasar Kripto Indonesia
========================================================================================

Tujuan Script:
1. Memverifikasi apakah data saham (NVDA, MSFT, TSLA, GOOGL, META) di folder data_raw
   sudah ter-adjust untuk peristiwa Stock Split historis:
   - NVDA : Split 10:1 pada 10 Juni 2024 (dan 4:1 pada 20 Juli 2021)
   - TSLA : Split 3:1 pada 25 Agustus 2022 (dan 5:1 pada 31 Agustus 2020)
   - GOOGL: Split 20:1 pada 18 Juli 2022
2. Menguji keberadaan anomali lompatan harga harian (|return| > 35%) yang berpotensi
   menjadi "false crash" jika split belum ter-adjust.
3. Membandingkan kolom 'Close' vs 'Adj Close' untuk panduan pemodelan volatilitas.
========================================================================================
"""

import os
import pandas as pd
import numpy as np

# Lokasi folder data: otomatis mendeteksi folder data/raw utama proyek
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR
DATA_DIR = str(_ROOT_DIR / "data" / "raw")

STOCKS_CONFIG = [
    {
        "ticker": "NVDA",
        "file": "saham_nvda.csv",
        "known_splits": [
            {"date": "2024-06-10", "ratio": "10:1", "window": ("2024-06-05", "2024-06-14")},
            {"date": "2021-07-20", "ratio": "4:1", "window": ("2021-07-15", "2021-07-23")}
        ]
    },
    {
        "ticker": "TSLA",
        "file": "saham_tsla.csv",
        "known_splits": [
            {"date": "2022-08-25", "ratio": "3:1", "window": ("2022-08-22", "2022-08-29")},
            {"date": "2020-08-31", "ratio": "5:1", "window": ("2020-08-26", "2020-09-03")}
        ]
    },
    {
        "ticker": "GOOGL",
        "file": "saham_googl.csv",
        "known_splits": [
            {"date": "2022-07-18", "ratio": "20:1", "window": ("2022-07-13", "2022-07-21")}
        ]
    },
    {
        "ticker": "MSFT",
        "file": "saham_msft.csv",
        "known_splits": []  # Tidak ada split selama periode 2020-sekarang
    },
    {
        "ticker": "META",
        "file": "saham_meta.csv",
        "known_splits": []  # Tidak ada split selama periode 2020-sekarang
    }
]

def verify_stock_splits():
    print("=" * 85)
    print("AUDIT METODOLOGI: VERIFIKASI STOCK SPLIT & INTEGRITAS RETURN HARGA")
    print(f"Direktori Data: {os.path.abspath(DATA_DIR)}")
    print("=" * 85)

    all_safe = True

    for item in STOCKS_CONFIG:
        ticker = item["ticker"]
        filepath = os.path.join(DATA_DIR, item["file"])

        if not os.path.exists(filepath):
            print(f"[SKIP] File {item['file']} tidak ditemukan di {DATA_DIR}!")
            all_safe = False
            continue

        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Hitung simple daily return untuk Close dan Adj Close
        df["ret_close"] = df["Close"].pct_change()
        df["ret_adj"] = df["Adj Close"].pct_change()

        print(f"\n>>> [ANALISIS] {ticker} ({item['file']}) - Total {len(df)} baris")

        # 1. Cek Anomali Ekstrem (|return| > 35%)
        # Jika stock split 2:1 belum diadjust, return hari itu adalah -50%.
        # Jika split 10:1 belum diadjust, return hari itu adalah -90%.
        anomalies_close = df[df["ret_close"].abs() > 0.35]
        anomalies_adj = df[df["ret_adj"].abs() > 0.35]

        if anomalies_close.empty and anomalies_adj.empty:
            print("  [OK] Tidak ada anomali lompatan harga > 35% pada seluruh rentang data.")
            print("       -> Ini membuktikan harga telah disesuaikan secara retroaktif (split-adjusted).")
        else:
            all_safe = False
            print("  [PERINGATAN] Ditemukan lonjakan harian ekstrem:")
            for _, row in anomalies_close.iterrows():
                print(f"    - Tanggal: {row['Date'].strftime('%Y-%m-%d')} | Return Close: {row['ret_close']*100:.2f}% | Return Adj: {row['ret_adj']*100:.2f}%")

        # 2. Cek Sampel Tanggal Stock Split Spesifik
        if item["known_splits"]:
            for split_info in item["known_splits"]:
                s_date = split_info["date"]
                s_ratio = split_info["ratio"]
                w_start, w_end = split_info["window"]

                window_df = df[(df["Date"] >= w_start) & (df["Date"] <= w_end)]
                print(f"\n  * Peristiwa Split {s_ratio} pada {s_date}:")
                for _, row in window_df.iterrows():
                    d_str = row["Date"].strftime("%Y-%m-%d")
                    tag = " <-- EX-DATE SPLIT" if d_str == s_date else ""
                    print(f"    {d_str} | Close: ${row['Close']:8.3f} | Adj Close: ${row['Adj Close']:8.3f} | Ret: {row['ret_close']*100:+6.2f}%{tag}")

        print("-" * 85)

    print("\n" + "=" * 85)
    print("KESIMPULAN AUDIT:")
    if all_safe:
        print("[STATUS: AMAN] Seluruh data saham AI telah ter-adjust terhadap peristiwa stock split.")
        print("Penjelasan Teknis:")
        print("1. Library 'yfinance' secara otomatis menerapkan penyesuaian split retroaktif pada")
        print("   kolom Open, High, Low, dan Close.")
        print("2. Kolom 'Close' = Split-adjusted.")
        print("3. Kolom 'Adj Close' = Split-adjusted DAN Dividend-adjusted (total return price).")
        print("Rekomendasi untuk Skripsi:")
        print("- Gunakan 'Adj Close' saat menghitung log-return: R_t = ln(Adj_Close_t / Adj_Close_{t-1})")
        print("  karena mencerminkan total return yang bersih dari distorsi dividen maupun split.")
    else:
        print("[STATUS: PERLU PERHATIAN] Periksa anomali yang tercetak di atas sebelum masuk modeling.")
    print("=" * 85)

if __name__ == "__main__":
    verify_stock_splits()
