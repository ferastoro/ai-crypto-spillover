"""
========================================================================================
SKRIPSI S1: DATA COLLECTION PIPELINE
Topik: Deteksi Spillover Volatilitas antara Saham Teknologi AI Global dan Cryptocurrency
       terhadap Pasar Kripto Indonesia Menggunakan Deep Learning
========================================================================================

Deskripsi:
Script ini bertugas mengumpulkan data historis pasar keuangan secara mentah (raw data):
1. Saham AI Global (NVDA, MSFT, TSLA, GOOGL, META) via yfinance.
2. Cryptocurrency Global (BTC-USD, ETH-USD) via yfinance.
3. Variabel Kontekstual & Robustness (^JKSE, ^VIX, IDR=X) via yfinance.
4. Cryptocurrency Lokal Indonesia (BTC/IDR) via Public API Indodax (TradingView endpoint).

Prinsip Pemrosesan:
- Script ini HANYA mengambil data mentah (raw data) apa adanya.
- TIDAK ADA alignment / sinkronisasi kalender antara hari bursa (Senin-Jumat) dan crypto (24/7).
  Penyelarasan kalender dan imputasi missing date dilakukan pada tahap Data Preprocessing terpisah.
- Menangani MultiIndex columns pada versi yfinance terbaru (>= 0.2.x).
- Dilengkapi rate limiting (delay) dan validasi kelengkapan data.
========================================================================================
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

# ======================================================================================
# 1. KONFIGURASI GLOBAL
# ======================================================================================

# Direktori output:
# Otomatis mengarah ke folder data/raw proyek, baik script dijalankan dari root maupun dari folder src/
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR
OUTPUT_DIR = str(_ROOT_DIR / "data" / "raw")

# Rentang waktu data (Cutoff Retrival Data untuk Reprodusibilitas Akademik)
START_DATE = "2020-01-01"

# Tanggal akhir pengambilan data (Data Retrieval Cutoff):
# PENTING UNTUK SKRIPSI: Gunakan tanggal tetap (fixed date) agar observasi di Bab 3 dan Bab 4 konsisten.
# Jika ingin data terbaru hingga hari ini, ubah menjadi: datetime.now().strftime("%Y-%m-%d")
END_DATE = "2026-09-05"

# Delay antar request yfinance (detik) untuk mematuhi rate limit Yahoo Finance
REQUEST_DELAY_SEC = 1.0

# Daftar instrumen yang akan diambil melalui yfinance:
# Format: (Ticker Yahoo Finance, Nama File Output CSV, Kategori/Deskripsi)
YFINANCE_TARGETS: List[Tuple[str, str, str]] = [
    # 1. Saham AI Global
    ("NVDA", "saham_nvda.csv", "Saham AI Global - NVIDIA"),
    ("MSFT", "saham_msft.csv", "Saham AI Global - Microsoft"),
    ("TSLA", "saham_tsla.csv", "Saham AI Global - Tesla"),
    ("GOOGL", "saham_googl.csv", "Saham AI Global - Alphabet (Google)"),
    ("META", "saham_meta.csv", "Saham AI Global - Meta Platforms"),
    # 2. Cryptocurrency Global
    ("BTC-USD", "crypto_btc_usd.csv", "Crypto Global - Bitcoin / USD"),
    ("ETH-USD", "crypto_eth_usd.csv", "Crypto Global - Ethereum / USD"),
    # 3. Variabel Kontekstual / Robustness
    ("^JKSE", "konteks_ihsg.csv", "Robustness - IHSG (Indeks Harga Saham Gabungan)"),
    ("^VIX", "konteks_vix.csv", "Kontekstual - CBOE Volatility Index (VIX)"),
    ("IDR=X", "konteks_usdidr.csv", "Kontekstual - Kurs Nilai Tukar USD/IDR"),
]

# Konfigurasi Indodax BTC/IDR
INDODAX_CONFIG = {
    "filename": "crypto_btc_idr_indodax.csv",
    "description": "Crypto Lokal - Bitcoin / IDR (Indodax)",
    "primary_symbol": "BTCIDR",
    "fallback_symbol": "btc_idr",
    "timeframe": "1D",
    "endpoint_url": "https://indodax.com/tradingview/history_v2",
}


# ======================================================================================
# 2. FUNGSI PENDUKUNG: STANDARISASI DAN SANITASI DATA
# ======================================================================================

def standardize_dataframe(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    """
    Menstandarkan format DataFrame agar memiliki kolom konsisten:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    Menangani:
    - MultiIndex column header dari yfinance >= 0.2.x
    - Parsing dan formatting kolom Date ke format 'YYYY-MM-DD'
    - Urutan kolom standar
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 1. Tangani MultiIndex columns (sering muncul di yfinance versi baru: Level 0 = Price, Level 1 = Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Ambil level pertama (Open, High, Low, Close, Volume)
        df.columns = df.columns.get_level_values(0)
    
    df.columns.name = None

    # 2. Pastikan 'Date' menjadi kolom biasa jika sebelumnya merupakan index
    if "Date" not in df.columns:
        df = df.reset_index()

    # 3. Standardisasi nama kolom (Title-case untuk OHLCV)
    col_mapping = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if c_lower in ["date", "time", "timestamp", "datetime"]:
            col_mapping[col] = "Date"
        elif c_lower in ["open", "o"]:
            col_mapping[col] = "Open"
        elif c_lower in ["high", "h"]:
            col_mapping[col] = "High"
        elif c_lower in ["low", "l"]:
            col_mapping[col] = "Low"
        elif c_lower in ["close", "c"]:
            col_mapping[col] = "Close"
        elif c_lower in ["volume", "vol", "v"]:
            col_mapping[col] = "Volume"
        elif c_lower in ["adj close", "adjclose"]:
            col_mapping[col] = "Adj Close"

    df.rename(columns=col_mapping, inplace=True)

    # 4. Validasi kolom minimal yang diwajibkan
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  [WARNING] Kolom wajib tidak lengkap untuk {source_name}: {missing_cols}")
        return None

    # 5. Konversi kolom Date ke string format 'YYYY-MM-DD' agar bersih dan seragam
    try:
        # Jika kolom Date memiliki timezone info, ubah ke naive / UTC date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"  [WARNING] Gagal memformat kolom Date pada {source_name}: {e}")
        return None

    # 6. Pastikan baris terurut kronologis (ascending) dan tidak ada duplikasi tanggal
    df.sort_values(by="Date", ascending=True, inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Urutkan kolom: kolom wajib di depan, diikuti kolom lain jika ada (misal Adj Close)
    other_cols = [c for c in df.columns if c not in required_cols]
    final_cols = required_cols + other_cols
    df = df[final_cols]

    return df


# ======================================================================================
# 3. FUNGSI PENGAMBILAN DATA (YFINANCE)
# ======================================================================================

def fetch_yfinance_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Mengunduh data harian dari Yahoo Finance menggunakan library yfinance.
    
    Parameter:
    - ticker: Kode instrumen (misal: 'NVDA', 'BTC-USD', '^JKSE')
    - start_date: Tanggal awal 'YYYY-MM-DD'
    - end_date: Tanggal akhir 'YYYY-MM-DD' (inklusif/eksklusif disesuaikan)
    """
    try:
        # yfinance download: end_date pada yf bersifat eksklusif secara default jika berupa string,
        # namun kita berikan tanggal hari ini atau esok untuk memastikan candle hari ini tercakup jika ada.
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=None,  # None akan mengambil hingga candle pasar terakhir yang tersedia
            interval="1d",
            auto_adjust=False,  # Tetap sediakan Open, High, Low, Close, Volume mentah
            progress=False
        )

        if df is None or df.empty:
            print(f"  [ERROR] Data yfinance untuk ticker '{ticker}' kosong!")
            return None

        # Standardisasi kolom & index
        df_clean = standardize_dataframe(df, source_name=f"yfinance:{ticker}")
        if df_clean is not None and not df_clean.empty:
            if start_date:
                df_clean = df_clean[df_clean["Date"] >= start_date]
            if end_date:
                df_clean = df_clean[df_clean["Date"] <= end_date]
            df_clean.reset_index(drop=True, inplace=True)
        return df_clean

    except Exception as e:
        print(f"  [ERROR] Gagal mengunduh yfinance untuk ticker '{ticker}'. Error: {e}")
        return None


# ======================================================================================
# 4. FUNGSI PENGAMBILAN DATA (INDODAX PUBLIC API)
# ======================================================================================

def fetch_indodax_btc_idr(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Mengunduh data historis harian BTC/IDR dari Public API Indodax (TradingView endpoint).
    Endpoint: https://indodax.com/tradingview/history_v2?symbol={symbol}&tf=1D&from={t_from}&to={t_to}
    
    Mekanisme Keandalan:
    - Mengonversi tanggal ke unix timestamp (UTC).
    - Mencoba simbol 'BTCIDR' (standar terkini API v2 Indodax) dan fallback ke 'btc_idr'.
    - Mendukung dua variasi struktur respons:
      1. Format list of dict: [{"Time": ..., "Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ...}, ...]
      2. Format TradingView UDF standard: {"c": [...], "h": [...], "l": [...], "o": [...], "t": [...], "v": [...], "s": "ok"}
    """
    # Konversi tanggal string ke Unix Timestamp (detik)
    try:
        dt_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        t_from = int(dt_start.timestamp())
        
        # Untuk to_timestamp, gunakan waktu saat ini (UTC)
        t_to = int(datetime.now(timezone.utc).timestamp())
    except Exception as e:
        print(f"  [ERROR] Gagal parsing timestamp rentang tanggal Indodax: {e}")
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    # Coba kedua format simbol (BTCIDR dan btc_idr) untuk mengantisipasi perubahan versi API Indodax
    symbols_to_try = [INDODAX_CONFIG["primary_symbol"], INDODAX_CONFIG["fallback_symbol"]]
    raw_response_data = None
    successful_symbol = None

    for sym in symbols_to_try:
        params = {
            "symbol": sym,
            "tf": INDODAX_CONFIG["timeframe"],
            "from": t_from,
            "to": t_to
        }
        url = INDODAX_CONFIG["endpoint_url"]
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                # Cek apakah response berupa list berisi data atau dict yang memiliki data
                if isinstance(data, list) and len(data) > 0:
                    raw_response_data = data
                    successful_symbol = sym
                    break
                elif isinstance(data, dict) and "t" in data and len(data["t"]) > 0:
                    raw_response_data = data
                    successful_symbol = sym
                    break
                else:
                    # Respons kosong atau tidak sesuai
                    continue
            else:
                print(f"  [INFO] Request Indodax dengan simbol '{sym}' return status {resp.status_code}")
        except Exception as e:
            print(f"  [WARNING] Percobaan request Indodax simbol '{sym}' error: {e}")
            continue

    if raw_response_data is None:
        print("  [ERROR] Seluruh percobaan request API Indodax BTC/IDR gagal atau mengembalikan data kosong!")
        return None

    # Parsing struktur respons ke DataFrame
    try:
        if isinstance(raw_response_data, list):
            # Format: [{"Time": 1577836800, "Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ...}, ...]
            df = pd.DataFrame(raw_response_data)
            df.rename(columns={"Time": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        elif isinstance(raw_response_data, dict) and "t" in raw_response_data:
            # Format UDF: {"t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...]}
            df = pd.DataFrame({
                "Date": pd.to_datetime(raw_response_data["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d"),
                "Open": raw_response_data.get("o", []),
                "High": raw_response_data.get("h", []),
                "Low": raw_response_data.get("l", []),
                "Close": raw_response_data.get("c", []),
                "Volume": raw_response_data.get("v", [])
            })
        else:
            print("  [ERROR] Struktur JSON response Indodax tidak dikenali!")
            return None

        # Filter agar data benar-benar berada dalam rentang [start_date, end_date]
        if start_date:
            df = df[df["Date"] >= start_date]
        if end_date:
            df = df[df["Date"] <= end_date]

        # Standardisasi kolom
        df_clean = standardize_dataframe(df, source_name=f"Indodax:{successful_symbol}")
        return df_clean

    except Exception as e:
        print(f"  [ERROR] Gagal memproses data respons Indodax: {e}")
        return None


# ======================================================================================
# 5. PIPELINE UTAMA (EXECUTION & SUMMARY)
# ======================================================================================

def run_data_collection():
    """
    Menjalankan seluruh siklus pengumpulan data:
    1. Membuat direktori output jika belum ada.
    2. Loop pengambilan data yfinance (Saham AI, Crypto Global, Variabel Kontekstual).
    3. Pengambilan data Indodax BTC/IDR.
    4. Menyimpan setiap file ke format CSV.
    5. Mencetak ringkasan dataset lengkap untuk keperluan audit skripsi.
    """
    print("=" * 85)
    print("MEMULAI PROSES PENGUMPULAN DATA MENTAH (RAW DATA COLLECTION)")
    print(f"Topik Skripsi : Spillover Volatilitas Saham AI & Crypto -> Kripto Indonesia")
    print(f"Target Output : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Rentang Waktu : {START_DATE} s.d. {END_DATE}")
    print("=" * 85)

    # Buat direktori output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_records: List[Dict[str, str]] = []

    # ----------------------------------------------------------------------------------
    # TAHAP A: Mengunduh Data yfinance
    # ----------------------------------------------------------------------------------
    print("\n>>> [1/2] Mengunduh Instrumen via Yahoo Finance (yfinance)...")
    for ticker, filename, description in YFINANCE_TARGETS:
        print(f"\n[FETCH] {ticker:8} | {description}")
        filepath = os.path.join(OUTPUT_DIR, filename)

        df = fetch_yfinance_data(ticker=ticker, start_date=START_DATE, end_date=END_DATE)

        if df is not None and not df.empty:
            df.to_csv(filepath, index=False)
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            row_count = len(df)
            print(f"  [SUCCESS] Disimpan ke '{filename}' ({row_count} baris: {min_date} s.d. {max_date})")
            
            summary_records.append({
                "Kategori / Instrumen": description,
                "File CSV": filename,
                "Jumlah Baris": str(row_count),
                "Tanggal Awal": str(min_date),
                "Tanggal Akhir": str(max_date),
                "Status": "BERHASIL"
            })
        else:
            print(f"  [FAILED] Gagal menyimpan data untuk {ticker}!")
            summary_records.append({
                "Kategori / Instrumen": description,
                "File CSV": filename,
                "Jumlah Baris": "0",
                "Tanggal Awal": "-",
                "Tanggal Akhir": "-",
                "Status": "GAGAL"
            })

        # Delay 1 detik antar-request agar tidak terkena IP rate limit
        time.sleep(REQUEST_DELAY_SEC)

    # ----------------------------------------------------------------------------------
    # TAHAP B: Mengunduh Data Indodax BTC/IDR
    # ----------------------------------------------------------------------------------
    print("\n>>> [2/2] Mengunduh BTC/IDR via Indodax Public API...")
    indodax_file = INDODAX_CONFIG["filename"]
    indodax_desc = INDODAX_CONFIG["description"]
    indodax_path = os.path.join(OUTPUT_DIR, indodax_file)
    print(f"\n[FETCH] BTC/IDR  | {indodax_desc}")

    df_indodax = fetch_indodax_btc_idr(start_date=START_DATE, end_date=END_DATE)

    if df_indodax is not None and not df_indodax.empty:
        df_indodax.to_csv(indodax_path, index=False)
        min_date = df_indodax["Date"].min()
        max_date = df_indodax["Date"].max()
        row_count = len(df_indodax)
        print(f"  [SUCCESS] Disimpan ke '{indodax_file}' ({row_count} baris: {min_date} s.d. {max_date})")

        summary_records.append({
            "Kategori / Instrumen": indodax_desc,
            "File CSV": indodax_file,
            "Jumlah Baris": str(row_count),
            "Tanggal Awal": str(min_date),
            "Tanggal Akhir": str(max_date),
            "Status": "BERHASIL"
        })
    else:
        print("  [FAILED] Gagal menyimpan data BTC/IDR Indodax!")
        summary_records.append({
            "Kategori / Instrumen": indodax_desc,
            "File CSV": indodax_file,
            "Jumlah Baris": "0",
            "Tanggal Awal": "-",
            "Tanggal Akhir": "-",
            "Status": "GAGAL"
        })

    # ----------------------------------------------------------------------------------
    # TAHAP C: Ringkasan Hasil Pengumpulan Data (Audit Trail)
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RINGKASAN AKHIR PENGUMPULAN DATA MENTAH (DATA COLLECTION SUMMARY)")
    print("=" * 85)

    summary_df = pd.DataFrame(summary_records)
    # Format tabel rapi di console
    print(summary_df.to_string(index=False))
    print("=" * 85)
    print(f"Total file tersimpan: {len([r for r in summary_records if r['Status'] == 'BERHASIL'])} / {len(summary_records)}")
    print(f"Lokasi folder       : {os.path.abspath(OUTPUT_DIR)}")
    print("Catatan Skripsi     : Data di atas adalah data mentah (raw data). Jangan lakukan alignment")
    print("                      kalender saham vs crypto di sini. Lakukan pada pipeline preprocessing.")
    print("=" * 85)


if __name__ == "__main__":
    run_data_collection()
