# Laporan Audit Penyelarasan Waktu (Timestamp Alignment) Pipeline RSIC

**Tanggal Audit:** 16 Oktober 2026  
**Status Audit:** READ-ONLY AUDIT (Tanpa Perubahan Dataset, Tanpa Rerun Pelatihan, Tanpa Panggilan API Eksternal)  
**Tujuan Audit:** Memeriksa secara mendalam konvensi tanggal, potensi ketidakselarasan waktu (*timestamp misalignment* / *lead-lag dislocation*), dan konsekuensi metodologis penggabungan data antara BTC/IDR Indodax, BTC-USD/ETH-USD global, dan 5 saham AI bursa Amerika Serikat.  
**Artefak Pendukung yang Dihasilkan:**
- `audit/leadlag_pearson.csv`
- `audit/leadlag_spearman.csv`
- `audit/leadlag_summary.csv`
- `audit/leadlag_heatmap_pearson.png`
- `audit/leadlag_heatmap_spearman.png`
- `audit/leadlag_comparison.png`

---

## 1. Daftar Berkas Input dan Verifikasi Hash Kriptografis (MD5)

Berikut adalah ringkasan seluruh berkas kode, data mentah, data bersih, dan notebook yang diaudit beserta nilai hash MD5 resminya:

| No | Path Berkas | Peran dalam Pipeline | MD5 Hash |
| :-: | :--- | :--- | :--- |
| 1 | `src/collect_data.py` | Skrip pengumpulan data mentah dari API | `931836309643e98b0bacf52f6940bd43` |
| 2 | `src/preprocess_data.py` | Skrip penyelarasan kalender & perhitungan return | `d7903d914fbc3e2a8d2acdcbc15f669a` |
| 3 | `data/raw/crypto_btc_idr_indodax.csv` | Data mentah harian BTC/IDR Indodax | `45bb5f212fa7e401d73fc5e6909c69f4` |
| 4 | `data/raw/crypto_btc_usd.csv` | Data mentah harian BTC-USD yfinance | `989c51f781982ca3715efba61222278b` |
| 5 | `data/raw/crypto_eth_usd.csv` | Data mentah harian ETH-USD yfinance | `ed773c4eaa057657f4948b800a8432f9` |
| 6 | `data/raw/saham_nvda.csv` | Data mentah harian saham NVIDIA yfinance | `b57c0bf69409a3c6d14bb1fdac44e054` |
| 7 | `data/raw/saham_msft.csv` | Data mentah harian saham Microsoft yfinance | `187857434608fb442e97fb12af5204b9` |
| 8 | `data/raw/saham_tsla.csv` | Data mentah harian saham Tesla yfinance | `71b3149375fc0a7469d7889bdc33a3f8` |
| 9 | `data/raw/saham_googl.csv` | Data mentah harian saham Alphabet yfinance | `10f2e090065e87a8658da84fe2d63b91` |
| 10 | `data/raw/saham_meta.csv` | Data mentah harian saham Meta Platforms yfinance | `5c8ae92d957af9902a4de0e53b2fc315` |
| 11 | `data/raw/konteks_ihsg.csv` | Data mentah harian IHSG (^JKSE) yfinance | `899198d6cad4f999b6a6153c82440049` |
| 12 | `data/raw/konteks_vix.csv` | Data mentah harian CBOE VIX (^VIX) yfinance | `955eacec7504f3b41b0876ffa692f5e3` |
| 13 | `data/raw/konteks_usdidr.csv` | Data mentah harian kurs USD/IDR (IDR=X) yfinance | `85ac3879bf5ce3e6a95995a725871fc2` |
| 14 | `data/clean/dataset_garch_baseline.csv` | Dataset gabungan Versi 1 (Inner Join, RQ1) | `90238c4edddc13fb2f6645d087b7da43` |
| 15 | `data/clean/dataset_deep_learning.csv` | Dataset gabungan Versi 2 (Stock Base, DL) | `ba64297407f9d8221488b4815a12ce8c` |
| 16 | `tvp_var_spillover.ipynb` | Notebook estimasi TVP-VAR RQ1 | `079317c077dc842bc7473f465062a519` |

---

## 2. LANGKAH 1: Dokumentasi Konvensi Tanggal dan Mekanisme Penggabungan

Pemeriksaan kode dilakukan langsung pada berkas sumber `src/collect_data.py` dan `src/preprocess_data.py`.

### 2.1 Indodax BTC/IDR
1. **Endpoint & Parameter Resolusi:**  
   Berdasarkan `src/collect_data.py` baris 76–83:
   ```python
   INDODAX_CONFIG = {
       "filename": "crypto_btc_idr_indodax.csv",
       "description": "Crypto Lokal - Bitcoin / IDR (Indodax)",
       "primary_symbol": "BTCIDR",
       "fallback_symbol": "btc_idr",
       "timeframe": "1D",
       "endpoint_url": "https://indodax.com/tradingview/history_v2",
   }
   ```
   Parameter resolusi yang diminta adalah `tf = "1D"` (resolusi candle harian).
2. **Format Timestamp Mentah:**  
   Berdasarkan `src/collect_data.py` baris 249–299, endpoint `https://indodax.com/tradingview/history_v2` mengembalikan stempel waktu unix epoch dalam satuan detik (`Time` atau `t`). Contoh format yang didukung oleh kode:
   - Format list of dict: `[{"Time": 1577836800, "Open": ..., ...}]`
   - Format TradingView UDF dict: `{"t": [1577836800, ...], "o": [...], ...}`
3. **Konversi ke Kolom Date (UTC vs WIB):**  
   Berdasarkan baris 289 dan 293:
   ```python
   df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
   ```
   dan baris 293:
   ```python
   "Date": pd.to_datetime(raw_response_data["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
   ```
   Lalu pada `standardize_dataframe()` (baris 147–149):
   ```python
   df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
   df.dropna(subset=["Date"], inplace=True)
   df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
   ```
   *Temuan Kritis:* Skrip secara eksplisit mem-parsing epoch detik dengan parameter `utc=True`, kemudian langsung memformatnya menjadi string `'YYYY-MM-DD'`. **TIDAK ADA konversi timezone ke Waktu Indonesia Barat (`Asia/Jakarta` atau UTC+7) yang dilakukan di dalam kode.**
4. **Contoh 5 Baris Data Mentah ke Hasil:**  
   Berdasarkan berkas tersimpan `data/raw/crypto_btc_idr_indodax.csv` baris 1 s.d. 6:
   ```csv
   Date,Open,High,Low,Close,Volume
   2020-01-01,100476000,102567000,100090000,101071000,246.42471806
   2020-01-02,101072000,101300000,97746000,98000000,239.56689480
   2020-01-03,98001000,102999000,97000000,101698000,286.83209243
   2020-01-04,101799000,102242000,101010000,101995000,128.18656464
   2020-01-05,101995000,103400000,101301000,101701000,137.95587831
   ```
   Nilai epoch mentah `1577836800` setara dengan `2020-01-01 00:00:00 UTC` (atau `2020-01-01 07:00:00 WIB`). Karena kode memotong format tanggal pada `utc=True`, label tanggal yang tersimpan di kolom `Date` adalah `2020-01-01`.

### 2.2 Cryptocurrency Global (BTC-USD dan ETH-USD via yfinance)
1. **Pengunduhan & Parameter:**  
   Berdasarkan `src/collect_data.py` baris 183–190:
   ```python
   df = yf.download(
       tickers=ticker,
       start=start_date,
       end=None,
       interval="1d",
       auto_adjust=False,
       progress=False
   )
   ```
2. **Konvensi Lilin (Candle):**  
   Standar library `yfinance` untuk instrumen cryptocurrency 24/7 memotong lilin harian dari pukul **00:00:00 UTC hingga 23:59:59 UTC**, diberi label tanggal awal periode tersebut (`YYYY-MM-DD`). Kolom waktu diubah menjadi string tanggal tanpa jam oleh `standardize_dataframe()` (baris 149).

### 2.3 Saham AI Global Amerika Serikat (NVDA, MSFT, TSLA, GOOGL, META)
1. **Bursa & Jam Perdagangan:**  
   Saham diperdagangkan di bursa NASDAQ (New York, zona waktu US Eastern Time / ET).
   - Jam perdagangan reguler: **09:30 s.d. 16:00 ET**.
   - Dalam UTC:
     - Waktu Musim Panas (*Eastern Daylight Time / EDT, UTC-4*): **13:30 s.d. 20:00 UTC**.
     - Waktu Standar (*Eastern Standard Time / EST, UTC-5*): **14:30 s.d. 21:00 UTC**.
2. **Label Tanggal:** Tanggal bursa lokal New York (`YYYY-MM-DD`). Penutupan lilin perdagangan terjadi pada pukul 16:00 waktu setempat (pukul 20:00 atau 21:00 UTC pada tanggal yang sama).

### 2.4 Cara Penggabungan Data (Merging Logic)
Diimplementasikan dalam `src/preprocess_data.py`:

1. **Versi 1: Baseline Ekonometrika (`dataset_garch_baseline.csv`):**
   * **Mekanisme:** *Inner Join* (*listwise deletion*) pada kolom `Date` di 11 instrumen (baris 137–138):
     ```python
     merged_prices = pd.merge(merged_prices, sub_df, on="Date", how="inner")
     ```
   * **Perlakuan Akhir Pekan & Hari Libur:** Seluruh hari Sabtu, Minggu, dan hari libur non-sinkron (hari libur Wall Street atau libur BEI) otomatis dibuang. Dari total 2.440 hari kalender, tersisa 1.557 hari harga bersama (~36.19% data dibuang, baris 147–156).
   * **Perhitungan Return & Fenomena Hari Senin:**
     Berdasarkan baris 163:
     ```python
     return_df[f"ret_{key}"] = np.log(merged_prices[price_col] / merged_prices[price_col].shift(1))
     ```
     *Fakta Kode:* Karena `shift(1)` dilakukan langsung pada tabel hasil *inner join*, maka untuk hari Senin (atau hari perdagangan pertama setelah libur), observasi $t-1$ adalah hari Jumat sebelumnya!  
     **Dengan demikian, return hari Senin pada `dataset_garch_baseline.csv` secara matematis mencakup return akumulasi 3 hari kalender (72 jam: Jumat penutupan s.d. Senin penutupan), baik untuk saham maupun untuk cryptocurrency.** Sebanyak 1 baris pertama terbuang akibat differencing, menyisakan $N = 1.556$ baris observasi aktif (2020-01-03 s.d. 2026-09-04).

2. **Versi 2: Dataset Deep Learning (`dataset_deep_learning.csv`):**
   * **Mekanisme:** Garis waktu mengacu pada kalender perdagangan saham AI (`saham_nvda`, baris 198–202).
   * **Perlakuan Akhir Pekan:** Pergerakan harga crypto akhir pekan tidak dibuang, melainkan direkayasa menjadi fitur terpisah `crypto_weekend_return`:
     - Baris 243–245: $P_{\text{start}} = P_{\text{Jumat}}$, $P_{\text{end}} = P_{\text{Minggu}}$, dihitung sebagai $\ln(P_{\text{Minggu}} / P_{\text{Jumat}})$.
     - Baris 253: Return crypto hari Senin dihitung terpisah dari penutupan Minggu ke Senin: $\ln(P_{\text{Senin}} / P_{\text{Minggu}})$.
     - Pada hari biasa (Selasa s.d. Jumat, `gap_days == 1`), `weekend_return` bernilai `0.0`, dan return crypto adalah return harian biasa $\ln(P_t / P_{t-1})$.

### 2.5 Tabel Ringkasan Konvensi Waktu dan Penutupan Lilin (Candle)

| Aset | Sumber Data | Buka–Tutup Lilin (UTC) menurut Kode / Bukti | Label Tanggal Dataset | Bukti (Berkas:Baris) |
| :--- | :--- | :--- | :---: | :--- |
| **BTC-IDR** | Indodax Public API (`history_v2`) | Jam potong harian: **UNVERIFIED** (Kode mengonversi epoch detik dengan `utc=True`, tanpa stempel jam intra-hari di CSV) | `YYYY-MM-DD` | `src/collect_data.py`: baris 81–82, 289, 293 |
| **BTC-USD** | yfinance | 00:00:00 s.d. 23:59:59 UTC (*fakta umum yf crypto*; dalam file jam tidak tersimpan: **UNVERIFIED**) | `YYYY-MM-DD` | `src/collect_data.py`: baris 67, 183–190 |
| **ETH-USD** | yfinance | 00:00:00 s.d. 23:59:59 UTC (*fakta umum yf crypto*; dalam file jam tidak tersimpan: **UNVERIFIED**) | `YYYY-MM-DD` | `src/collect_data.py`: baris 68, 183–190 |
| **Saham AI (5 aset)** | yfinance (NASDAQ) | 13:30 s.d. 20:00 UTC (EDT / Musim Panas) atau 14:30 s.d. 21:00 UTC (EST / Musim Dingin). Jam tutup lokal: 16:00 ET (*fakta umum bursa AS*; dalam file jam tidak tersimpan: **UNVERIFIED**) | `YYYY-MM-DD` | `src/collect_data.py`: baris 61–65, 183–190 |
| **IHSG (^JKSE)** | yfinance (BEI) | 02:00 s.d. 09:00 UTC (09:00 s.d. 16:00 WIB, *fakta umum BEI*; dalam file jam tidak tersimpan: **UNVERIFIED**) | `YYYY-MM-DD` | `src/collect_data.py`: baris 70 |

---

## 3. LANGKAH 2: Diagnostik Empiris Lead-Lag (Tanpa Retraining)

### 3.1 Prosedur dan Spesifikasi Pengujian
- **Dataset Input:** `data/clean/dataset_garch_baseline.csv` ($N = 1.556$ baris observasi aktif, rentang tanggal 2020-01-03 s.d. 2026-09-04).
- **Transformasi Variabel (Sesuai RQ1 di `tvp_var_spillover.ipynb` baris 108–125):**
  Untuk tiap aset $i$:
  $$y_{i,t} = \log\left(r_{i,t}^2 + 10^{-10}\right)$$
  $$z_{i,t} = \frac{y_{i,t} - \mu_i}{\sigma_i}$$
- **Indeks Komposit Saham AI (`ai_index`):** Rata-rata sederhana dari $z$-score volatilitas kelima saham AI (NVDA, MSFT, TSLA, GOOGL, META).
- **Definisi Lag $k$ ($k \in \{-3, -2, -1, 0, +1, +2, +3\}$):**
  - $k < 0$ (misal $k = -1$): Fitur saham/prediktor berada pada hari $t-1$ (*kemarin*), dipasangkan dengan target pada hari $t$. Ini merefleksikan hipotesis bahwa pergerakan saham kemarin menuntun (*leads*) target hari ini.
  - $k = 0$: Hubungan kontemporer (hari perdagangan $t$ yang sama pada tabel gabungan).
  - $k > 0$ (misal $k = +1$): Fitur prediktor berada pada hari $t+1$ (*esok hari*), dipasangkan dengan target pada hari $t$.
- **Batas Signifikansi Kasar (95% Confidence Level):**
  $$\text{Threshold} = \pm \frac{1.96}{\sqrt{N}} \approx \pm \frac{1.96}{\sqrt{1556}} \approx \pm 0.0497$$

---

### 3.2 Hasil Korelasi Lead-Lag: BTC-IDR vs Saham AI & Indeks AI

Tabel berikut menyajikan nilai korelasi Pearson ($r$) dan Spearman ($\rho$) antara volatilitas BTC-IDR hari $t$ dengan volatilitas saham AI hari $t+k$:

**Tabel 2A: Korelasi Pearson ($r$) — BTC-IDR vs Saham AI & AI Index**  
*(Signifikan pada $\alpha=0.05$ jika $|r| > 0.0497$)*

| Pasangan Prediktor | Lag $k=-3$ | Lag $k=-2$ | Lag $k=-1$ (Kemarin) | Lag $k=0$ (Kontemporer) | Lag $k=+1$ | Lag $k=+2$ | Lag $k=+3$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC-IDR vs NVDA** | -0.0060 | +0.0139 | -0.0210 | **+0.0686\*** | +0.0129 | +0.0469 | +0.0272 |
| **BTC-IDR vs MSFT** | +0.0025 | +0.0239 | -0.0340 | **+0.1032\*** | -0.0031 | +0.0056 | +0.0057 |
| **BTC-IDR vs TSLA** | +0.0332 | +0.0082 | +0.0021 | **+0.0588\*** | -0.0074 | +0.0179 | -0.0093 |
| **BTC-IDR vs GOOGL** | -0.0180 | +0.0274 | +0.0008 | **+0.0542\*** | -0.0127 | +0.0401 | -0.0244 |
| **BTC-IDR vs META** | -0.0028 | **+0.0740\*** | +0.0070 | **+0.0626\*** | +0.0019 | +0.0366 | +0.0092 |
| **BTC-IDR vs AI_Index**| +0.0028 | +0.0472 | -0.0144 | **+0.1111\*** | -0.0027 | +0.0470 | +0.0027 |

**Tabel 2B: Korelasi Spearman ($\rho$) — BTC-IDR vs Saham AI & AI Index**

| Pasangan Prediktor | Lag $k=-3$ | Lag $k=-2$ | Lag $k=-1$ (Kemarin) | Lag $k=0$ (Kontemporer) | Lag $k=+1$ | Lag $k=+2$ | Lag $k=+3$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC-IDR vs NVDA** | -0.0134 | +0.0235 | +0.0039 | **+0.1036\*** | -0.0104 | +0.0450 | +0.0361 |
| **BTC-IDR vs MSFT** | -0.0262 | +0.0321 | -0.0328 | **+0.0997\*** | +0.0021 | +0.0085 | +0.0187 |
| **BTC-IDR vs TSLA** | +0.0172 | +0.0112 | +0.0042 | **+0.0941\*** | -0.0111 | +0.0132 | -0.0104 |
| **BTC-IDR vs GOOGL** | -0.0468 | +0.0281 | +0.0000 | **+0.0898\*** | -0.0209 | +0.0353 | +0.0052 |
| **BTC-IDR vs META** | -0.0167 | **+0.0716\*** | -0.0063 | **+0.0785\*** | -0.0116 | +0.0524 | +0.0204 |
| **BTC-IDR vs AI_Index**| -0.0256 | +0.0447 | -0.0189 | **+0.1321\*** | -0.0105 | +0.0452 | +0.0102 |

---

### 3.3 Hasil Korelasi Lead-Lag Pembanding: BTC-USD vs Saham AI & BTC-IDR vs BTC-USD

**Tabel 2C: Korelasi Pearson ($r$) — Pembanding BTC-USD vs Saham AI & BTC-IDR vs BTC-USD**

| Pasangan Prediktor | Lag $k=-3$ | Lag $k=-2$ | Lag $k=-1$ (Kemarin) | Lag $k=0$ (Kontemporer) | Lag $k=+1$ | Lag $k=+2$ | Lag $k=+3$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC-USD vs NVDA** | -0.0053 | +0.0212 | -0.0073 | **+0.0883\*** | -0.0056 | +0.0404 | +0.0388 |
| **BTC-USD vs MSFT** | -0.0177 | +0.0284 | -0.0384 | **+0.1104\*** | +0.0055 | -0.0086 | -0.0084 |
| **BTC-USD vs TSLA** | +0.0285 | +0.0043 | -0.0243 | **+0.0779\*** | -0.0080 | -0.0135 | +0.0004 |
| **BTC-USD vs GOOGL** | -0.0234 | +0.0092 | +0.0221 | **+0.0847\*** | -0.0200 | +0.0432 | -0.0198 |
| **BTC-USD vs META** | +0.0075 | **+0.0569\*** | +0.0286 | **+0.0526\*** | -0.0046 | +0.0287 | +0.0154 |
| **BTC-USD vs AI_Index**| -0.0033 | +0.0384 | -0.0062 | **+0.1324\*** | -0.0105 | +0.0288 | +0.0085 |
| **BTC-IDR vs BTC-USD** | +0.0495 | **+0.1237\*** | **+0.0848\*** | **+0.7737\*** | **+0.0746\*** | **+0.1208\*** | +0.0154 |

**Tabel 2D: Korelasi Spearman ($\rho$) — Pembanding BTC-USD vs Saham AI & BTC-IDR vs BTC-USD**

| Pasangan Prediktor | Lag $k=-3$ | Lag $k=-2$ | Lag $k=-1$ (Kemarin) | Lag $k=0$ (Kontemporer) | Lag $k=+1$ | Lag $k=+2$ | Lag $k=+3$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC-USD vs NVDA** | -0.0242 | +0.0076 | -0.0007 | **+0.1008\*** | -0.0229 | +0.0432 | +0.0417 |
| **BTC-USD vs MSFT** | -0.0351 | +0.0271 | -0.0382 | **+0.0947\*** | +0.0014 | -0.0086 | -0.0002 |
| **BTC-USD vs TSLA** | +0.0116 | +0.0092 | -0.0130 | **+0.1107\*** | -0.0022 | -0.0065 | -0.0030 |
| **BTC-USD vs GOOGL** | -0.0580 | +0.0150 | +0.0039 | **+0.0889\*** | -0.0280 | +0.0290 | +0.0031 |
| **BTC-USD vs META** | -0.0192 | **+0.0607\*** | +0.0182 | **+0.0612\*** | -0.0105 | +0.0385 | +0.0263 |
| **BTC-USD vs AI_Index**| -0.0375 | +0.0323 | -0.0170 | **+0.1298\*** | -0.0107 | +0.0305 | +0.0104 |
| **BTC-IDR vs BTC-USD** | **+0.0663\*** | **+0.1510\*** | **+0.0864\*** | **+0.9089\*** | **+0.0750\*** | **+0.1268\*** | +0.0414 |

---

### 3.4 Evaluasi Statistik: Apakah Lag $k=-1$ Lebih Kuat dari Lag $k=0$?

Untuk menguji hipotesis bahwa pergerakan saham kemarin ($k = -1$) lebih kuat daripada pergerakan hari yang sama ($k = 0$) secara formal, dilakukan **Uji Steiger (1980)** untuk perbedaan dua korelasi dependen berpasangan yang berbagi satu variabel dependen ($Y_t$ dengan $X_t$ vs $Y_t$ dengan $X_{t-1}$, $N = 1.555$):

**Tabel 2E: Ringkasan Pengujian Komparasi Lag 0 vs Lag -1 (Steiger Test)**  
*(Sumber Berkas: `audit/leadlag_summary.csv`)*

| Pasangan Aset | $r(k=0)$ | $p$-value ($k=0$) | $r(k=-1)$ | $p$-value ($k=-1$) | Selisih $\Delta r$ | Steiger $z$ | $p$-value Selisih | Kesimpulan Statistik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BTC-IDR vs NVDA** | +0.0686 | 0.0068 | -0.0210 | 0.4072 | +0.0896 | +2.563 | 0.0104 | **Lag 0 lebih kuat** |
| **BTC-IDR vs MSFT** | +0.1032 | 0.0000 | -0.0340 | 0.1804 | +0.1372 | +3.968 | 0.0001 | **Lag 0 lebih kuat** |
| **BTC-IDR vs TSLA** | +0.0588 | 0.0204 | +0.0021 | 0.9341 | +0.0567 | +1.630 | 0.1032 | Tidak dapat dibedakan |
| **BTC-IDR vs GOOGL** | +0.0542 | 0.0326 | +0.0008 | 0.9733 | +0.0533 | +1.533 | 0.1252 | Tidak dapat dibedakan |
| **BTC-IDR vs META** | +0.0626 | 0.0136 | +0.0070 | 0.7813 | +0.0555 | +1.599 | 0.1098 | Tidak dapat dibedakan |
| **BTC-IDR vs AI_Index** | **+0.1111** | **0.0000** | **-0.0144** | **0.5703** | **+0.1255** | **+3.772** | **0.0002** | **Lag 0 lebih kuat** |
| **BTC-USD vs NVDA** | +0.0883 | 0.0005 | -0.0073 | 0.7737 | +0.0956 | +2.736 | 0.0062 | **Lag 0 lebih kuat** |
| **BTC-USD vs MSFT** | +0.1104 | 0.0000 | -0.0384 | 0.1299 | +0.1489 | +4.306 | 0.0000 | **Lag 0 lebih kuat** |
| **BTC-USD vs TSLA** | +0.0779 | 0.0021 | -0.0243 | 0.3389 | +0.1021 | +2.940 | 0.0033 | **Lag 0 lebih kuat** |
| **BTC-USD vs GOOGL** | +0.0847 | 0.0008 | +0.0221 | 0.3834 | +0.0626 | +1.803 | 0.0713 | Tidak dapat dibedakan |
| **BTC-USD vs META** | +0.0526 | 0.0382 | +0.0286 | 0.2593 | +0.0239 | +0.698 | 0.4855 | Tidak dapat dibedakan |
| **BTC-USD vs AI_Index** | **+0.1324** | **0.0000** | **-0.0062** | **0.8085** | **+0.1386** | **+4.171** | **0.0000** | **Lag 0 lebih kuat** |
| **BTC-IDR vs BTC-USD** | **+0.7737** | **0.0000** | **+0.0848** | **0.0008** | **+0.6889** | **+25.654** | **0.0000** | **Lag 0 jauh lebih kuat** |

#### Temuan Analisis:
1. **Tidak Ada Pasangan di Mana Lag $k=-1$ Lebih Kuat dari Lag $k=0$:**  
   Pada seluruh pasangan saham individual maupun indeks gabungan, korelasi pada lag $k=-1$ (*saham kemarin*) tidak pernah lebih besar daripada lag $k=0$ (*kontemporer*).
2. **Korelasi Lag $k=-1$ Berada di Sekitar Nol:**  
   Untuk indeks gabungan AI (`ai_index`), korelasi lag $k=-1$ adalah $-0.0144$ ($p = 0.5703$) pada BTC-IDR dan $-0.0062$ ($p = 0.8085$) pada BTC-USD. Keduanya berada jauh di bawah ambang signifikansi $1.96/\sqrt{N} \approx 0.0497$ dan tidak dapat dibedakan dari nol secara statistik.
3. **Pola BTC-IDR Identik dengan BTC-USD:**  
   Struktur kurva lead-lag BTC-IDR dan BTC-USD terhadap saham AI memiliki bentuk yang hampir sama persis: puncak tajam pada lag $k=0$, penurunan drastis ke nol pada lag $k=-1$ dan $k=+1$, serta pantulan kecil simetris pada lag $k = \pm 2$ (sekitar $0.03$ s.d. $0.05$) yang merupakan konsekuensi alami dari persistensi autoregresif mingguan (*volatility clustering*).
4. **Hubungan BTC-IDR vs BTC-USD Mendominasi Total pada Lag $k=0$:**  
   Korelasi kontemporer antara BTC-IDR dan BTC-USD mencapai **$r = +0.7737$** (Pearson) dan **$\rho = +0.9089$** (Spearman), dengan $p < 10^{-100}$. Sebaliknya, pada lag $k=-1$, korelasinya anjlok menjadi $0.0848$. (Sebagai informasi tambahan, korelasi log-return harga antara BTC-IDR dan BTC-USD pada lag $k=0$ mencapai **$+0.9811$**, sedangkan pada lag $k=-1$ bernilai $-0.0046$).

---

### 3.5 Dampak Perhitungan Lag dalam Hari Bursa (*Trading Days*) vs Hari Kalender

Pengujian lead-lag di atas dihitung pada urutan baris `data/clean/dataset_garch_baseline.csv` yang merupakan hasil *inner join* (hari bursa). Implikasi strukturalnya meliputi:
1. **Jarak Waktu Fisik Tidak Homogen:**  
   - Jeda antara baris Selasa s.d. Jumat adalah 1 hari kalender (24 jam).
   - Jeda antara baris Jumat ke Senin adalah 3 hari kalender (72 jam), atau 4 hari kalender jika ada hari libur bursa.
2. **Dampak terhadap Return Hari Senin:**  
   Karena baris akhir pekan dibuang dan return dihitung via `shift(1)`, return hari Senin mencakup akumulasi pergerakan 72 jam penuh (Jumat sore s.d. Senin sore). Dengan demikian, syok volatilitas akhir pekan pada cryptocurrency sudah tercermin langsung pada baris hari Senin.
3. **Konfirmasi Konsistensi:**  
   Karena saham AS dan bursa crypto sama-sama memiliki baris aktif pada hari Senin yang mencakup jeda akhir pekan tersebut, perbandingan lag $k=0$ tetap membandingkan rentang waktu bursa yang setara secara simultan.

---

## 4. LANGKAH 3: Ringkasan Keputusan dan Rekomendasi

### 4.1 Apakah Terdapat Bukti Ketidakselarasan Waktu (*Misalignment*) yang Material?

> **KEPUTUSAN AUDIT: TIDAK.**  
> **Tidak ditemukan bukti empiris adanya ketidakselarasan tanggal (*day-level timestamp misalignment* / dislokasi 1 hari) yang material antara BTC/IDR Indodax, BTC-USD, dan saham AI AS pada pipeline RSIC.**

**Alasan Ilmiah Berbasis Data:**
1. **Puncak Kontemporer Sempurna pada Lag $k=0$:**  
   Jika data Indodax BTC/IDR mengalami keterlambatan pencatatan 1 hari kalender (misalnya akibat perbedaan penutupan UTC vs WIB), maka pergerakan saham AS kemarin ($k=-1$) seharusnya memiliki korelasi yang jauh lebih tinggi terhadap BTC-IDR hari ini dibanding pergerakan saham AS hari yang sama ($k=0$). Fakta empiris membuktikan sebaliknya: korelasi lag $k=0$ adalah puncak tertinggi ($p < 0.0001$), sementara korelasi lag $k=-1$ tidak berbeda secara statistik dari nol ($r = -0.0144, p = 0.5703$).
2. **Korelasi Ekstrem BTC-IDR vs BTC-USD pada Hari yang Sama:**  
   Korelasi return harga kontemporer ($r = 0.9811$) dan volatilitas ($r = 0.7737, \rho = 0.9089$) membuktikan bahwa bursa Indodax dan bursa global bergerak secara instan dan simultan pada tanggal kalender yang sama. Jika terdapat dislokasi 1 hari, korelasi kontemporer tidak mungkin mencapai 0.98.
3. **Penjelasan Temuan Net Spillover (+5.45) di RQ1:**  
   Hipotesis awal bahwa status *net transmitter* BTC-IDR (+5.45) pada estimasi TVP-VAR merupakan artefak kesalahan pergeseran tanggal 1 hari **tidak didukung oleh data**. Sebagaimana dibuktikan pada audit 14 Oktober 2026, nilai net spillover BTC-IDR bergeser dan berbalik tanda menjadi *net receiver* (-4.98 pada $r^2$ dan -4.10 pada Parkinson) akibat bentuk transformasi matematis proksi volatilitas dan keterhubungan mutual arbitrase harga dengan BTC-USD, bukan karena tanggal yang bergeser.

---

### 4.2 Analisis Skenario Koreksi dan Dampak Pekerjaan Ulang (*Rerun Impact*)

Meskipun audit memutuskan bahwa pergeseran tanggal **TIDAK diperlukan dan TIDAK dibenarkan secara empiris**, berikut adalah analisis transparansi jika di masa depan ada skenario pengujian sensitivitas pergeseran tanggal ($t \pm 1$):

1. **Koreksi Minimal yang Dibutuhkan (Hipotetis):**  
   Menggeser kolom return BTC-IDR sebanyak 1 langkah waktu menggunakan `shift(-1)` atau `shift(+1)` pada `src/preprocess_data.py`.
2. **Berkas yang Terdampak Luas (*Downstream Files*):**  
   - `data/clean/dataset_garch_baseline.csv` (seluruh baris target berubah).
   - `data/clean/tvp_var_dynamic_features.csv` dan `tvp_var_dynamic_features_causal.csv` (seluruh fitur turunan TVP-VAR berubah).
   - `data/clean/dataset_deep_learning.csv` (variabel target dan rolling split berubah).
   - Seluruh estimasi TVP-VAR pada `tvp_var_spillover.ipynb` (seluruh tabel spillover, matriks 8x8, dan grafik TSI).
   - Seluruh model baseline ekonometrika di `results/`: HAR v2, GARCH(1,1), EGARCH.
   - Seluruh model Deep Learning di `results/v5/` dan `results/v5_sens/`: TFT dan N-BEATSx pada `NOSPILL`, `SPILL`, `VOLX`.
   - Seluruh tabel metrik evaluasi dan uji inferensi Diebold-Mariano (`v5_metrics_table.csv`, `v5_dm_table.csv`, `v5s_metrics_table.csv`, `v5s_dm_table.csv`).
3. **Perkiraan Beban Pekerjaan Ulang:**  
   Beban pekerjaan ulang berskala **MASIF** (membutuhkan waktu komputasi berjam-jam/berhari-hari untuk melatih ulang seluruh model neural pada 335 cutoff evaluasi).
4. **Rekomendasi Tegas:**  
   **JANGAN melakukan pergeseran tanggal.** Melakukan pergeseran tanggal secara buatan justru akan merusak korelasi kontemporer fundamental ($r = 0.98$ pada return dan $r = 0.77$ pada volatilitas), menciptakan *mismatch* artifisial baru yang berlawanan dengan realitas pasar keuangan.

---

### 4.3 Ketersediaan Resolusi Intraday pada Data Mentah

1. **Status Berkas Lokal:**  
   Seluruh 11 berkas CSV di dalam folder `data/raw/` hanya memiliki resolusi harian (`1D`, tanggal berformat `YYYY-MM-DD`). Tidak ada berkas intraday (menit/jam) yang tersimpan di repositori saat ini.
2. **Potensi Eksternal API Indodax & yfinance:**  
   - Endpoint TradingView Indodax (`https://indodax.com/tradingview/history_v2`) secara teknis memiliki parameter `tf` yang mendukung resolusi sub-harian (misal `tf=60` untuk 1 jam atau `tf=15` untuk 15 menit). Selain itu, Public API Indodax memiliki endpoint `https://indodax.com/api/trades/btcidr` dengan stempel waktu detik per transaksi.
   - Library `yfinance` juga mendukung parameter `interval='1h'` atau `interval='5m'` untuk jangka waktu terbatas.
   - *Catatan Kepatuhan:* Sesuai instruksi audit *read-only*, data intraday dari API eksternal **tidak diunduh**. Analisis dibatasi secara sah pada artefak yang ada di dalam repositori.

---

## 5. Daftar Item UNVERIFIED (Keterbatasan Pembuktian Berkas)

Berikut adalah aspek-aspek yang tidak dapat dibuktikan secara mutlak dari berkas kode dan data yang ada:
1. **Jam Pemotongan Lilin Harian Indodax (UTC vs WIB):**  
   *Status: UNVERIFIED.* Berkas mentah `crypto_btc_idr_indodax.csv` hanya memuat tanggal string `YYYY-MM-DD` tanpa komponen jam/menit. Kode `src/collect_data.py` (baris 289 & 293) mengonversi stempel waktu epoch detik dengan `utc=True`. Apakah server backend Indodax memotong lilin 1D pada pukul 00:00:00 UTC (07:00:00 WIB) atau pukul 00:00:00 WIB (17:00:00 UTC) tidak terdokumentasi dalam berkas repositori lokal.
2. **Jam Penutupan Lilin Harian Crypto Global (yfinance):**  
   *Status: UNVERIFIED.* Walaupun konvensi umum Yahoo Finance crypto adalah 00:00:00 s.d. 23:59:59 UTC, berkas `crypto_btc_usd.csv` dan `crypto_eth_usd.csv` tidak menyimpan stempel jam intra-hari.
3. **Jam Penutupan Perdagangan Bursa Saham AS:**  
   *Status: UNVERIFIED dalam berkas.* Berkas `saham_*.csv` tidak mencatat jam penutupan bursa. Jam tutup 16:00 ET (20:00 UTC pada EDT / 21:00 UTC pada EST) merupakan fakta institusional bursa saham Amerika Serikat yang berlaku umum.
4. **Stempel Waktu Transaksi Tingkat Detik Intra-hari Indodax:**  
   *Status: UNVERIFIED.* Tidak ada data buku pesanan (*order book*) atau log transaksi (*tick data*) intra-hari di dalam repositori untuk mengukur selisih waktu reaksi pasar domestik dalam hitungan menit/detik terhadap pengumuman laba saham AI di AS.
