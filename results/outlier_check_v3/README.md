# Outlier Sensitivity Check v3: Verifikasi Hari Ekstrem & QLIKE DM

Dokumen ini menyajikan verifikasi data asli untuk tanggal ekstrem **2026-02-05** dan **2026-02-06**, perbandingannya dengan 5 hari ekstrem berikutnya (Rank 3–7), audit candle data mentah Indodax, serta uji signifikansi prediktif Diebold-Mariano berbasis rugi **QLIKE Patton (2011)** tanpa clipping.

Seluruh perhitungan dijalankan secara deterministik pada CPU dan berstatus **EXPLORATORY**.

---

## 1. Verifikasi Hari Ekstrem (2026-02-02 s/d 2026-02-10)

### 1.1 Return dan Harga Penutupan Mentah

Tabel berikut memuat harga penutupan mentah dan return log harian untuk BTC-IDR (Indodax), BTC-USD (Yahoo Finance), USD/IDR, selisih return, serta ETH-USD.

| Date       | Day | Close Indodax | Close BTC-USD | Close USD/IDR | Ret BTC-IDR (%) | Ret BTC-USD (%) | Diff (%) | Ret USD/IDR (%) | Implied Diff (%) | Ret ETH-USD (%) |
| ---------- | --- | ------------- | ------------- | ------------- | --------------- | --------------- | -------- | --------------- | ---------------- | --------------- |
| 2026-02-02 | Mon | 1,327,739,000 | $78,688.77    | 16,758.40     | +1.82%          | +2.20%          | -0.38%   | -0.12%          | -0.26%           | +3.31%          |
| 2026-02-03 | Tue | 1,273,500,000 | $75,633.55    | 16,758.40     | -4.17%          | -3.96%          | -0.21%   | +0.00%          | -0.21%           | -5.11%          |
| 2026-02-04 | Wed | 1,233,766,000 | $73,019.70    | 16,759.00     | -3.17%          | -3.52%          | +0.35%   | +0.00%          | +0.34%           | -3.85%          |
| 2026-02-05 | Thu | 1,063,498,000 | $62,702.10    | 16,800.00     | -14.85%         | -15.23%         | +0.38%   | +0.24%          | +0.14%           | -16.27%         |
| 2026-02-06 | Fri | 1,196,068,000 | $70,555.39    | 16,879.90     | +11.75%         | +11.80%         | -0.05%   | +0.47%          | -0.53%           | +12.46%         |
| 2026-02-09 | Mon | 1,181,025,000 | $70,120.78    | 16,843.50     | -0.82%          | -0.21%          | -0.61%   | -0.22%          | -0.40%           | +0.71%          |
| 2026-02-10 | Tue | 1,155,956,000 | $68,793.96    | 16,796.00     | -2.15%          | -1.91%          | -0.24%   | -0.28%          | +0.05%           | -4.08%          |

### 1.2 Return Saham AI Global pada Jendela yang Sama

Pergerakan return log harian untuk lima saham AI acuan (NVDA, MSFT, TSLA, GOOGL, META):

| Date       | Ret NVDA (%) | Ret MSFT (%) | Ret TSLA (%) | Ret GOOGL (%) | Ret META (%) |
| ---------- | ------------ | ------------ | ------------ | ------------- | ------------ |
| 2026-02-02 | -2.93%       | -1.62%       | -2.02%       | +1.67%        | -1.42%       |
| 2026-02-03 | -2.88%       | -2.91%       | +0.04%       | -1.16%        | -2.10%       |
| 2026-02-04 | -3.47%       | +0.72%       | -3.85%       | -1.98%        | -3.34%       |
| 2026-02-05 | -1.34%       | -5.08%       | -2.19%       | -0.54%        | +0.18%       |
| 2026-02-06 | +7.58%       | +1.88%       | +3.44%       | -2.57%        | -1.31%       |
| 2026-02-09 | +2.47%       | +3.06%       | +1.50%       | +0.45%        | +2.35%       |
| 2026-02-10 | -0.79%       | -0.08%       | +1.87%       | -1.79%        | -0.96%       |

---

## 2. Analisis Selisih Return BTC-IDR vs BTC-USD

Selisih return harian didefinisikan sebagai:

- Delta r_t = r_{BTC-IDR} - r_{BTC-USD}

- Delta r_t (implied FX) = r_{BTC-IDR} - (r_{BTC-USD} + r_{USD/IDR})


| Metric / Percentile           | Full Sample (N=1,677) | 2026-02-05 (Rank 1) | 2026-02-06 (Rank 2) |
| ----------------------------- | --------------------- | ------------------- | ------------------- |
| Mean Difference               | +0.0102%              | -                   | -                   |
| Std Difference                | 0.8085%               | -                   | -                   |
| Percentile 1%                 | -2.2302%              | -                   | -                   |
| Percentile 5%                 | -1.2110%              | -                   | -                   |
| Percentile 95%                | +1.2192%              | -                   | -                   |
| Percentile 99%                | +2.2373%              | -                   | -                   |
| Return Difference (IDR - USD) | -                     | +0.3827%            | -0.0527%            |
| Z-score vs Full Sample        | -                     | +0.4606 std         | -0.0778 std         |
| Inside P1-P99 Interval?       | -                     | YES                 | YES                 |

### Temuan Faktual:

1. **Arah dan Besaran**: Penurunan -14.85% (Feb 5) dan rebound +11.75% (Feb 6) di Indodax tercermin secara langsung dan serentak di pasar global BTC-USD (-15.23% pada Feb 5 dan +11.80% pada Feb 6).

2. **Besaran Tracking Error**: Selisih return BTC-IDR terhadap BTC-USD pada 5 Februari adalah **+0.38%** (z-score: +0.46) dan pada 6 Februari adalah **-0.05%** (z-score: -0.08). Keduanya berada jauh di dalam batas rentang persentil 1% s/d 99% dari seluruh sampel data (N = 1.677, [-2.23%, +2.24%]).

3. **Peran Kurs USD/IDR**: Depresiasi Rupiah (+0.24% pada Feb 5 dan +0.47% pada Feb 6) mempersempit tracking error implied menjadi hanya **+0.14%** pada 5 Februari dan **-0.53%** pada 6 Februari.

---

## 3. Audit Data Mentah Indodax (Candle Integrity)

| Date       | Candles | Open (IDR)    | High (IDR)    | Low (IDR)     | Close (IDR)   | Volume (BTC) | H/L Ratio | Open vs Prev Close |
| ---------- | ------- | ------------- | ------------- | ------------- | ------------- | ------------ | --------- | ------------------ |
| 2026-02-02 | 1       | 1,303,729,000 | 1,339,152,000 | 1,260,517,000 | 1,327,739,000 | 67.17        | 1.0624    | N/A                |
| 2026-02-03 | 1       | 1,327,739,000 | 1,331,937,000 | 1,230,000,000 | 1,273,500,000 | 49.43        | 1.0829    | +0.0000%           |
| 2026-02-04 | 1       | 1,273,500,000 | 1,292,678,000 | 1,213,818,000 | 1,233,766,000 | 45.78        | 1.0650    | +0.0000%           |
| 2026-02-05 | 1       | 1,233,765,000 | 1,233,766,000 | 1,045,000,000 | 1,063,498,000 | 176.51       | 1.1806    | -0.0001%           |
| 2026-02-06 | 1       | 1,063,500,000 | 1,219,000,000 | 1,020,000,000 | 1,196,068,000 | 162.21       | 1.1951    | +0.0002%           |
| 2026-02-07 | 1       | 1,196,068,000 | 1,212,000,000 | 1,138,106,000 | 1,172,808,000 | 46.98        | 1.0649    | +0.0000%           |
| 2026-02-08 | 1       | 1,172,808,000 | 1,219,000,000 | 1,166,925,000 | 1,190,709,000 | 26.05        | 1.0446    | +0.0000%           |
| 2026-02-09 | 1       | 1,190,710,000 | 1,207,936,000 | 1,155,000,000 | 1,181,025,000 | 29.99        | 1.0458    | +0.0001%           |
| 2026-02-10 | 1       | 1,181,025,000 | 1,186,595,000 | 1,145,000,000 | 1,155,956,000 | 26.52        | 1.0363    | +0.0000%           |

### Parameter Audit:

- **Jumlah Candle**: Tepat 1 candle per hari kalender tanpa ada candle duplikat dan tanpa ada hari yang hilang (kelanjutan tanggal sempurna 1 hari kalender penuh).

- **Likuiditas dan Volume**: Volume perdagangan di Indodax melonjak tajam pada 5 Februari (**176.51 BTC**) dan 6 Februari (**162.21 BTC**), jauh di atas rata-rata volume harian normal (20–60 BTC). Volume global BTC-USD juga melonjak ke rekor $125.5 Miliar (Feb 5) dan $114.7 Miliar (Feb 6). Lonjakan volume ini menunjukkan adanya aktivitas likuidasi dan akumulasi nyata, bukan akibat ilikuiditas tipis (*thin liquidity*).

- **Kontinuitas Harga**: Harga Open tanggal 5 Februari (Rp 1.233.765.000) dan 6 Februari (Rp 1.063.500.000) bersambung secara kontinu dengan harga Close hari sebelumnya (gap terbuka kurang dari 0.0002%). Tidak terdapat lonjakan harga anomali atau *spike glitch* antarcandle.

- **Jam Cut Candle**: **UNVERIFIED**. Skrip pengumpulan data (`src/collect_data.py`) mem-parsing timestamp detik Unix (`unit="s", utc=True`) dari endpoint TradingView Indodax. Namun, apakah server Indodax memotong lilin harian pada pukul 00:00 UTC (07:00 WIB) atau pukul 00:00 WIB (17:00 UTC hari sebelumnya) tidak dapat diverifikasi secara independen dari kode maupun CSV mentah.

---

## 4. Status Gap-Day dan Akumulasi Return

Berdasarkan definisi fitur penataan kalender pada `dataset_deep_learning.csv`:

- `gap_days = 1` untuk tanggal 2026-02-05 (Kamis) dan 2026-02-06 (Jumat).

- `is_monday = 0` untuk kedua tanggal tersebut.

- `crypto_btc_idr_indodax_weekend_return = 0.0` dan `crypto_btc_usd_weekend_return = 0.0`.


Kedua tanggal tersebut **bukan** merupakan *gap-day* dan **bukan** akumulasi akhir pekan (Sun→Mon). Return yang tercatat adalah return perdagangan tunggal selama 24 jam kalender.

---

## 5. Perbandingan dengan 5 Hari Ekstrem Berikutnya (Rank 3–7)

Tabel perbandingan data aktual untuk Rank 1 hingga Rank 7 dari `extreme_days_v2.csv`:

| Rank   | Date       | gap_days | Indodax Vol (BTC) | Close Indodax | Close BTC-USD | Ret BTC-IDR (%) | Ret BTC-USD (%) | Diff (%) | Ret USD/IDR (%) | Implied Diff (%) | Ret ETH-USD (%) |
| ------ | ---------- | -------- | ----------------- | ------------- | ------------- | --------------- | --------------- | -------- | --------------- | ---------------- | --------------- |
| Rank 1 | 2026-02-05 | 1        | 176.51            | 1,063,498,000 | $62,702.10    | -14.85%         | -15.23%         | +0.38%   | +0.24%          | +0.14%           | -16.27%         |
| Rank 2 | 2026-02-06 | 1        | 162.21            | 1,196,068,000 | $70,555.39    | +11.75%         | +11.80%         | -0.05%   | +0.47%          | -0.53%           | +12.46%         |
| Rank 3 | 2026-08-21 | 1        | 112.67            | 1,374,295,000 | $78,335.19    | +6.53%          | +7.01%          | -0.48%   | -0.29%          | -0.19%           | +7.81%          |
| Rank 4 | 2026-06-02 | 1        | 79.88             | 1,196,483,000 | $66,703.66    | -6.11%          | -6.69%          | +0.58%   | +0.26%          | +0.32%           | -7.54%          |
| Rank 5 | 2026-03-04 | 1        | 50.89             | 1,225,005,000 | $72,710.58    | +5.71%          | +6.27%          | -0.55%   | +0.21%          | -0.76%           | +7.00%          |
| Rank 6 | 2026-08-19 | 1        | 54.16             | 1,218,796,000 | $69,266.19    | +5.64%          | +6.85%          | -1.21%   | -0.31%          | -0.90%           | +16.11%         |
| Rank 7 | 2025-11-14 | 1        | 122.52            | 1,588,000,000 | $94,397.79    | -5.63%          | -5.46%          | -0.17%   | -0.06%          | -0.11%           | -4.07%          |

### Pola Konsistensi:

- Seluruh 7 hari ekstrem memiliki `gap_days = 1` (hari perdagangan tunggal biasa).

- Pada seluruh 7 hari ekstrem, arah pergerakan BTC-IDR selalu identik dengan arah BTC-USD dan ETH-USD.

- Tracking error antara BTC-IDR dan BTC-USD pada seluruh 7 hari ekstrem sangat kecil (berkisar antara -1.21% hingga +0.58%), menunjukkan hubungan arbitrase pasar lokal terhadap pasar global yang tetap terjaga selama periode volatilitas ekstrem.

---

## 6. Kesimpulan Faktual Hari Ekstrem

1. Pergerakan ekstrem pada tanggal 5 dan 6 Februari 2026 di Indodax (-14.85% dan +11.75%) **konsisten secara langsung dengan pergerakan pasar global** (BTC-USD -15.23% dan +11.80%, ETH-USD -16.27% dan +12.46%).

2. Fenomena ini **bukan anomali lokal spesifik Indodax**, bukan akibat kesalahan pencatatan (*data glitch*), bukan akibat ilikuiditas tipis, dan bukan akumulasi akhir pekan (*non-gap day*).

3. Faktor penggerak fundamental makroekonomi atau pemicu pasar eksternal yang melatarbelakangi jatuhnya harga global tersebut **tidak dapat diverifikasi dari data** numerik yang tersedia di repositori ini.

---

## 7. Uji Diebold-Mariano QLIKE (Patton 2011)

### 7.1 Kelayakan Model (Domain Prediksi)

Fungsi rugi QLIKE Patton (2011) didefinisikan sebagai:

  QLIKE(y, y_hat) = y / y_hat - ln(y / y_hat) - 1


Fungsi ini mensyaratkan y_hat > 0. Model dengan prediksi non-positif (<= 0) tidak dapat dievaluasi menggunakan QLIKE tanpa clipping buatan.

| Model           | Min Pred (h=1) | Min Pred (h=10) | All Pred > 0? | QLIKE Status    |
| --------------- | -------------- | --------------- | ------------- | --------------- |
| HAR             | +0.00049345    | +0.00056331     | YES           | Included (> 0)  |
| Hist_Exp        | +0.00100219    | +0.00100219     | YES           | Included (> 0)  |
| Roll_20         | +0.00006637    | +0.00006637     | YES           | Included (> 0)  |
| GARCH11         | +0.00030390    | +0.00047180     | YES           | Included (> 0)  |
| EGARCH          | +0.00022588    | +0.00045439     | YES           | Included (> 0)  |
| TFT_SPILL       | +0.00011795    | +0.00014220     | YES           | Included (> 0)  |
| TFT_NOSPILL     | +0.00011884    | +0.00013452     | YES           | Included (> 0)  |
| NBEATSx_SPILL   | -0.00287326    | -0.00119319     | NO            | EXCLUDED (<= 0) |
| NBEATSx_NOSPILL | -0.00139993    | -0.00113684     | NO            | EXCLUDED (<= 0) |

> **Catatan**: N-BEATSx (baik SPILL maupun NOSPILL) menghasilkan prediksi negatif (minimum -0.00287 pada h=1 dan -0.00119 pada h=10). Sesuai metodologi yang ketat tanpa clipping buatan, **N-BEATSx dikecualikan secara eksplisit** dari evaluasi QLIKE.

### 7.2 Hasil Uji Diebold-Mariano (Full Sample, N=335)

- Horizon h=1: Lag Newey-West = 0.

- Horizon h=10: Lag Newey-West = 9 dengan bobot Bartlett dan koreksi sampel kecil Harvey-Leybourne-Newbold (HLN).

- Koreksi Family-wise: m = 20 (seluruh 20 uji pada tabel) dan m = 10 (per horizon).

- Status: **EXPLORATORY**.

| Horizon | Model 1 (DL) | Model 2 (Benchmark) | QLIKE M1 | QLIKE M2 | Diff (M1 - M2) | DM Stat | p_raw  | p_bonf (m=20) | Verdict (alpha=0.05) |
| ------- | ------------ | ------------------- | -------- | -------- | -------------- | ------- | ------ | ------------- | -------------------- |
| h=1     | TFT_SPILL    | HAR                 | 1.7738   | 1.7020   | +0.0718        | +0.7646 | 0.4445 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_SPILL    | Hist_Exp            | 1.7738   | 1.9695   | -0.1957        | -1.6264 | 0.1039 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_SPILL    | Roll_20             | 1.7738   | 1.7475   | +0.0263        | +0.4265 | 0.6697 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_SPILL    | GARCH11             | 1.7738   | 1.6537   | +0.1201        | +1.3713 | 0.1703 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_SPILL    | EGARCH              | 1.7738   | 1.6406   | +0.1332        | +1.3559 | 0.1751 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_NOSPILL  | HAR                 | 1.7705   | 1.7020   | +0.0685        | +0.7454 | 0.4560 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_NOSPILL  | Hist_Exp            | 1.7705   | 1.9695   | -0.1990        | -1.6967 | 0.0897 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_NOSPILL  | Roll_20             | 1.7705   | 1.7475   | +0.0230        | +0.3682 | 0.7127 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_NOSPILL  | GARCH11             | 1.7705   | 1.6537   | +0.1168        | +1.3667 | 0.1717 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=1     | TFT_NOSPILL  | EGARCH              | 1.7705   | 1.6406   | +0.1299        | +1.3308 | 0.1833 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_SPILL    | HAR                 | 1.8032   | 1.7992   | +0.0040        | +0.0352 | 0.9719 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_SPILL    | Hist_Exp            | 1.8032   | 1.9435   | -0.1403        | -0.8096 | 0.4182 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_SPILL    | Roll_20             | 1.8032   | 2.0094   | -0.2063        | -1.4252 | 0.1541 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_SPILL    | GARCH11             | 1.8032   | 1.7960   | +0.0072        | +0.0679 | 0.9459 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_SPILL    | EGARCH              | 1.8032   | 1.8275   | -0.0244        | -0.1848 | 0.8534 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_NOSPILL  | HAR                 | 1.7844   | 1.7992   | -0.0148        | -0.1882 | 0.8507 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_NOSPILL  | Hist_Exp            | 1.7844   | 1.9435   | -0.1591        | -1.2047 | 0.2283 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_NOSPILL  | Roll_20             | 1.7844   | 2.0094   | -0.2250        | -1.1585 | 0.2467 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_NOSPILL  | GARCH11             | 1.7844   | 1.7960   | -0.0115        | -0.1570 | 0.8752 | 1.0000        | Tie (p_bonf >= 0.05) |
| h=10    | TFT_NOSPILL  | EGARCH              | 1.7844   | 1.8275   | -0.0431        | -0.4611 | 0.6447 | 1.0000        | Tie (p_bonf >= 0.05) |

### Rangkuman QLIKE:

1. Pada seluruh 20 pasangan uji (TFT_SPILL dan TFT_NOSPILL vs 5 model pembanding pada h=1 dan h=10), seluruh nilai p_bonf bernilai **1.0000** (atau p_raw > 0.05).

2. Tidak terdapat perbedaan performa prediktif yang signifikan secara statistik antara TFT dan model ekonometrika konvensional (HAR, Hist_Exp, Roll_20, GARCH(1,1), EGARCH) di bawah metrik kerugian asimetris QLIKE.

3. Seluruh perbandingan berstatus **Tie** pada taraf signifikansi alpha = 0.05.

---

## 8. Verifikasi Integritas File (MD5 Checksum)

### Input Files (Read-Only)

| File | Rel Path | MD5 Hash |

| :--- | :--- | :--- |
| `dataset_deep_learning.csv` | `data/clean/dataset_deep_learning.csv` | `ba64297407f9d8221488b4815a12ce8c` |
| `crypto_btc_idr_indodax.csv` | `data/raw/crypto_btc_idr_indodax.csv` | `45bb5f212fa7e401d73fc5e6909c69f4` |
| `crypto_btc_usd.csv` | `data/raw/crypto_btc_usd.csv` | `989c51f781982ca3715efba61222278b` |
| `crypto_eth_usd.csv` | `data/raw/crypto_eth_usd.csv` | `ed773c4eaa057657f4948b800a8432f9` |
| `konteks_usdidr.csv` | `data/raw/konteks_usdidr.csv` | `85ac3879bf5ce3e6a95995a725871fc2` |
| `saham_nvda.csv` | `data/raw/saham_nvda.csv` | `b57c0bf69409a3c6d14bb1fdac44e054` |
| `saham_msft.csv` | `data/raw/saham_msft.csv` | `187857434608fb442e97fb12af5204b9` |
| `saham_tsla.csv` | `data/raw/saham_tsla.csv` | `71b3149375fc0a7469d7889bdc33a3f8` |
| `saham_googl.csv` | `data/raw/saham_googl.csv` | `10f2e090065e87a8658da84fe2d63b91` |
| `saham_meta.csv` | `data/raw/saham_meta.csv` | `5c8ae92d957af9902a4de0e53b2fc315` |
| `CV_Predictions_V4.csv` | `inputs/v5/CV_Predictions_V4.csv` | `6251bfb6ce543ffcd746e5fab98c3a63` |
| `har_preds_T0_h1_v2.csv` | `inputs/v5/har_preds_T0_h1_v2.csv` | `e340a7ba8effd60c8c6d7b85b073aded` |
| `har_preds_T0_h10_v2.csv` | `inputs/v5/har_preds_T0_h10_v2.csv` | `43baf149f5f60be52a80b962ab93e481` |
| `GARCH_Predictions_V4.csv` | `inputs/v5/GARCH_Predictions_V4.csv` | `5e84fa827fd5cfd13bea445230daf2ee` |
| `v6_ensemble_predictions.csv` | `results/v6/v6_ensemble_predictions.csv` | `9e1bdd958db2414afed2c045773d0d78` |
| `extreme_days_v2.csv` | `results/outlier_check_v2/extreme_days_v2.csv` | `67e6748b0f8918603ba467b0c24ae809` |

### Output Files
| File | Rel Path | MD5 Hash |
| :--- | :--- | :--- |
| `event_check.csv` | `results/outlier_check_v3/event_check.csv` | `8d57b180b10be0786a2ec2fb6f7ef7ef` |
| `qlike_dm.csv` | `results/outlier_check_v3/qlike_dm.csv` | `835adc058b595659e8ef01f59fc5ecc1` |
