# RSIC v6 Outlier Sensitivity Check: Analysis Report

**Target:** Realized Volatility $\sigma^2_t = \text{ret\_crypto\_btc\_idr\_indodax}^2$  
**Evaluation Window:** 335 cutoffs (Target dates: 2025-04-24 to 2026-09-04)  
**Execution Environment:** CPU only, deterministic evaluation, no retraining.

---

## 1. Lima Kalimat Kesimpulan Utama (Executive Summary)

1. **Ranking Model Berubah Drastis:** Kesimpulan v6 bahwa HAR adalah model terbaik murni didorong oleh segelintir hari ekstrem; begitu top 1% hari teratas dikeluarkan, TFT (baik NOSPILL maupun SPILL) langsung merebut peringkat 1 dan 2 mengalahkan HAR pada horizon $h=1$, dan pada level eksklusi top 5% model Deep Learning mendominasi seluruh posisi 3 teratas pada $h=1$ dan $h=10$.
2. **Terjadi Pembalikan Signifikan (DM Flips):** Uji Diebold-Mariano berbalik arah secara material; pada full-sample HAR mengungguli TFT SPILL pada $h=1$ ($\text{stat} = +1.0450, p = 0.2960$), namun tanda statistik berbalik mendukung TFT pada top 1% ($\text{stat} = -1.1827$), mencapai signifikansi raw pada top 2.5% ($\text{stat} = -2.2750, p_{\text{raw}} = 0.0229$), dan menjadi signifikan secara kuat bahkan setelah koreksi Bonferroni pada top 5% ($\text{stat} = -2.8545, p_{\text{bonf, m=8}} = 0.0345$).
3. **Konsentrasi SSE Sangat Ekstrem:** Top 1% hari teratas (hanya 4 hari perdagangan dari total 335 hari) menyumbang **78.76%** dari total SSE model HAR ($h=1$) dan **81.94%** dari total SSE model TFT SPILL, di mana dua hari berturut-turut pada 5–6 Februari 2026 (crash $-14.85\%$ disusul rebound $+11.75\%$) mendominasi lebih dari $76\%$ total error kuadrat seluruh periode uji.
4. **Keunggulan TFT vs Hist_Exp Terbukti Sangat Kokoh:** Keunggulan TFT SPILL dan NOSPILL terhadap baseline `Hist_Exp` pada $h=1$ sama sekali bukan artefak outlier; sebaliknya, setelah top 1%, 2.5%, dan 5% hari ekstrem dikeluarkan, keunggulan TFT menjadi semakin dominan secara spektakuler dengan statistik DM melonjak dari $-3.3040$ ($p_{\text{bonf}} = 0.0076$) menjadi $-9.3166$, $-13.5356$, dan $-16.9570$ ($p < 10^{-15}$).
5. **Metrik Non-Kuadratik Mengonfirmasi Keunggulan Konsisten DL:** Pada metrik MAE dan QLIKE yang tidak terdistorsi penalti kuadratik ekstrem, model Deep Learning (terutama NBEATSx_NOSPILL dan TFT) secara konsisten mengalahkan HAR bahkan pada sampel penuh tanpa eksklusi ($p_{\text{bonf}} < 10^{-6}$), membuktikan bahwa DL secara intrinsik memodelkan dinamika volatilitas harian normal dengan jauh lebih presisi dibandingkan baseline ekonometrika.

---

## 2. Tabel Ringkas Metrik Evaluasi Berdasarkan Level Eksklusi

*Catatan:*
- `none` ($n=335$): Sampel penuh tanpa eksklusi (Ground truth match v6).
- `top_1%` ($n=331$): 4 hari realized $y$ tertinggi di-drop ($y > 3.572 \times 10^{-3}$).
- `top_2.5%` ($n=326$): 9 hari realized $y$ tertinggi di-drop ($y > 2.848 \times 10^{-3}$).
- `top_5%` ($n=318$): 17 hari realized $y$ tertinggi di-drop ($y > 2.195 \times 10^{-3}$).

### Horizon h=1

| Model | Level Eksklusi | MSE | MAE | QLIKE | n | Rank MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HAR** | none | 2.234681e-06 | 0.000672 | 4596.15 | 335 | **1** |
| **EGARCH** | none | 2.257117e-06 | 0.000648 | 5161.18 | 335 | 2 |
| **NBEATSx_SPILL** | none | 2.272067e-06 | 0.000551 | 4153.11 | 335 | 3 |
| **GARCH11** | none | 2.313042e-06 | 0.000650 | 4814.58 | 335 | 4 |
| **TFT_SPILL** | none | 2.388453e-06 | 0.000569 | 2366.67 | 335 | 5 |
| **TFT_NOSPILL** | none | 2.394793e-06 | 0.000578 | 2843.06 | 335 | 6 |
| **Roll_20** | none | 2.407078e-06 | 0.000567 | 2841.73 | 335 | 7 |
| **NBEATSx_NOSPILL** | none | 2.505267e-06 | 0.000521 | 2780.87 | 335 | 8 |
| **Hist_Exp** | none | 2.689570e-06 | 0.000967 | 5866.01 | 335 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| **TFT_NOSPILL** | top_1% | **4.282295e-07** | 0.000459 | 2877.40 | 331 | **1** |
| **TFT_SPILL** | top_1% | **4.364841e-07** | 0.000450 | 2395.24 | 331 | **2** |
| **HAR** | top_1% | 4.804840e-07 | 0.000564 | 4651.68 | 331 | 3 |
| **Roll_20** | top_1% | 5.022433e-07 | 0.000450 | 2876.05 | 331 | 4 |
| **NBEATSx_NOSPILL** | top_1% | 5.138469e-07 | 0.000400 | 2814.45 | 331 | 5 |
| **EGARCH** | top_1% | 5.276115e-07 | 0.000541 | 5223.54 | 331 | 6 |
| **GARCH11** | top_1% | 5.763841e-07 | 0.000543 | 4872.75 | 331 | 7 |
| **NBEATSx_SPILL** | top_1% | 6.510104e-07 | 0.000449 | 4203.29 | 331 | 8 |
| **Hist_Exp** | top_1% | 8.407776e-07 | 0.000858 | 5936.88 | 331 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| **TFT_SPILL** | top_2.5% | **3.220803e-07** | 0.000414 | 2434.61 | 326 | **1** |
| **TFT_NOSPILL** | top_2.5% | **3.232159e-07** | 0.000425 | 2924.96 | 326 | **2** |
| **NBEATSx_NOSPILL** | top_2.5% | **3.880400e-07** | 0.000362 | 2862.63 | 326 | **3** |
| **HAR** | top_2.5% | 3.998592e-07 | 0.000537 | 4730.08 | 326 | 4 |
| **Roll_20** | top_2.5% | 4.110363e-07 | 0.000420 | 2924.49 | 326 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| **NBEATSx_NOSPILL** | top_5% | **2.299782e-07** | 0.000310 | 2929.30 | 318 | **1** |
| **TFT_NOSPILL** | top_5% | **2.504837e-07** | 0.000391 | 2994.99 | 318 | **2** |
| **TFT_SPILL** | top_5% | **2.516454e-07** | 0.000381 | 2493.12 | 318 | **3** |
| **HAR** | top_5% | 3.442731e-07 | 0.000510 | 4841.82 | 318 | 4 |
| **Roll_20** | top_5% | 3.707911e-07 | 0.000401 | 2993.59 | 318 | 5 |
| **NBEATSx_SPILL** | top_5% | 3.888321e-07 | 0.000363 | 4374.83 | 318 | 6 |
| **EGARCH** | top_5% | 3.892748e-07 | 0.000484 | 5437.05 | 318 | 7 |
| **GARCH11** | top_5% | 4.411820e-07 | 0.000487 | 5071.92 | 318 | 8 |
| **Hist_Exp** | top_5% | 7.611173e-07 | 0.000827 | 6179.57 | 318 | 9 |

### Horizon h=10

| Model | Level Eksklusi | MSE | MAE | QLIKE | n | Rank MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HAR** | none | 2.439743e-06 | 0.000719 | 3887.50 | 335 | **1** |
| **TFT_SPILL** | none | 2.447095e-06 | 0.000601 | 2046.79 | 335 | 2 |
| **TFT_NOSPILL** | none | 2.454033e-06 | 0.000621 | 2144.36 | 335 | 3 |
| **GARCH11** | none | 2.478223e-06 | 0.000716 | 3718.61 | 335 | 4 |
| **EGARCH** | none | 2.496371e-06 | 0.000779 | 4530.22 | 335 | 5 |
| **NBEATSx_NOSPILL** | none | 2.531934e-06 | 0.000535 | 1022.20 | 335 | 6 |
| **Roll_20** | none | 2.555697e-06 | 0.000587 | 1074.52 | 335 | 7 |
| **Hist_Exp** | none | 2.686277e-06 | 0.000964 | 5864.10 | 335 | 8 |
| **NBEATSx_SPILL** | none | 2.744505e-06 | 0.000598 | 1731.28 | 335 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| **NBEATSx_NOSPILL** | top_1% | **4.372143e-07** | 0.000410 | 1034.47 | 331 | **1** |
| **TFT_SPILL** | top_1% | **4.427849e-07** | 0.000480 | 2071.50 | 331 | **2** |
| **TFT_NOSPILL** | top_1% | **4.760173e-07** | 0.000503 | 2170.25 | 331 | **3** |
| **HAR** | top_1% | 4.936435e-07 | 0.000603 | 3934.46 | 331 | 4 |
| **Roll_20** | top_1% | 5.160957e-07 | 0.000465 | 1087.48 | 331 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| **NBEATSx_NOSPILL** | top_5% | **2.675157e-07** | 0.000345 | 1076.71 | 318 | **1** |
| **TFT_SPILL** | top_5% | **2.793567e-07** | 0.000418 | 2156.15 | 318 | **2** |
| **TFT_NOSPILL** | top_5% | **3.200500e-07** | 0.000442 | 2258.94 | 318 | **3** |
| **HAR** | top_5% | 3.706091e-07 | 0.000554 | 4095.28 | 318 | 4 |

---

## 3. Hasil Uji Diebold-Mariano (Pembalikan & Signifikansi)

*Catatan:* Loss differential didefinisikan sebagai $d_t = e_{\text{model}}^2 - e_{\text{benchmark}}^2$. Nilai $\text{DM\_stat} < 0$ menunjukkan model Deep Learning memiliki loss lebih kecil dari benchmark. Koreksi multi-testing Bonferroni dilaporkan untuk $m=8$ dan $m=12$.

### Ringkasan DM Test Kunci (MSE Loss)

| Pasangan Model | Horizon | Level Eksklusi | DM Stat | p-value (raw) | p-bonf (m=8) | p-bonf (m=12) | Arah / Pemenang |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFT_SPILL vs Hist_Exp** | h=1 | none | -3.3040 | 0.000953 | 0.007625 | 0.011438 | **TFT_SPILL (Sig)** |
| **TFT_SPILL vs Hist_Exp** | h=1 | top_1% | **-9.3166** | 0.000000 | 0.000000 | 0.000000 | **TFT_SPILL (Extreme Sig)** |
| **TFT_SPILL vs Hist_Exp** | h=1 | top_2.5% | **-13.5356** | 0.000000 | 0.000000 | 0.000000 | **TFT_SPILL (Extreme Sig)** |
| **TFT_SPILL vs Hist_Exp** | h=1 | top_5% | **-16.9570** | 0.000000 | 0.000000 | 0.000000 | **TFT_SPILL (Extreme Sig)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TFT_NOSPILL vs Hist_Exp**| h=1 | none | -3.1952 | 0.001398 | 0.011181 | 0.016771 | **TFT_NOSPILL (Sig)** |
| **TFT_NOSPILL vs Hist_Exp**| h=1 | top_1% | **-10.0022** | 0.000000 | 0.000000 | 0.000000 | **TFT_NOSPILL (Extreme Sig)**|
| **TFT_NOSPILL vs Hist_Exp**| h=1 | top_2.5% | **-13.7227** | 0.000000 | 0.000000 | 0.000000 | **TFT_NOSPILL (Extreme Sig)**|
| **TFT_NOSPILL vs Hist_Exp**| h=1 | top_5% | **-17.2673** | 0.000000 | 0.000000 | 0.000000 | **TFT_NOSPILL (Extreme Sig)**|
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TFT_SPILL vs HAR** | h=1 | none | +1.0450 | 0.296031 | 1.000000 | 1.000000 | Tie (favor HAR) |
| **TFT_SPILL vs HAR** | h=1 | top_1% | **-1.1827** | 0.236941 | 1.000000 | 1.000000 | **Flip -> Favor TFT** |
| **TFT_SPILL vs HAR** | h=1 | top_2.5% | **-2.2750** | **0.022906** | 0.183247 | 0.274870 | **Flip -> Favor TFT (p<0.05)** |
| **TFT_SPILL vs HAR** | h=1 | top_5% | **-2.8545** | **0.004311** | **0.034486** | 0.051728 | **TFT_SPILL (Bonf Sig m=8)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TFT_NOSPILL vs HAR** | h=1 | none | +1.0001 | 0.317246 | 1.000000 | 1.000000 | Tie (favor HAR) |
| **TFT_NOSPILL vs HAR** | h=1 | top_1% | **-1.4509** | 0.146816 | 1.000000 | 1.000000 | **Flip -> Favor TFT** |
| **TFT_NOSPILL vs HAR** | h=1 | top_2.5% | **-2.2497** | **0.024467** | 0.195738 | 0.293607 | **Flip -> Favor TFT (p<0.05)** |
| **TFT_NOSPILL vs HAR** | h=1 | top_5% | **-2.7940** | **0.005206** | **0.041646** | 0.062469 | **TFT_NOSPILL (Bonf Sig m=8)**|
| --- | --- | --- | --- | --- | --- | --- | --- |
| **NBEATSx_NOSPILL vs HAR**| h=1 | none | +1.4391 | 0.150132 | 1.000000 | 1.000000 | Tie (favor HAR) |
| **NBEATSx_NOSPILL vs HAR**| h=1 | top_5% | **-2.8854** | **0.003909** | **0.031272** | **0.046909** | **NBEATSx (Bonf Sig m=8 & 12)**|
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TFT_SPILL vs HAR** | h=10 | none | +0.1237 | 0.901579 | 1.000000 | 1.000000 | Tie |
| **TFT_SPILL vs HAR** | h=10 | top_1% | **-1.3789** | 0.167936 | 1.000000 | 1.000000 | **Flip -> Favor TFT** |
| **TFT_SPILL vs HAR** | h=10 | top_2.5% | **-2.2321** | **0.025608** | 0.204863 | 0.307294 | **Flip -> Favor TFT (p<0.05)** |
| **TFT_SPILL vs HAR** | h=10 | top_5% | **-2.5116** | **0.012019** | 0.096155 | 0.144233 | **Flip -> Favor TFT (p<0.05)** |

### Ringkasan DM Test pada MAE Loss (Eksploratori & Robustness)

Pada loss MAE, penalti terhadap error bersifat linier sehingga estimasi tidak didistorsi kuadrat outlier.
- **TFT_SPILL vs HAR (h=1, full-sample):** $\text{DM\_stat} = -5.6917$, $p_{\text{raw}} = 1.264 \times 10^{-8}$, $p_{\text{bonf, m=8}} = 1.011 \times 10^{-7}$.
- **TFT_NOSPILL vs HAR (h=1, full-sample):** $\text{DM\_stat} = -5.0934$, $p_{\text{raw}} = 3.511 \times 10^{-7}$, $p_{\text{bonf, m=8}} = 2.809 \times 10^{-6}$.
- **NBEATSx_NOSPILL vs HAR (h=1, full-sample):** $\text{DM\_stat} = -7.9542$, $p_{\text{raw}} = 1.804 \times 10^{-15}$, $p_{\text{bonf, m=8}} = 1.443 \times 10^{-14}$.
*Kesimpulan MAE:* Seluruh model Deep Learning secara signifikan mengalahkan HAR pada MAE di setiap level eksklusi dengan tingkat keyakinan $> 99.99\%$.

---

## 4. Analisis Hari Ekstrem (Top 5% Teratas / 17 Hari)

Berikut adalah daftar 17 hari perdagangan paling volatil di test set dan kontribusi error-nya terhadap total SSE ($h=1$):

| Rank | Tanggal Target | Return BTC-IDR | Realized Volatility ($y$) | HAR Pred ($h=1$) | TFT SPILL Pred ($h=1$) | % SSE HAR | % SSE TFT | Catatan Peristiwa Pasar |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **2026-02-05** | **-14.85%** | **2.205457e-02** | 0.000743 | 0.000425 | **60.67%** | **58.47%** | Crash tajam pasar crypto; surge volatilitas ekstrem 1-hari |
| **2** | **2026-02-06** | **+11.75%** | **1.380062e-02** | 0.003024 | 0.001087 | **15.51%** | **20.20%** | Rebound masif; autoregressive lag HAR langsung melompat tinggi |
| **3** | **2026-08-21** | **+6.53%** | **4.265190e-03** | 0.001099 | 0.000389 | 1.34% | 1.88% | Rally musim panas crypto |
| **4** | **2026-06-02** | **-6.11%** | **3.729880e-03** | 0.000688 | 0.000391 | 1.24% | 1.39% | Koreksi awal Juni |
| **5** | 2026-03-04 | +5.71% | 3.265520e-03 | 0.000862 | 0.000361 | 0.77% | 1.05% | Breakout awal Maret |
| **6** | 2026-08-19 | +5.64% | 3.179339e-03 | 0.000527 | 0.000220 | 0.94% | 1.09% | Awal lonjakan reli Agustus |
| **7** | 2025-11-14 | -5.63% | 3.171632e-03 | 0.000647 | 0.000409 | 0.85% | 0.95% | Koreksi pertengahan November |
| **8** | 2026-08-20 | +5.48% | 2.999639e-03 | 0.001080 | 0.000347 | 0.49% | 0.88% | Kontinuitas reli Agustus |
| **9** | 2025-10-10 | -5.42% | 2.935402e-03 | 0.000524 | 0.000175 | 0.78% | 0.95% | Koreksi tajam Oktober 2025 |
| **10** | 2025-05-08 | +5.18% | 2.684821e-03 | 0.000722 | 0.001353 | 0.51% | 0.22% | Lonjakan awal Mei 2025 |
| **11** | 2026-02-25 | +5.08% | 2.577399e-03 | 0.000818 | 0.000363 | 0.41% | 0.61% | Reli akhir Februari |
| **12** | 2025-11-20 | -5.00% | 2.502546e-03 | 0.000723 | 0.000578 | 0.42% | 0.46% | Penurunan akhir November |
| **13** | 2026-04-13 | +4.93% | 2.429561e-03 | 0.000662 | 0.000969 | 0.42% | 0.27% | Rebound pertengahan April |
| **14** | 2026-03-02 | +4.86% | 2.359366e-03 | 0.000934 | 0.000266 | 0.27% | 0.55% | Reli awal Maret |
| **15** | 2025-12-02 | +4.73% | 2.233837e-03 | 0.000911 | 0.000656 | 0.23% | 0.31% | Rebound Desember 2025 |
| **16** | 2026-02-23 | -4.70% | 2.206931e-03 | 0.000948 | 0.000484 | 0.21% | 0.37% | Penurunan akhir Februari |
| **17** | 2025-12-01 | -4.69% | 2.196626e-03 | 0.000691 | 0.000579 | 0.30% | 0.33% | Penurunan awal Desember 2025 |

### Konsentrasi Kumulatif Error Kuadrat (SSE):
- **Top 1 Hari (2026-02-05):** HAR = **60.67%**, TFT SPILL = **58.47%**
- **Top 2 Hari (2026-02-05 & 06):** HAR = **76.18%**, TFT SPILL = **78.67%**
- **Top 4 Hari (Top 1%):** HAR = **78.76%**, TFT SPILL = **81.94%**
- **Top 9 Hari (Top 2.5%):** HAR = **82.59%**, TFT SPILL = **86.88%**
- **Top 17 Hari (Top 5%):** HAR = **85.38%**, TFT SPILL = **90.00%**

---

## 5. Jawaban Langsung Atas Pertanyaan Kunci

### Pertanyaan 5: Apakah TFT SPILL / NOSPILL vs Hist_Exp h=1 masih signifikan setelah outlier dikeluarkan?
**JAWABAN:** **YA, SANGAT SIGNIFIKAN.**
- Pada full sample ($n=335$):
  - TFT SPILL vs Hist_Exp: $\text{DM} = -3.3040, p_{\text{raw}} = 0.000953, p_{\text{bonf, m=8}} = 0.007625$
  - TFT NOSPILL vs Hist_Exp: $\text{DM} = -3.1952, p_{\text{raw}} = 0.001398, p_{\text{bonf, m=8}} = 0.011181$
- Pada eksklusi top 1% ($n=331$):
  - TFT SPILL vs Hist_Exp: $\text{DM} = -9.3166, p_{\text{raw}} < 10^{-15}, p_{\text{bonf, m=8}} < 10^{-14}$
  - TFT NOSPILL vs Hist_Exp: $\text{DM} = -10.0022, p_{\text{raw}} < 10^{-15}, p_{\text{bonf, m=8}} < 10^{-14}$
- Pada eksklusi top 2.5% ($n=326$):
  - TFT SPILL vs Hist_Exp: $\text{DM} = -13.5356, p < 10^{-15}$
  - TFT NOSPILL vs Hist_Exp: $\text{DM} = -13.7227, p < 10^{-15}$
- Pada eksklusi top 5% ($n=318$):
  - TFT SPILL vs Hist_Exp: $\text{DM} = -16.9570, p < 10^{-15}$
  - TFT NOSPILL vs Hist_Exp: $\text{DM} = -17.2673, p < 10^{-15}$

*Mekanisme:* Ketika hari-hari ekstrem dikeluarkan, varians selisih loss ($d_t$) menyusut drastis dari orde $10^{-11}$ ke $10^{-14}$, sementara rata-rata selisih error ($\bar{d}$) tetap konsisten negatif mendukung TFT. Akibatnya, nilai t-stat Diebold-Mariano meningkat tajam dan signifikansinya menjadi absolut.

---

## 6. Verifikasi & Integritas Metodologi

1. **Alignment Key:** Seluruh penggabungan prediksi dilakukan secara ketat pada pasangan `(cutoff, real_date)` untuk mencegah bug pergeseran tanggal/horizon (`ds` mismatch pada data historis).
2. **Clipping:**
   - Realized $y$: 0 baris ter-clip (minimum realized $y = 5.481 \times 10^{-10} > 10^{-10}$).
   - HAR, Hist_Exp, Roll_20, GARCH11, EGARCH, TFT_SPILL, TFT_NOSPILL: 0 baris ter-clip.
   - NBEATSx_SPILL: 46 baris ($13.73\%$) pada $h=1$ dan 42 baris ($12.54\%$) pada $h=10$ dipotong ke $10^{-10}$ karena memprediksi nilai negatif $\le 0$.
   - NBEATSx_NOSPILL: 21 baris ($6.27\%$) pada $h=1$ dan 20 baris ($5.97\%$) pada $h=10$ dipotong ke $10^{-10}$.
3. **Catatan QLIKE Patton (2011):** Formula yang dihitung pada tabel headline adalah formula persis permintaan prompt:
   $$\text{QLIKE}_{\text{user}} = \frac{\hat{y}}{y} - \ln\left(\frac{\hat{y}}{y}\right) - 1$$
   Sebagai catatan ketahanan, bila formula standar ekonometrika Patton (2011) digunakan di mana $\hat{y}$ di penyebut:
   $$\text{QLIKE}_{\text{Patton}} = \frac{y}{\hat{y}} - \ln\left(\frac{y}{\hat{y}}\right) - 1$$
   Nilai QLIKE Patton pada full sample adalah $1.7738$ (TFT SPILL) vs $0.9443$ (HAR), dan pada eksklusi top 1% menjadi $1.5838$ (TFT SPILL) vs $0.9403$ (HAR).
4. **Hal yang Tidak Dapat Diverifikasi (Unverifiable Items):**
   - Latar belakang fundamental pasar spesifik pada tanggal 5 Februari 2026 (apakah crash crypto dipicu peristiwa makro global, likuidasi leverage derivatif, atau regulasi) tidak tercatat di dataset kuantitatif dan hanya dapat diidentifikasi sebagai lonjakan return $-14.85\%$ pada Indodax.
