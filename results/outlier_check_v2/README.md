# RSIC v6 Outlier Sensitivity Check v2: Analysis Report

**Status:** EXPLORATORY SENSITIVITY ANALYSIS (READ-ONLY EVALUATION)
**Evaluation Window:** 335 cutoffs (Target dates: 2025-04-24 to 2026-09-04)
**Target:** Realized Volatility $y_t = r_t^2$ (BTC/IDR Indodax)
**Execution Environment:** CPU only, deterministic evaluation, no model retraining.

---

## 1. Temuan Faktual Utama (Apa Adanya)

1. **Posisi HAR pada Full Sample Didorong Hari Ekstrem 5–6 Februari 2026:**
   - Dua hari berturut-turut pada 5 dan 6 Februari 2026 (crash $-14.85\%$ disusul rebound $+11.75\%$) menyumbang **76.12%** dari total SSE model HAR ($h=1$) dan **79.61%** dari total SSE model TFT SPILL ($h=1$).
   - Pada sampel penuh ($N=335$), HAR mencatat MSE terendah pada $h=1$ ($2.235 \times 10^{-6}$) dan $h=10$ ($2.440 \times 10^{-6}$).

2. **Pengaruh Winsorize dan Event Exclusion terhadap Ranking MSE:**
   - Saat target $y$ di-winsorize pada persentil 99, 97.5, atau 95 (prediksi $\hat{y}$ tidak diubah), atau saat hanya 2 hari event-window (5–6 Feb 2026) dibuang ($N=333$):
     - Pada $h=1$: TFT_NOSPILL dan TFT_SPILL memperoleh nilai point MSE lebih rendah daripada HAR (peringkat 1 dan 2), sementara HAR berada pada peringkat 3 atau 4.
     - Pada $h=10$: TFT_SPILL dan NBEATSx_NOSPILL memperoleh point MSE lebih rendah daripada HAR (peringkat 1 dan 2), sementara HAR berada pada peringkat 4.

3. **Uji Diebold-Mariano TFT vs. HAR Tetap Tidak Signifikan (Statistically a Tie):**
   - Meskipun point MSE TFT menjadi lebih kecil daripada HAR setelah winsorize atau event exclusion, perbedaan loss antara TFT dan HAR **tidak signifikan secara statistik** pada level manapun ($p_{\text{raw}} \ge 0.17$, $p_{\text{bonf}} = 1.0000$).
   - Dengan demikian, data menunjukkan status seri statistik (tie), bukan pembalikan keunggulan yang signifikan.

4. **Keunggulan TFT vs. Hist_Exp Tetap Signifikan:**
   - Performa TFT SPILL dan TFT_NOSPILL terhadap baseline `Hist_Exp` pada $h=1$ tetap signifikan secara statistik di seluruh level winsorize dan event exclusion ($p_{\text{bonf}} < 0.05$).

5. **Metrik QLIKE (Patton) dan Prediksi Negatif NBEATSx:**
   - Patton (2011) mendefinisikan loss QLIKE sebagai $L(y, \hat{y}) = y/\hat{y} - \ln(y/\hat{y}) - 1$.
   - Model NBEATSx menghasilkan prediksi $\le 0$ pada 20 hingga 46 hari perdagangan ($h=1$ dan $h=10$). Tanpa clipping buatan, nilai QLIKE untuk NBEATSx tidak dapat dihitung secara matematis dan ditandai sebagai **QLIKE tidak applicable**.
   - Untuk model yang seluruh prediksinya $> 0$ (HAR, TFT, GARCH, EGARCH, Hist_Exp, Roll_20), nilai rata-rata QLIKE berkisar antara 1.52 hingga 1.97.

6. **Mean Bias Menjelaskan Pergeseran Ranking:**
   - HAR memiliki bias positif rata-rata ($+1.81 \times 10^{-4}$ pada $h=1$, $+2.37 \times 10^{-4}$ pada $h=10$) karena memproyeksikan tingkat volatilitas baseline yang lebih tinggi.
   - TFT memiliki bias rata-rata mendekati nol ($-5.07 \times 10^{-5}$ untuk SPILL, $-4.84 \times 10^{-6}$ untuk NOSPILL pada $h=1$).
   - NBEATSx memiliki bias negatif rata-rata ($-2.51 \times 10^{-4}$ hingga $-2.70 \times 10^{-4}$).
   - Ketika nilai target puncak di-cap oleh winsorize atau dipotong oleh event exclusion, nilai rata-rata $y$ turun dari $\approx 5.0 \times 10^{-4}$ menjadi $\approx 3.8 \times 10^{-4}$. Hal ini secara mekanis memperbesar penalti kuadratik bagi model dengan level prediksi rata-rata lebih tinggi (seperti HAR) dan menguntungkan model dengan level prediksi rata-rata lebih rendah (seperti TFT dan NBEATSx).

---

## 2. Tabel Metrik Evaluasi Berdasarkan Regime (MSE, MAE, Mean Bias, QLIKE)

*Catatan Regime:*
- `Full Sample` ($N=335$): Data uji lengkap 24 April 2025 s.d. 4 September 2026.
- `Winsorize 99%` ($N=335$): Target $y$ di-cap pada persentil 99 ($y_{\text{cap}} = 3.572 \times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.
- `Winsorize 97.5%` ($N=335$): Target $y$ di-cap pada persentil 97.5 ($y_{\text{cap}} = 2.848 \times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.
- `Winsorize 95%` ($N=335$): Target $y$ di-cap pada persentil 95 ($y_{\text{cap}} = 2.195 \times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.
- `Event Excl (Feb 5-6)` ($N=333$): Hanya tanggal 2026-02-05 dan 2026-02-06 yang dikeluarkan dari evaluasi.

### Horizon h=1

| Model | Regime | n | MSE | MAE | Mean Bias (pred - y) | QLIKE (Patton) | Rank MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFT_NOSPILL** | Event Excl (Feb 5-6) | 333 | 5.030723e-07 | 4.776151e-04 | +9.885397e-05 | 1.6003 | 1 |
| **TFT_SPILL** | Event Excl (Feb 5-6) | 333 | 5.124554e-07 | 4.692027e-04 | +5.209923e-05 | 1.6159 | 2 |
| **HAR** | Event Excl (Feb 5-6) | 333 | 5.354802e-07 | 5.797120e-04 | +2.788664e-04 | 1.6301 | 3 |
| **Roll_20** | Event Excl (Feb 5-6) | 333 | 5.803447e-07 | 4.688730e-04 | +9.595830e-05 | 1.6265 | 4 |
| **EGARCH** | Event Excl (Feb 5-6) | 333 | 5.809148e-07 | 5.559109e-04 | +2.339514e-04 | 1.5799 | 5 |
| **NBEATSx_NOSPILL** | Event Excl (Feb 5-6) | 333 | 5.883199e-07 | 4.196015e-04 | -1.670654e-04 | N/A (pred $\le$ 0) | 6 |
| **GARCH11** | Event Excl (Feb 5-6) | 333 | 6.322898e-07 | 5.583752e-04 | +2.361549e-04 | 1.5797 | 7 |
| **NBEATSx_SPILL** | Event Excl (Feb 5-6) | 333 | 7.125237e-07 | 4.659493e-04 | -1.645833e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Event Excl (Feb 5-6) | 333 | 8.896326e-07 | 8.711075e-04 | +6.597573e-04 | 1.9003 | 9 |
| **HAR** | Full Sample | 335 | 2.234681e-06 | 6.720358e-04 | +1.814168e-04 | 1.7020 | 1 |
| **EGARCH** | Full Sample | 335 | 2.257117e-06 | 6.476385e-04 | +1.375082e-04 | 1.6406 | 2 |
| **NBEATSx_SPILL** | Full Sample | 335 | 2.272067e-06 | 5.508511e-04 | -2.512843e-04 | N/A (pred $\le$ 0) | 3 |
| **GARCH11** | Full Sample | 335 | 2.313042e-06 | 6.499103e-04 | +1.398763e-04 | 1.6537 | 4 |
| **TFT_SPILL** | Full Sample | 335 | 2.388453e-06 | 5.689191e-04 | -5.072938e-05 | 1.7738 | 5 |
| **TFT_NOSPILL** | Full Sample | 335 | 2.394793e-06 | 5.778646e-04 | -4.837101e-06 | 1.7705 | 6 |
| **Roll_20** | Full Sample | 335 | 2.407078e-06 | 5.668224e-04 | -5.363209e-06 | 1.7475 | 7 |
| **NBEATSx_NOSPILL** | Full Sample | 335 | 2.505267e-06 | 5.210675e-04 | -2.700390e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Full Sample | 335 | 2.689570e-06 | 9.667459e-04 | +5.549794e-04 | 1.9695 | 9 |
| **TFT_NOSPILL** | Winsorize 95% | 335 | 3.792594e-07 | 4.546400e-04 | +1.183875e-04 | 1.5357 | 1 |
| **TFT_SPILL** | Winsorize 95% | 335 | 3.840391e-07 | 4.456945e-04 | +7.249522e-05 | 1.5404 | 2 |
| **HAR** | Winsorize 95% | 335 | 4.249847e-07 | 5.537605e-04 | +3.046414e-04 | 1.5950 | 3 |
| **NBEATSx_NOSPILL** | Winsorize 95% | 335 | 4.510083e-07 | 3.978429e-04 | -1.468144e-04 | N/A (pred $\le$ 0) | 4 |
| **Roll_20** | Winsorize 95% | 335 | 4.638455e-07 | 4.472145e-04 | +1.178614e-04 | 1.5430 | 5 |
| **EGARCH** | Winsorize 95% | 335 | 4.700797e-07 | 5.301840e-04 | +2.607328e-04 | 1.5405 | 6 |
| **GARCH11** | Winsorize 95% | 335 | 5.216467e-07 | 5.336137e-04 | +2.631009e-04 | 1.5392 | 7 |
| **NBEATSx_SPILL** | Winsorize 95% | 335 | 6.175460e-07 | 4.498228e-04 | -1.280597e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 95% | 335 | 7.899198e-07 | 8.435213e-04 | +6.782040e-04 | 1.8735 | 9 |
| **TFT_NOSPILL** | Winsorize 97.5% | 335 | 4.678868e-07 | 4.770512e-04 | +9.597633e-05 | 1.5820 | 1 |
| **TFT_SPILL** | Winsorize 97.5% | 335 | 4.730659e-07 | 4.681057e-04 | +5.008405e-05 | 1.5921 | 2 |
| **HAR** | Winsorize 97.5% | 335 | 4.932240e-07 | 5.722741e-04 | +2.822302e-04 | 1.6147 | 3 |
| **EGARCH** | Winsorize 97.5% | 335 | 5.382089e-07 | 5.486976e-04 | +2.383217e-04 | 1.5629 | 4 |
| **Roll_20** | Winsorize 97.5% | 335 | 5.393506e-07 | 4.660090e-04 | +9.545023e-05 | 1.5944 | 5 |
| **NBEATSx_NOSPILL** | Winsorize 97.5% | 335 | 5.532787e-07 | 4.202541e-04 | -1.692256e-04 | N/A (pred $\le$ 0) | 6 |
| **GARCH11** | Winsorize 97.5% | 335 | 5.901387e-07 | 5.521274e-04 | +2.406898e-04 | 1.5630 | 7 |
| **NBEATSx_SPILL** | Winsorize 97.5% | 335 | 6.927220e-07 | 4.683364e-04 | -1.504709e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 97.5% | 335 | 8.548900e-07 | 8.659325e-04 | +6.557928e-04 | 1.8861 | 9 |
| **TFT_NOSPILL** | Winsorize 99% | 335 | 5.337961e-07 | 4.896190e-04 | +8.340847e-05 | 1.6070 | 1 |
| **TFT_SPILL** | Winsorize 99% | 335 | 5.397167e-07 | 4.806735e-04 | +3.751619e-05 | 1.6194 | 2 |
| **HAR** | Winsorize 99% | 335 | 5.426035e-07 | 5.837903e-04 | +2.696624e-04 | 1.6252 | 3 |
| **EGARCH** | Winsorize 99% | 335 | 5.862755e-07 | 5.593930e-04 | +2.257538e-04 | 1.5740 | 4 |
| **Roll_20** | Winsorize 99% | 335 | 5.988977e-07 | 4.785768e-04 | +8.288236e-05 | 1.6239 | 5 |
| **NBEATSx_NOSPILL** | Winsorize 99% | 335 | 6.236284e-07 | 4.328219e-04 | -1.817935e-04 | N/A (pred $\le$ 0) | 6 |
| **GARCH11** | Winsorize 99% | 335 | 6.387601e-07 | 5.616647e-04 | +2.281219e-04 | 1.5754 | 7 |
| **NBEATSx_SPILL** | Winsorize 99% | 335 | 7.364482e-07 | 4.765801e-04 | -1.630388e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 99% | 335 | 9.081509e-07 | 8.785004e-04 | +6.432250e-04 | 1.8943 | 9 |

### Horizon h=10

| Model | Regime | n | MSE | MAE | Mean Bias (pred - y) | QLIKE (Patton) | Rank MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFT_SPILL** | Event Excl (Feb 5-6) | 333 | 5.219887e-07 | 4.995592e-04 | +1.135936e-04 | 1.5954 | 1 |
| **NBEATSx_NOSPILL** | Event Excl (Feb 5-6) | 333 | 5.258622e-07 | 4.307501e-04 | -1.549285e-04 | N/A (pred $\le$ 0) | 2 |
| **TFT_NOSPILL** | Event Excl (Feb 5-6) | 333 | 5.477388e-07 | 5.206476e-04 | +1.453921e-04 | 1.6172 | 3 |
| **HAR** | Event Excl (Feb 5-6) | 333 | 5.577775e-07 | 6.193585e-04 | +3.425202e-04 | 1.6684 | 4 |
| **GARCH11** | Event Excl (Feb 5-6) | 333 | 5.908355e-07 | 6.161573e-04 | +3.513765e-04 | 1.6579 | 5 |
| **Roll_20** | Event Excl (Feb 5-6) | 333 | 5.997429e-07 | 4.847288e-04 | +9.431802e-05 | 1.6770 | 6 |
| **EGARCH** | Event Excl (Feb 5-6) | 333 | 6.342031e-07 | 6.802698e-04 | +4.345319e-04 | 1.7174 | 7 |
| **NBEATSx_SPILL** | Event Excl (Feb 5-6) | 333 | 7.262223e-07 | 4.935389e-04 | -1.007552e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Event Excl (Feb 5-6) | 333 | 8.857039e-07 | 8.687604e-04 | +6.535467e-04 | 1.8738 | 9 |
| **HAR** | Full Sample | 335 | 2.439743e-06 | 7.188632e-04 | +2.372729e-04 | 1.7992 | 1 |
| **TFT_SPILL** | Full Sample | 335 | 2.447095e-06 | 6.010015e-04 | +8.490609e-06 | 1.8032 | 2 |
| **TFT_NOSPILL** | Full Sample | 335 | 2.454033e-06 | 6.214534e-04 | +4.060999e-05 | 1.7844 | 3 |
| **GARCH11** | Full Sample | 335 | 2.478223e-06 | 7.158521e-04 | +2.459053e-04 | 1.7960 | 4 |
| **EGARCH** | Full Sample | 335 | 2.496371e-06 | 7.788658e-04 | +3.292803e-04 | 1.8275 | 5 |
| **NBEATSx_NOSPILL** | Full Sample | 335 | 2.531934e-06 | 5.350320e-04 | -2.608570e-04 | N/A (pred $\le$ 0) | 6 |
| **Roll_20** | Full Sample | 335 | 2.555697e-06 | 5.871553e-04 | -1.156542e-05 | 2.0094 | 7 |
| **Hist_Exp** | Full Sample | 335 | 2.686277e-06 | 9.644415e-04 | +5.487772e-04 | 1.9435 | 8 |
| **NBEATSx_SPILL** | Full Sample | 335 | 2.744505e-06 | 5.977772e-04 | -2.073384e-04 | N/A (pred $\le$ 0) | 9 |
| **TFT_SPILL** | Winsorize 95% | 335 | 4.006229e-07 | 4.777769e-04 | +1.317152e-04 | 1.5282 | 1 |
| **NBEATSx_NOSPILL** | Winsorize 95% | 335 | 4.104649e-07 | 4.118073e-04 | -1.376324e-04 | N/A (pred $\le$ 0) | 2 |
| **TFT_NOSPILL** | Winsorize 95% | 335 | 4.305042e-07 | 4.982288e-04 | +1.638346e-04 | 1.5602 | 3 |
| **HAR** | Winsorize 95% | 335 | 4.502130e-07 | 5.956386e-04 | +3.604976e-04 | 1.6359 | 4 |
| **GARCH11** | Winsorize 95% | 335 | 4.872428e-07 | 5.943634e-04 | +3.691299e-04 | 1.6251 | 5 |
| **Roll_20** | Winsorize 95% | 335 | 4.909129e-07 | 4.654746e-04 | +1.116592e-04 | 1.6013 | 6 |
| **EGARCH** | Winsorize 95% | 335 | 5.292507e-07 | 6.556412e-04 | +4.525049e-04 | 1.6865 | 7 |
| **NBEATSx_SPILL** | Winsorize 95% | 335 | 6.047943e-07 | 4.746246e-04 | -8.411381e-05 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 95% | 335 | 7.862603e-07 | 8.412168e-04 | +6.720018e-04 | 1.8474 | 9 |
| **TFT_SPILL** | Winsorize 97.5% | 335 | 4.906121e-07 | 5.001881e-04 | +1.093040e-04 | 1.5785 | 1 |
| **NBEATSx_NOSPILL** | Winsorize 97.5% | 335 | 5.057763e-07 | 4.342185e-04 | -1.600436e-04 | N/A (pred $\le$ 0) | 2 |
| **TFT_NOSPILL** | Winsorize 97.5% | 335 | 5.165873e-07 | 5.206400e-04 | +1.414234e-04 | 1.6033 | 3 |
| **HAR** | Winsorize 97.5% | 335 | 5.280404e-07 | 6.180498e-04 | +3.380864e-04 | 1.6579 | 4 |
| **GARCH11** | Winsorize 97.5% | 335 | 5.612460e-07 | 6.150387e-04 | +3.467188e-04 | 1.6475 | 5 |
| **Roll_20** | Winsorize 97.5% | 335 | 5.741872e-07 | 4.863418e-04 | +8.924802e-05 | 1.6657 | 6 |
| **EGARCH** | Winsorize 97.5% | 335 | 6.026865e-07 | 6.780524e-04 | +4.300937e-04 | 1.7052 | 7 |
| **NBEATSx_SPILL** | Winsorize 97.5% | 335 | 7.054219e-07 | 4.969637e-04 | -1.065250e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 97.5% | 335 | 8.511434e-07 | 8.636280e-04 | +6.495906e-04 | 1.8599 | 9 |
| **TFT_SPILL** | Winsorize 99% | 335 | 5.594073e-07 | 5.127559e-04 | +9.673618e-05 | 1.6081 | 1 |
| **NBEATSx_NOSPILL** | Winsorize 99% | 335 | 5.786264e-07 | 4.467864e-04 | -1.726115e-04 | N/A (pred $\le$ 0) | 2 |
| **TFT_NOSPILL** | Winsorize 99% | 335 | 5.827171e-07 | 5.332078e-04 | +1.288556e-04 | 1.6262 | 3 |
| **HAR** | Winsorize 99% | 335 | 5.894068e-07 | 6.306176e-04 | +3.255185e-04 | 1.6722 | 4 |
| **GARCH11** | Winsorize 99% | 335 | 6.230672e-07 | 6.276065e-04 | +3.341509e-04 | 1.6629 | 5 |
| **Roll_20** | Winsorize 99% | 335 | 6.420154e-07 | 4.989097e-04 | +7.668015e-05 | 1.7089 | 6 |
| **EGARCH** | Winsorize 99% | 335 | 6.625150e-07 | 6.906203e-04 | +4.175259e-04 | 1.7179 | 7 |
| **NBEATSx_SPILL** | Winsorize 99% | 335 | 7.798611e-07 | 5.095316e-04 | -1.190928e-04 | N/A (pred $\le$ 0) | 8 |
| **Hist_Exp** | Winsorize 99% | 335 | 9.043997e-07 | 8.761959e-04 | +6.370228e-04 | 1.8681 | 9 |

---

## 3. Ringkasan Mean Bias tiap Model across Regimes

Tabel ini menunjukkan rata-rata selisih prediksi terhadap target $\frac{1}{N} \sum (\hat{y}_t - y_t)$:

### Horizon h=1

| Model | full | win99 | win975 | win95 | event |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EGARCH** | +1.375082e-04 | +2.257538e-04 | +2.383217e-04 | +2.607328e-04 | +2.339514e-04 |
| **GARCH11** | +1.398763e-04 | +2.281219e-04 | +2.406898e-04 | +2.631009e-04 | +2.361549e-04 |
| **HAR** | +1.814168e-04 | +2.696624e-04 | +2.822302e-04 | +3.046414e-04 | +2.788664e-04 |
| **Hist_Exp** | +5.549794e-04 | +6.432250e-04 | +6.557928e-04 | +6.782040e-04 | +6.597573e-04 |
| **NBEATSx_NOSPILL** | -2.700390e-04 | -1.817935e-04 | -1.692256e-04 | -1.468144e-04 | -1.670654e-04 |
| **NBEATSx_SPILL** | -2.512843e-04 | -1.630388e-04 | -1.504709e-04 | -1.280597e-04 | -1.645833e-04 |
| **Roll_20** | -5.363209e-06 | +8.288236e-05 | +9.545023e-05 | +1.178614e-04 | +9.595830e-05 |
| **TFT_NOSPILL** | -4.837101e-06 | +8.340847e-05 | +9.597633e-05 | +1.183875e-04 | +9.885397e-05 |
| **TFT_SPILL** | -5.072938e-05 | +3.751619e-05 | +5.008405e-05 | +7.249522e-05 | +5.209923e-05 |

### Horizon h=10

| Model | full | win99 | win975 | win95 | event |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EGARCH** | +3.292803e-04 | +4.175259e-04 | +4.300937e-04 | +4.525049e-04 | +4.345319e-04 |
| **GARCH11** | +2.459053e-04 | +3.341509e-04 | +3.467188e-04 | +3.691299e-04 | +3.513765e-04 |
| **HAR** | +2.372729e-04 | +3.255185e-04 | +3.380864e-04 | +3.604976e-04 | +3.425202e-04 |
| **Hist_Exp** | +5.487772e-04 | +6.370228e-04 | +6.495906e-04 | +6.720018e-04 | +6.535467e-04 |
| **NBEATSx_NOSPILL** | -2.608570e-04 | -1.726115e-04 | -1.600436e-04 | -1.376324e-04 | -1.549285e-04 |
| **NBEATSx_SPILL** | -2.073384e-04 | -1.190928e-04 | -1.065250e-04 | -8.411381e-05 | -1.007552e-04 |
| **Roll_20** | -1.156542e-05 | +7.668015e-05 | +8.924802e-05 | +1.116592e-04 | +9.431802e-05 |
| **TFT_NOSPILL** | +4.060999e-05 | +1.288556e-04 | +1.414234e-04 | +1.638346e-04 | +1.453921e-04 |
| **TFT_SPILL** | +8.490609e-06 | +9.673618e-05 | +1.093040e-04 | +1.317152e-04 | +1.135936e-04 |

---

## 4. Hasil Uji Diebold-Mariano (EXPLORATORY)

Multi-testing correction:
- Family DL vs. 3 Baselines: $m = 12$ per horizon, $m = 24$ total per table.
- $p_{\text{bonf}} = \min(1.0, m \times p_{\text{raw}})$.
- Sign convention: $\text{DM\_stat} < 0$ menunjukkan Model 1 memiliki loss lebih rendah.

### Panel A: Deep Learning Models vs. HAR Benchmark (Loss = MSE)

| Pasangan | Horizon | Regime | DM Stat | p-value (raw) | p-bonf (m=12) | p-bonf (m=24) | Kesimpulan |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| TFT_SPILL vs HAR | h=1 | Full Sample | +1.0450 | 0.2960 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=1 | Full Sample | +1.0001 | 0.3172 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=1 | Full Sample | +0.2100 | 0.8337 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=1 | Full Sample | +1.4391 | 0.1501 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=1 | Winsorize 99% | -0.0668 | 0.9468 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=1 | Winsorize 99% | -0.2038 | 0.8385 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=1 | Winsorize 99% | +2.6287 | 0.0086 | 0.1028 | 0.2057 | Tie |
| NBEATSx_NOSPILL vs HAR | h=1 | Winsorize 99% | +1.0951 | 0.2735 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=1 | Winsorize 97.5% | -0.5183 | 0.6042 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=1 | Winsorize 97.5% | -0.6598 | 0.5094 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=1 | Winsorize 97.5% | +2.6158 | 0.0089 | 0.1068 | 0.2136 | Tie |
| NBEATSx_NOSPILL vs HAR | h=1 | Winsorize 97.5% | +0.8590 | 0.3903 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=1 | Winsorize 95% | -1.1667 | 0.2433 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=1 | Winsorize 95% | -1.3082 | 0.1908 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=1 | Winsorize 95% | +2.4646 | 0.0137 | 0.1646 | 0.3292 | Tie |
| NBEATSx_NOSPILL vs HAR | h=1 | Winsorize 95% | +0.4080 | 0.6833 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=1 | Event Excl (Feb 5-6) | -0.5703 | 0.5685 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=1 | Event Excl (Feb 5-6) | -0.8380 | 0.4020 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=1 | Event Excl (Feb 5-6) | +2.4375 | 0.0148 | 0.1775 | 0.3550 | Tie |
| NBEATSx_NOSPILL vs HAR | h=1 | Event Excl (Feb 5-6) | +0.7498 | 0.4534 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=10 | Full Sample | +0.1237 | 0.9016 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=10 | Full Sample | +0.2591 | 0.7956 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=10 | Full Sample | +1.4689 | 0.1419 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=10 | Full Sample | +0.6519 | 0.5144 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=10 | Winsorize 99% | -0.7486 | 0.4541 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=10 | Winsorize 99% | -0.1349 | 0.8927 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=10 | Winsorize 99% | +1.2133 | 0.2250 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=10 | Winsorize 99% | -0.1699 | 0.8651 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=10 | Winsorize 97.5% | -0.9745 | 0.3298 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=10 | Winsorize 97.5% | -0.2330 | 0.8158 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=10 | Winsorize 97.5% | +1.1382 | 0.2551 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=10 | Winsorize 97.5% | -0.3735 | 0.7087 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=10 | Winsorize 95% | -1.3654 | 0.1721 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=10 | Winsorize 95% | -0.4033 | 0.6867 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=10 | Winsorize 95% | +1.0099 | 0.3125 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=10 | Winsorize 95% | -0.7078 | 0.4790 | 1.0000 | 1.0000 | Tie |
| TFT_SPILL vs HAR | h=10 | Event Excl (Feb 5-6) | -0.8964 | 0.3700 | 1.0000 | 1.0000 | Tie |
| TFT_NOSPILL vs HAR | h=10 | Event Excl (Feb 5-6) | -0.2016 | 0.8402 | 1.0000 | 1.0000 | Tie |
| NBEATSx_SPILL vs HAR | h=10 | Event Excl (Feb 5-6) | +1.0824 | 0.2791 | 1.0000 | 1.0000 | Tie |
| NBEATSx_NOSPILL vs HAR | h=10 | Event Excl (Feb 5-6) | -0.5545 | 0.5793 | 1.0000 | 1.0000 | Tie |

### Panel B: TFT Models vs. Hist_Exp Baseline (Loss = MSE)

| Pasangan | Horizon | Regime | DM Stat | p-value (raw) | p-bonf (m=12) | p-bonf (m=24) | Kesimpulan |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| TFT_SPILL vs Hist_Exp | h=1 | Full Sample | -3.3040 | 9.5318e-04 | 1.1438e-02 | 2.2876e-02 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=1 | Full Sample | -3.1952 | 1.3976e-03 | 1.6771e-02 | 3.3542e-02 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=1 | Winsorize 99% | -7.7666 | 7.9936e-15 | 9.5923e-14 | 1.9185e-13 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=1 | Winsorize 99% | -8.2148 | 2.2204e-16 | 2.6645e-15 | 5.3291e-15 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=1 | Winsorize 97.5% | -8.6218 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=1 | Winsorize 97.5% | -9.0658 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=1 | Winsorize 95% | -10.3789 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=1 | Winsorize 95% | -10.8336 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=1 | Event Excl (Feb 5-6) | -7.9917 | 1.3323e-15 | 1.5987e-14 | 3.1974e-14 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=1 | Event Excl (Feb 5-6) | -8.6056 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=10 | Full Sample | -1.6516 | 9.8625e-02 | 1.0000e+00 | 1.0000e+00 | Tie |
| TFT_NOSPILL vs Hist_Exp | h=10 | Full Sample | -1.7612 | 7.8203e-02 | 9.3844e-01 | 1.0000e+00 | Tie |
| TFT_SPILL vs Hist_Exp | h=10 | Winsorize 99% | -4.9294 | 8.2470e-07 | 9.8964e-06 | 1.9793e-05 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=10 | Winsorize 99% | -4.1805 | 2.9081e-05 | 3.4897e-04 | 6.9795e-04 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=10 | Winsorize 97.5% | -5.5387 | 3.0471e-08 | 3.6565e-07 | 7.3130e-07 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=10 | Winsorize 97.5% | -4.5157 | 6.3103e-06 | 7.5724e-05 | 1.5145e-04 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=10 | Winsorize 95% | -6.5341 | 6.3984e-11 | 7.6780e-10 | 1.5356e-09 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=10 | Winsorize 95% | -5.0269 | 4.9852e-07 | 5.9822e-06 | 1.1964e-05 | TFT_NOSPILL |
| TFT_SPILL vs Hist_Exp | h=10 | Event Excl (Feb 5-6) | -5.4760 | 4.3508e-08 | 5.2210e-07 | 1.0442e-06 | TFT_SPILL |
| TFT_NOSPILL vs Hist_Exp | h=10 | Event Excl (Feb 5-6) | -4.5274 | 5.9702e-06 | 7.1642e-05 | 1.4328e-04 | TFT_NOSPILL |

---

## 5. Ringkasan Hari Ekstrem (Top 5)

| Rank y | Tanggal | y (Target $r^2$) | Return (%) | Pred HAR (h=1) | Pred TFT SPILL (h=1) | % Total SSE HAR | % Total SSE TFT SPILL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2026-02-05 | 2.205457e-02 | -14.85% | 7.434320e-04 | 4.247304e-04 | 60.67% | 58.47% |
| 2 | 2026-02-06 | 1.380062e-02 | +11.75% | 3.023858e-03 | 1.087070e-03 | 15.51% | 20.20% |
| 3 | 2026-08-21 | 4.265190e-03 | +6.53% | 1.099455e-03 | 3.889863e-04 | 1.34% | 1.88% |
| 4 | 2026-06-02 | 3.729880e-03 | -6.11% | 6.880357e-04 | 3.912533e-04 | 1.24% | 1.39% |
| 5 | 2026-03-04 | 3.265520e-03 | +5.71% | 8.618380e-04 | 3.611964e-04 | 0.77% | 1.05% |


---
*Laporan disusun secara otomatis dan objektif tanpa interpretasi berlebihan. Seluruh data diverifikasi dari artefak eksperimen v6.*
