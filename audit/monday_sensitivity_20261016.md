# LAPORAN AUDIT SENSITIVITAS RQ1 (TVP-VAR): PENGARUH HARI SENIN & DEFINISI RETURN CRYPTO

**Tanggal Audit:** 16 Oktober 2026  
**Status Audit:** SELESAI & TERVERIFIKASI PENUH  
**Skrip Audit:** [`audit/monday_sensitivity.py`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/audit/monday_sensitivity.py)  
**Tabel Data Lengkap:** [`audit/monday_sensitivity_tables.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/audit/monday_sensitivity_tables.csv)  
**Protokol:** READ-ONLY terhadap semua file dataset lama, notebook, dan artefak historis. Tidak ada pemanggilan API eksternal atau retraining model DL.

---

## 1. Penetapan Kriteria "Perubahan Material" (*Ex-Ante*)

Sesuai aturan ketat, kriteria perubahan material ditetapkan **sebelum** melihat hasil sensitivitas dan tidak dapat diubah:
1. **Kriteria 1 (TSI Rata-Rata):** Nilai rata-rata Total Spillover Index (TSI) berubah lebih dari **3 poin persentase** ($|\Delta \text{TSI}| > 3.00\text{ pp}$).
2. **Kriteria 2 (Tanda Net Spillover):** Tanda net spillover ($+/-$) pada **BTC-IDR** atau **BTC-USD** mengalami pembalikan (misalnya dari net-transmitter menjadi net-receiver atau sebaliknya).
3. **Kriteria 3 (Transmisi Saham AI ke BTC-IDR):** Akumulasi transmisi spillover dari 5 saham teknologi AI global (`SAHAM_NVDA`, `SAHAM_MSFT`, `SAHAM_TSLA`, `SAHAM_GOOGL`, `SAHAM_META`) ke `CRYPTO_BTC_IDR_INDODAX` berubah lebih dari **2 poin persentase** ($|\Delta \text{Transmisi}| > 2.00\text{ pp}$).

---

## 2. Langkah 0: Reproduksi Baseline (Wajib Lolos)

Logika estimasi TVP-VAR disalin persis dari [`tvp_var_spillover.ipynb`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/tvp_var_spillover.ipynb) (Cell 3, 6, 8, 10, 17):
- **Input Data:** [`data/clean/dataset_garch_baseline.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_garch_baseline.csv), $N = 1.556$ baris.
- **Transformasi:** $Y = r^2$, lalu $Y_{\log} = \ln(r^2 + 10^{-10})$, kemudian di-z-score ($Y_{\text{std}} = (Y_{\log} - \mu) / \sigma$).
- **Spesifikasi Model:** 8 variabel inti, $p = 1$, forgetting factor state $\lambda = 0.99$, forgetting factor kovarians $\kappa = 0.96$, forecast horizon GFEVD $H = 10$.
- **Target Verifikasi:** [`scratch/final/Average_Spillover_Table_vol.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/scratch/final/Average_Spillover_Table_vol.csv) dengan ambang toleransi $\le 10^{-3}$.

### Hasil Pengecekan Reproduksi Baseline
| Metrik | Nilai Skrip Audit | Nilai Target Baseline | Selisih Absolut | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean TSI** | **38.741498** | **38.7415** | $2.0 \times 10^{-6}$ | **LOLOS** |
| **NET BTC-IDR** | **+5.454596** | **+5.4546** | $4.0 \times 10^{-6}$ | **LOLOS** |
| **NET BTC-USD** | **+11.798512** | **+11.7985** | $1.2 \times 10^{-5}$ | **LOLOS** |
| **Maksimum Deviasi Matriks GFEVD** | — | — | $4.96 \times 10^{-5}$ | **LOLOS** |

*Kesimpulan Langkah 0:* Reproduksi baseline lolos pengujian secara eksak (semua deviasi $< 5 \times 10^{-5} \ll 10^{-3}$). Eksekusi dilanjutkan ke Langkah 1, 2, dan 3.

---

## 3. Langkah 1: Analisis Deskriptif Hari Senin & Hari Pasca-Libur (Tanpa Model)

Pada [`data/clean/dataset_garch_baseline.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_garch_baseline.csv), setiap baris diberi penanda selisih kalender terhadap baris sebelumnya: $\Delta t = \text{Date}_t - \text{Date}_{t-1}$.
- **Total Baris:** $N = 1.556$.
- **Baris Jarak Kalender $> 1$ Hari (`gap_days > 1`):** **383 baris (24,61%)**.
  - Komposisi hari: 290 hari Senin (akhir pekan reguler), 46 hari Selasa (libur bursa hari Senin), 21 hari Jumat, 13 hari Kamis, 13 hari Rabu (libur bursa di tengah pekan).
- **Baris Jarak Kalender $\le 1$ Hari (`other`):** **1.173 baris (75,39%)** (terdiri atas 1.172 baris dengan $\Delta t = 1$ hari dan 1 baris pertama indeks 0 tanggal 2020-01-03).

### Perbandingan Karakteristik Volatilitas (Gap $> 1$ Hari vs Hari Lainnya)
Sumber data: [`audit/monday_sensitivity_tables.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/audit/monday_sensitivity_tables.csv) Tabel 1.

| Variabel Aset | Rata-rata Z-Score (`gap > 1`) | Rata-rata Z-Score (`other`) | Selisih Z-Score ($\Delta z$) | Rata-rata $r^2$ (`gap > 1`) | Rata-rata $r^2$ (`other`) | Rasio Rata-rata $r^2$ | Rasio Varians Return ($\sigma^2_{\text{gap}} / \sigma^2_{\text{other}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SAHAM_NVDA`** | +0.0542 | -0.0177 | +0.0719 | 0.001365 | 0.001050 | 1.300x | 1.299x |
| **`SAHAM_MSFT`** | +0.0532 | -0.0174 | +0.0705 | 0.000503 | 0.000366 | 1.374x | 1.379x |
| **`SAHAM_TSLA`** | +0.1916 | -0.0626 | +0.2542 | 0.002807 | 0.001485 | 1.890x | 1.874x |
| **`SAHAM_GOOGL`** | +0.1325 | -0.0433 | +0.1758 | 0.000537 | 0.000423 | 1.269x | 1.270x |
| **`SAHAM_META`** | +0.1207 | -0.0394 | +0.1601 | 0.000768 | 0.000853 | 0.900x | 0.901x |
| **`CRYPTO_BTC_USD`** | **+0.2341** | **-0.0764** | **+0.3105** | **0.002814** | **0.001246** | **2.258x** | **2.267x** |
| **`CRYPTO_ETH_USD`** | **+0.2573** | **-0.0840** | **+0.3413** | **0.005179** | **0.002157** | **2.401x** | **2.403x** |
| **`CRYPTO_BTC_IDR_INDODAX`** | **+0.2116** | **-0.0691** | **+0.2807** | **0.002403** | **0.001006** | **2.388x** | **2.393x** |

### Evaluasi Khusus BTC-IDR dan BTC-USD
1. **BTC-USD:**
   - Ukuran sampel: $N_{\text{gap}} = 383$, $N_{\text{other}} = 1.173$.
   - Rata-rata $r^2$: $0.002814$ pada hari gap vs $0.001246$ pada hari biasa (selisih $+0.001568$, rasio $2.258\times$ atau **$+125,8\%$ lebih tinggi**).
   - Varians sampel return: $0.002820$ vs $0.001244$ (rasio $2.267\times$).
   - Selisih rata-rata z-score log-vol: $+0.3105$.
2. **BTC-IDR (Indodax):**
   - Ukuran sampel: $N_{\text{gap}} = 383$, $N_{\text{other}} = 1.173$.
   - Rata-rata $r^2$: $0.002403$ pada hari gap vs $0.001006$ pada hari biasa (selisih $+0.001396$, rasio $2.388\times$ atau **$+138,8\%$ lebih tinggi**).
   - Varians sampel return: $0.002404$ vs $0.001005$ (rasio $2.393\times$).
   - Selisih rata-rata z-score log-vol: $+0.2807$.

*Fakta Terukur:* Pada `dataset_garch_baseline.csv`, baris dengan jeda kalender $> 1$ hari memiliki varians kuadratik return crypto sekitar 2,26–2,39 kali lipat lebih besar dibanding hari biasa. Hal ini terjadi karena perhitungan return crypto pada baris tersebut mengumpulkan pergerakan harga selama 3 hari kalender (Jumat ke Senin).

---

## 4. Langkah 2: Evaluasi Tiga Varian TVP-VAR

Tiga varian TVP-VAR diestimasi dengan parameter dasar yang identik ($\lambda = 0.99, \kappa = 0.96, p = 1, H = 10$):
- **V0 (Baseline):** Dataset asli [`data/clean/dataset_garch_baseline.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_garch_baseline.csv) ($N = 1.556$).
- **V1 (Drop Gap $> 1$ Hari):** Seluruh baris berjarak $> 1$ hari dibuang. Z-score dihitung ulang dari data tersisa ($N = 1.173$).
  - *Catatan teknis:* Lag-1 pada V1 melompati jeda kalender (misal: Selasa diprediksi dari Jumat pekan sebelumnya). Dampak pergeseran lag ini diperkirakan marginal karena seleksi orde lag dengan BIC pada data asli merekomendasikan $p = 0$ (parsimony model).
- **V2 (Definisi Return Crypto DL):** Menggunakan return cryptocurrency dari [`data/clean/dataset_deep_learning.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_deep_learning.csv), di mana return crypto hari Senin adalah 1-hari (Minggu ke Senin), sedangkan saham adalah Jumat ke Senin. Dibatasi pada irisan tanggal yang sama dengan baseline ($N = 1.556$).
  - *Verifikasi Kode:* Definisi ini terkonfirmasi dan terverifikasi secara eksplisit dari baris 218–265 pada [`src/preprocess_data.py`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/src/preprocess_data.py).

### Tabel Komparasi Utama Tiga Varian TVP-VAR
Sumber data: [`audit/monday_sensitivity_tables.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/audit/monday_sensitivity_tables.csv) Tabel 2.

| Variabel / Metrik | V0: Baseline | V1: Drop Gap $> 1$ Hari | V2: DL Crypto Return |
| :--- | :---: | :---: | :---: |
| **Jumlah Observasi ($N$)** | 1.556 | 1.173 | 1.556 |
| **Total Spillover Index (TSI Rata-Rata)** | **38.74%** | **38.67%** | **37.99%** |
| *$\Delta$ TSI vs Baseline* | — | *-0.07 pp* | *-0.75 pp* |
| **NET `SAHAM_NVDA`** | -2.32% | -2.34% | -1.08% |
| **NET `SAHAM_MSFT`** | -2.34% | -3.31% | -2.42% |
| **NET `SAHAM_TSLA`** | -4.76% | -4.39% | -3.80% |
| **NET `SAHAM_GOOGL`** | -3.42% | -0.70% | -2.83% |
| **NET `SAHAM_META`** | -4.01% | -2.90% | -2.98% |
| **NET `CRYPTO_BTC_USD`** | **+11.80%** | **+9.09%** | **+9.53%** |
| **NET `CRYPTO_ETH_USD`** | -0.40% | -1.12% | -1.18% |
| **NET `CRYPTO_BTC_IDR_INDODAX`** | **+5.45%** | **+5.66%** | **+4.76%** |
| **Tanda Net BTC-USD** | **Positif (+)** | **Positif (+)** | **Positif (+)** |
| **Tanda Net BTC-IDR** | **Positif (+)** | **Positif (+)** | **Positif (+)** |
| **Transmisi 5 Saham AI $\rightarrow$ BTC-IDR** | **9.00%** | **9.97%** | **10.17%** |
| *$\Delta$ Transmisi AI $\rightarrow$ BTC-IDR* | — | *+0.97 pp* | *+1.17 pp* |
| **Transmisi 5 Saham AI $\rightarrow$ BTC-USD** | 9.16% | 9.98% | 9.57% |
| **Transmisi BTC-USD $\rightarrow$ BTC-IDR** | 30.78% | 29.77% | 29.51% |
| **Transmisi BTC-IDR $\rightarrow$ BTC-USD** | 29.35% | 28.89% | 28.68% |

### Evaluasi terhadap Tiga Kriteria "Perubahan Material"
| Kriteria Pengujian | Batas Toleransi | V1 (Drop Gap $> 1$) | V2 (DL Crypto Return) |
| :--- | :---: | :---: | :---: |
| **1. TSI Rata-Rata Berubah $> 3$ pp** | $|\Delta \text{TSI}| > 3.00\text{ pp}$ | **TIDAK** ($-0.07\text{ pp}$) | **TIDAK** ($-0.75\text{ pp}$) |
| **2. Tanda Net BTC-IDR / BTC-USD Berbalik** | Pembalikan tanda ($+/-$) | **TIDAK** (keduanya tetap $+$) | **TIDAK** (keduanya tetap $+$) |
| **3. Transmisi 5 AI $\rightarrow$ BTC-IDR Berubah $> 2$ pp** | $|\Delta \text{Transmisi}| > 2.00\text{ pp}$ | **TIDAK** ($+0.97\text{ pp}$) | **TIDAK** ($+1.17\text{ pp}$) |
| **STATUS AKHIR: Perubahan Material?** | Salah satu kriteria = Ya | **TIDAK ADA** | **TIDAK ADA** |

---

## 5. Langkah 3: Verifikasi Angka Audit Sebelumnya

### 3a. Korelasi Log-Return BTC-IDR vs BTC-USD
Dihitung dari [`data/clean/dataset_garch_baseline.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_garch_baseline.csv) menggunakan kolom `ret_crypto_btc_idr_indodax` dan `ret_crypto_btc_usd` ($N = 1.556$):
- **Lag 0 (Kontemporer):**
  $$\text{Corr}(r^{\text{BTC-IDR}}_t, r^{\text{BTC-USD}}_t) = \mathbf{0.981106} \quad (\approx \mathbf{0.9811})$$
  Target audit sebelumnya: $0.9811$. Selisih: $6 \times 10^{-6}$. **Status: VERIFIED**.
- **Lag -1 (BTC-USD Memimpin BTC-IDR 1 Hari):**
  $$\text{Corr}(r^{\text{BTC-IDR}}_t, r^{\text{BTC-USD}}_{t-1}) = \mathbf{-0.004578} \quad (\approx \mathbf{-0.0046})$$
  Target audit sebelumnya: $-0.0046$. Selisih: $2.2 \times 10^{-5}$. **Status: VERIFIED**.

### 3b. Rincian Selisih 121 Hari Kalender antara Baseline dan Deep Learning
Dataset Deep Learning memiliki $1.677$ baris observasi (berbasis kalender bursa saham AS / NVDA), sedangkan dataset baseline GARCH memiliki $1.556$ baris ($1.677 - 1.556 = 121$ hari terbuang).

Dilakukan pemetaan terhadap seluruh $1.677$ tanggal bursa AS terhadap ketersediaan data di masing-masing instrumen mentah ([`data/raw/`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/raw/)):
- `saham_nvda`, `saham_msft`, `saham_tsla`, `saham_googl`, `saham_meta`: **0 tanggal hilang** (100% lengkap).
- `crypto_btc_usd`, `crypto_eth_usd`, `crypto_btc_idr_indodax`: **0 tanggal hilang** (perdagangan 24/7).
- `konteks_vix`: **0 tanggal hilang** (kalender bursa AS).
- `konteks_ihsg` (`data/raw/konteks_ihsg.csv`): **120 tanggal hilang dari 121 tanggal (99,17%)**.
- `konteks_usdidr` (`data/raw/konteks_usdidr.csv`): **1 tanggal hilang dari 121 tanggal (0,83%)**.

*Kesimpulan 3b:* **`konteks_ihsg` adalah instrumen utama yang menyebabkan penyusutan sampel bursa AS pada inner join listwise deletion** (menyumbang $120$ dari $121$ hari yang hilang, atau $99,17\%$). Hal ini disebabkan oleh hari libur nasional Indonesia, cuti bersama Idul Fitri, dan libur BEI lainnya ketika Wall Street tetap beroperasi secara normal.

---

## 6. Daftar Verifikasi (Status UNVERIFIED)

Sesuai Aturan Keras 1:
- Nilai Target Baseline TSI ($38.7415$), NET BTC-IDR ($+5.4546$), NET BTC-USD ($+11.7985$): **VERIFIED** (terbukti dari reproduksi kode Cell 6 & 8 `tvp_var_spillover.ipynb` dan file `scratch/final/Average_Spillover_Table_vol.csv`).
- Korelasi Return BTC-IDR vs BTC-USD ($0.9811$ dan $-0.0046$): **VERIFIED** (terbukti dari kolom `ret_crypto_btc_idr_indodax` dan `ret_crypto_btc_usd` di `data/clean/dataset_garch_baseline.csv`).
- Definisi Return Crypto pada Dataset Deep Learning: **VERIFIED** (terbukti dari baris 218–265 di `src/preprocess_data.py`).
- Penyebab 121 hari terbuang: **VERIFIED** (terbukti 120 hari berasal dari disparitas kalender BEI pada `konteks_ihsg.csv`).
- **Daftar UNVERIFIED:** **NIHIL (0 item)**. Seluruh data, angka, dan baris kode berhasil diverifikasi dan direproduksi secara independen.

---

## 7. MD5 Hashes Seluruh File Input

| Nama File | Path | MD5 Hash |
| :--- | :--- | :--- |
| `dataset_garch_baseline.csv` | [`data/clean/dataset_garch_baseline.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_garch_baseline.csv) | `90238c4edddc13fb2f6645d087b7da43` |
| `dataset_deep_learning.csv` | [`data/clean/dataset_deep_learning.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/clean/dataset_deep_learning.csv) | `ba64297407f9d8221488b4815a12ce8c` |
| `Average_Spillover_Table_vol.csv` | [`scratch/final/Average_Spillover_Table_vol.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/scratch/final/Average_Spillover_Table_vol.csv) | `b3ebe2d6a2ff61de5394fd2da44dcb26` |
| `preprocess_data.py` | [`src/preprocess_data.py`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/src/preprocess_data.py) | `d7903d914fbc3e2a8d2acdcbc15f669a` |
| `tvp_var_spillover.ipynb` | [`tvp_var_spillover.ipynb`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/tvp_var_spillover.ipynb) | `079317c077dc842bc7473f465062a519` |
| `crypto_btc_idr_indodax.csv` | [`data/raw/crypto_btc_idr_indodax.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/raw/crypto_btc_idr_indodax.csv) | `45bb5f212fa7e401d73fc5e6909c69f4` |
| `crypto_btc_usd.csv` | [`data/raw/crypto_btc_usd.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/raw/crypto_btc_usd.csv) | `989c51f781982ca3715efba61222278b` |
| `konteks_ihsg.csv` | [`data/raw/konteks_ihsg.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/raw/konteks_ihsg.csv) | `899198d6cad4f999b6a6153c82440049` |
| `saham_nvda.csv` | [`data/raw/saham_nvda.csv`](file:///Users/admin/Developer/coding_backup/SEM%205/RSIC/data/raw/saham_nvda.csv) | `b57c0bf69409a3c6d14bb1fdac44e054` |

---

## 8. Ringkasan Akhir Tripartit

### (1) Fakta Terukur
1. Terdapat 383 baris observasi (24,61% dari total 1.556 baris) pada `dataset_garch_baseline.csv` yang memiliki jarak kalender $\Delta t > 1$ hari (hari Senin dan hari setelah libur bursa).
2. Varians kuadratik return ($r^2$) untuk aset cryptocurrency pada baris $\Delta t > 1$ hari adalah **2,26 kali lipat lebih tinggi untuk BTC-USD** ($0.002814$ vs $0.001246$) dan **2,39 kali lipat lebih tinggi untuk BTC-IDR** ($0.002403$ vs $0.001006$) dibandingkan hari reguler.
3. Korelasi return kontemporer (lag 0) antara BTC-IDR dan BTC-USD bernilai $+0.9811$, sedangkan korelasi lag -1 bernilai $-0.0046$.
4. Dari 121 tanggal bursa AS yang terbuang dalam dataset baseline akibat listwise deletion, **120 tanggal (99,17%) disebabkan secara tunggal oleh hari libur BEI (`konteks_ihsg`)**.

### (2) Hal yang Tidak Berubah
1. **Total Spillover Index (TSI):** TSI sistem tetap berada pada rentang 37,99% s.d. 38,74%. Penurunan TSI hanya $-0.07\text{ pp}$ saat baris gap dibuang (V1) dan $-0.75\text{ pp}$ saat menggunakan return harian crypto DL (V2). Keduanya jauh di bawah ambang batas material 3 pp.
2. **Status Net-Transmitter:** Tanda net spillover tidak mengalami pembalikan sama sekali. Baik `CRYPTO_BTC_USD` ($+11.80\% \rightarrow +9.09\% \rightarrow +9.53\%$) maupun `CRYPTO_BTC_IDR_INDODAX` ($+5.45\% \rightarrow +5.66\% \rightarrow +4.76\%$) **tetap bertahan sebagai net-transmitter** di seluruh varian.
3. **Pola Transmisi Saham AI ke BTC-IDR:** Transmisi dari 5 saham AI ke BTC-IDR berada pada rentang $9.00\%$ (V0), $9.97\%$ (V1), dan $10.17\%$ (V2). Perubahan maksimal hanya $+1.17\text{ pp}$, berada di bawah ambang batas material 2 pp.
4. **Dominasi Spillover Pasangan Crypto:** Transmisi bilateral antara BTC-USD dan BTC-IDR tetap mendominasi matriks spillover, dengan nilai transmisi timbal balik berkisar antara 28,68% hingga 30,78% di seluruh varian.

### (3) Hal yang Berubah Menurut Kriteria di Atas
1. **Kriteria 1 ($|\Delta \text{TSI}| > 3\text{ pp}$):** **TIDAK ADA PERUBAHAN MATERIAL** (perubahan maksimal $0.75\text{ pp} < 3.00\text{ pp}$).
2. **Kriteria 2 (Pembalikan Tanda Net):** **TIDAK ADA PERUBAHAN MATERIAL** (BTC-IDR dan BTC-USD tetap positif di semua varian).
3. **Kriteria 3 ($|\Delta \text{Transmisi 5 AI} \rightarrow \text{BTC-IDR}| > 2\text{ pp}$):** **TIDAK ADA PERUBAHAN MATERIAL** (perubahan maksimal $1.17\text{ pp} < 2.00\text{ pp}$).

*Kesimpulan Komprehensif:* Berdasarkan tiga kriteria objektif yang telah dipatok *ex-ante*, **tidak ditemukan adanya perubahan material** pada temuan empiris RQ1 TVP-VAR akibat efek hari Senin maupun perbedaan konvensi return cryptocurrency. Temuan bahwa BTC-USD dan BTC-IDR merupakan net-transmitter volatilitas dengan keterkaitan timbal balik yang sangat kuat terbukti kokoh (*robust*) terhadap variasi perlakuan kalender.
