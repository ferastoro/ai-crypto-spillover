# RSIC v6: Deep Learning Multi-Seed Replication & Light Hyperparameter Tuning

Direktori ini berisi notebook Kaggle dan dokumentasi metodologis untuk replikasi multi-seed (5 seed: `[42, 7, 13, 21, 99]`) dan tuning ringan model Deep Learning RSIC (v6) pada peramalan volatilitas Bitcoin (target $T0 = r^2$).

---

## 1. Parameter Pelatihan Stage B dan Stage C (A Priori Design)
Parameter pelatihan ditetapkan secara mutlak **SEBELUM** melihat hasil pengujian out-of-sample (test set 335 cutoff) dan **TIDAK DIUBAH** tanpa persetujuan:

| Tahap | Model Arsitektur | `max_steps` | `val_check_steps` | `early_stop_patience_steps` | Jendela Evaluasi ES | Jumlah Run Latihan |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Stage B** (Tuning Light) | **TFT** | `1000` | `25` | `5` | $5 \times 25 = 125$ steps | 8 grid $\times$ 2 seed = 16 runs |
| **Stage B** (Tuning Light) | **NBEATSx** | `1000` | `25` | `5` | $5 \times 25 = 125$ steps | 8 grid $\times$ 2 seed = 16 runs |
| **Stage C** (Main Multi-Seed) | **TFT** | `1000` | `25` | `5` | $5 \times 25 = 125$ steps | 2 config (SPILL/NOSPILL) $\times$ 5 seed = 10 runs |
| **Stage C** (Main Multi-Seed) | **NBEATSx** | `1000` | `25` | `5` | $5 \times 25 = 125$ steps | 2 config (SPILL/NOSPILL) $\times$ 5 seed = 10 runs |

> [!IMPORTANT]
> **Keputusan Metodologis A Priori**: Nilai `max_steps = 1000`, `val_check_steps = 25`, dan `early_stop_patience_steps = 5` identik di seluruh model dan kedua tahapan. Penetapan ini dilakukan sebelum eksekusi eksperimen untuk menjamin integritas ilmiah dan menghindari snooping bias.

---

## 2. Aturan Keras & Protokol Pre-Eksperimen
1. **Pre-Experimental Design & Pemilihan Loss**:
   - Seluruh desain, grid hyperparameter (termasuk dimensi loss MAE vs MSE), arsitektur model (TFT & NBEATSx), horizon ($h=1$ dan $h=10$), dan keluarga pengujian hipotesis (Bonferroni) ditetapkan **SEBELUM** melihat hasil empiris run v6.
   - **Pemilihan Dimensi Loss**: Pemilihan loss MAE vs MSE ditetapkan secara a priori pada validasi (167 observasi) SEBELUM melihat hasil test set (335 cutoff out-of-sample). Desain tidak diubah setelah hasil test keluar.
   - **Satuan Val Loss**: `ptl/val_loss` yang dipantau oleh early stopping dan `RestoreBestWeightsCallback` memakai satuan loss pelatihan masing-masing (MAE atau MSE).
2. **Tanpa Kebocoran Data (Strict Anti-Leakage Guard & Koreksi Batas Fit/Val/Test)**:
   - Evaluasi rolling out-of-sample pada test set memiliki 335 cutoff dengan horizon $h=10$. Maka rentang peramalan mencakup $335 + 10 - 1 = 344$ baris target unik (bukan 335).
   - Data pelatihan (`fit`) pada Stage C ditegaskan secara ketat hanya menggunakan baris 1 s.d. 1332 ($1676 - 344 = 1332$ baris, terdiri dari 1165 Train + 167 Val).
   - **Split Sebenarnya**: **1165 Train / 167 Val / 344 Baris Target (335 cutoff rolling)**, bukan 1174/167/335.
   - Tiga assertion anti-leakage wajib dieksekusi secara programatik:
     - (a) `max real_date data fit == tanggal cutoff pertama` (`2025-04-23`).
     - (b) `tanggal target pertama window pertama > max real_date data fit` (`2025-04-24 > 2025-04-23`).
     - (c) `cv_test['cutoff'].min() == max_ds_fit` (`2023-08-24`) dan `cv_test['ds'].min() == max_ds_fit + 1 hari` (`2023-08-25`).
3. **Pemberantasan Pelatihan Ganda (Single Fit per Seed/Config)**:
   - Model dilatih tepat **SATU KALI** per `(config, seed)` pada rentang pelatihan baris 1-1332.
   - Evaluasi rolling out-of-sample dijalankan murni secara inferensial melalui `NeuralForecast.cross_validation(..., use_fitted=True)`.
   - Integritas bobot diverifikasi secara matematis via checksum MD5 `state_dict`: checksum sebelum cross-validation harus identik sempurna dengan sesudah cross-validation (`chk_before == chk_after`).
4. **Preservasi Callback Bawaan & Early Stopping Aktual**:
   - Callback bawaan NeuralForecast (`EarlyStopping`) **TIDAK BOLEH DITIMPA** (menimpa list callback adalah bug v3/v4). Callback tambahan selalu di-append/extend ke `model.trainer_kwargs['callbacks']`.
   - Early stopping memonitor `ptl/val_loss` dengan parameter `val_check_steps=25` dan `early_stop_patience_steps=5` (jendela evaluasi 125 steps).
   - Didukung oleh callback in-memory `RestoreBestWeightsCallback(pl.Callback)` yang menyimpan klon `state_dict` di CPU saat validation loss mencapai nilai terbaik dan memulihkannya pada `on_train_end`.
   - **Abaikan Sanity Check / Step 0**: Evaluasi validasi sanity-check (saat `trainer.sanity_checking` aktif atau `global_step == 0`) diabaikan secara ketat agar bobot acak sebelum pelatihan tidak pernah menjadi `best_step`.
   - Telah diverifikasi secara matematis lokal pada CPU bahwa checksum bobot akhir model sama persis dengan checksum bobot di `best_step` (dengan `best_step > 0`).
   - Step penghentian aktual (`actual_final_step`), `best_step`, `best_val_loss`, dan status `es_triggered` dicatat ke `v6_es_log.csv`.
5. **Multi-Seed Replication**:
   - Dievaluasi pada 5 seed deterministik: `[42, 7, 13, 21, 99]`.
   - Seed disuntikkan secara komprehensif pada level global (`random`, `numpy`, `torch`, `pytorch_lightning.seed_everything`) dan level model (`random_seed`).
6. **Keluarga Bonferroni A Priori & Transparansi Dua Seri Hist_Exp**:
   - **Komparasi DL Ensemble vs 3 Baselines** (`Hist_Exp`, `Roll_20`, `HAR`): $m = 12$ per horizon ($4 \text{ konfigurasi} \times 3 \text{ baseline}$).
   - **Uji Ablasi Ensemble** (`SPILL` vs `NOSPILL`): $m = 4$ ($2 \text{ model} \times 2 \text{ horizon}$).
   - **Per-Seed Descriptive Summary**: Hanya menyajikan $p_{\text{raw}}$ dan rasio benih dengan $p_{\text{raw}} < 0.05$, tanpa klaim signifikansi statistik individual.
   - **Transparansi Seri Hist_Exp**:
     - Seri DL-base (MSE $h=1$: $2.689570 \times 10^{-6}$, $h=10$: $2.686277 \times 10^{-6}$) dipakai untuk evaluasi model DL.
     - Seri HAR-file (MSE $h=1$: $2.689342 \times 10^{-6}$, $h=10$: $2.686052 \times 10^{-6}$) diverifikasi pada reproduksi benchmark HAR v2.
     - **Penyebab Selisih**: Baris `2020-01-03` terbuang di `df_har` oleh `dropna()` setelah left-merge dengan file TVP (karena fitur TVP kausal baru dimulai pada `2020-01-06`).

---

## 3. Spesifikasi Grid Tuning Stage B (Maksimal 8 Kombinasi per Model)
Grid tuning Stage B ditetapkan a priori pada feature set `NOSPILL` (11 fitur) dengan mengevaluasi rata-rata MSE validasi pada $h=1$ dan $h=10$ lintas 2 seed (`[42, 7]`):
- **TFT (Temporal Fusion Transformer)**:
  - Tetap: `hidden_size = 64`
  - Variasi ($2 \times 2 \times 2 = 8$ kombinasi):
    - `input_size` $\in \{20, 50\}$
    - `learning_rate` $\in \{10^{-3}, 3 \times 10^{-4}\}$
    - `loss` $\in \{\text{MAE}, \text{MSE}\}$
- **NBEATSx (Neural Basis Expansion Analysis with Exogenous)**:
  - Tetap: `mlp_units = [[256, 256], [256, 256], [256, 256]]`
  - Variasi ($2 \times 2 \times 2 = 8$ kombinasi):
    - `input_size` $\in \{20, 50\}$
    - `learning_rate` $\in \{10^{-3}, 3 \times 10^{-4}\}$
    - `loss` $\in \{\text{MAE}, \text{MSE}\}$

Data Stage B menggunakan baris 1 s.d. 1332 (1165 Train + 167 Val). Evaluasi rolling validasi (158 window dengan $h=10$) berakhir persis di baris 1331 (`2025-04-23`), sama sekali tidak menyentuh target test window yang dimulai di baris 1332 (`2025-04-24`).

### Hasil Empiris Pemenang Stage B (Kaggle Run)
Dari eksekusi Stage B (16 runs pada `NOSPILL`, 2 seed: `[42, 7]`):
1. **Pemenang TFT**: `TFT_cfg6`
   - **Rata-rata MSE Validasi**: **$1.562859 \times 10^{-6}$**
   - **Hyperparameters Terpilih**:
     - `input_size`: `50`
     - `learning_rate`: `0.001` ($10^{-3}$)
     - `hidden_size`: `64`
     - `loss_name`: `'MSE'`
2. **Pemenang NBEATSx**: `NBEATSx_cfg3`
   - **Rata-rata MSE Validasi**: **$1.772739 \times 10^{-6}$**
   - **Hyperparameters Terpilih**:
     - `input_size`: `20`
     - `learning_rate`: `0.0003` ($3 \times 10^{-4}$)
     - `mlp_units`: `[[256, 256], [256, 256], [256, 256]]`
     - `loss_name`: `'MAE'`

> [!IMPORTANT]
> **Aturan Pre-Eksperimen**: Hyperparameter dan loss function terbaik yang terpilih dari `NOSPILL` ini langsung diterapkan untuk konfigurasi `SPILL` dan `NOSPILL` pada Stage C (5 seeds $\times$ 4 konfigurasi = 20 runs). Pemilihan ini murni ditentukan pada validation set **SEBELUM** melihat hasil empiris test set.

---

## 4. Bukti Verifikasi Callback & Sanity Guard (CPU Smoke Test)
### A. Preservasi Callback Bawaan EarlyStopping
Daftar callback pada objek model hasil fit:
`['EarlyStopping', 'RestoreBestWeightsCallback', 'ModelSummary']`
- `EarlyStopping` bawaan NeuralForecast tetap aktif di urutan pertama.
- `RestoreBestWeightsCallback` berhasil disuntikkan tanpa menimpa.

### B. Uji Early Stopping Berhenti Sebelum Batas Maksimal
Diuji dengan `max_steps=100`, `val_check_steps=5`, `early_stop_patience_steps=2`:
- Training berhenti secara aktual pada **step 15** (jauh sebelum step 100).
- `es_triggered` bernilai `True`.
- Riwayat validasi yang tercatat:
  - Step 5 : `val_loss = 0.469727` <-- BEST STEP
  - Step 10: `val_loss = 0.486093`
  - Step 15: `val_loss = 0.493172` (Patience habis, training dihentikan)

### C. Abaikan Sanity Check & Step 0
- Validasi sanity-check di step 0 (bobot acak) berhasil dilewati secara ketat.
- Tidak ada entri step 0 di riwayat validasi.
- `best_step = 5` ($> 0$).
- Bobot terbaik berhasil dimuat kembali pada `on_train_end` (`restored = True`).

---

## 5. Validasi Batas Fit/Val/Test & Assert Anti-Leakage
Berdasarkan audit `CV_Predictions_V4.csv`:
- Tanggal Cutoff Pertama (ds sintetis) : `2023-08-24`
- Tanggal Cutoff Terakhir (ds sintetis): `2024-07-23`
- Tanggal Cutoff Pertama (real_date)   : `2025-04-23`
- Tanggal Cutoff Terakhir (real_date)  : `2026-08-21`
- Tanggal Target Pertama (real_date)   : `2025-04-24`
- Tanggal Target Terakhir (real_date)  : `2026-09-04`
- Jumlah Cutoff: `335` | Jumlah Target Unik: `344` ($335 + 10 - 1 = 344$)

### Assert Anti-Leakage
1. **Assert (a)**: `max real_date data fit == tanggal cutoff pertama`
   - Max real_date fit: `2025-04-23`
   - Cutoff pertama: `2025-04-23`
   - Status: **PASS**
2. **Assert (b)**: `tanggal target pertama window pertama > max real_date data fit`
   - Target pertama window 1: `2025-04-24`
   - Max real_date fit: `2025-04-23`
   - Status: **PASS** (`2025-04-24 > 2025-04-23`)
3. **Assert (c)**: `cv_test['cutoff'].min() == max_ds_fit` dan `cv_test['ds'].min() == max_ds_fit + 1 hari`
   - `cv_test['cutoff'].min()` : `2023-08-24 00:00:00`
   - `cv_test['cutoff'].max()` : `2024-07-23 00:00:00`
   - `cv_test['cutoff'].nunique()` : `335`
   - `cv_test['ds'].min()` : `2023-08-25 00:00:00`
   - `cv_test['ds'].max()` : `2024-08-02 00:00:00`
   - `cv_test['ds'].nunique()` : `344`
   - `max_ds_fit` : `2023-08-24 00:00:00`
   - `max_ds_fit + 1 hari` : `2023-08-25 00:00:00`
   - Status: **PASS** (100% cocok sempurna)

---

## 6. Dokumentasi Dua Seri Hist_Exp pada Komparasi DM
1. **Seri DL-base (`Hist_Exp_DL`)**:
   - MSE $h=1$: $2.689570 \times 10^{-6}$
   - MSE $h=10$: $2.686277 \times 10^{-6}$
   - Dihitung dari `hist_series` (1677 baris, mulai `2020-01-03`) pada 335 cutoff origin via `v5_helpers.compute_baselines`.
   - Dipakai sebagai acuan komparasi model DL di `v6_metrics_per_seed.csv`, `v6_metrics_ensemble.csv`, dan `v6_dm_ensemble.csv`.
2. **Seri HAR-file (`Hist_Exp_HAR`)**:
   - MSE $h=1$: $2.689342 \times 10^{-6}$
   - MSE $h=10$: **$2.686052 \times 10^{-6}$** (terkoreksi dari penulisan sebelumnya)
   - Tersimpan di dalam file benchmark `har_preds_T0_h1_v2.csv` dan `har_preds_T0_h10_v2.csv`.
   - Dipakai pada Stage A untuk mereproduksi hasil uji Diebold-Mariano benchmark HAR v2 (`stat = -3.0651`, $p_{\text{raw}} = 0.0022$ pada $h=1$; `stat = -2.6871`, $p_{\text{raw}} = 0.0072$ pada $h=10$).
3. **Penyebab Selisih**:
   Baris `2020-01-03` terbuang di `df_har` oleh `dropna()` setelah left-merge dengan file TVP (karena fitur TVP kausal baru dimulai `2020-01-06`). Ini menyebabkan rentang historis pada file HAR memiliki selisih 1 baris awal dibanding `hist_series`, menghasilkan perbedaan kecil sebesar $\sim 0.008\%$ ($2.28 \times 10^{-10}$).

---

## 7. Kebutuhan Lingkungan Kaggle & Dataset Input
- **Akselerator**: GPU Kaggle (T4 x2 atau P100 disarankan).
- **Environment**: Python 3.10+, PyTorch 2.x, PyTorch Lightning 2.x.
- **Pinning Wajib**: `neuralforecast==3.3.0`, `arch`, `statsmodels`.
- **Dataset Input yang Harus Di-attach**:
  - Direktori input: `inputs/v5` (atau `./inputs/v5`).
  - Berkas terverifikasi MD5:
    1. `dataset_deep_learning.csv` (`ba64297407f9d8221488b4815a12ce8c`)
    2. `tvp_var_dynamic_features_causal.csv` (`066ad3cdf980cd187a79c25dc210f1b2`)
    3. `v5_helpers.py` (`bdd7bf54c3cca56f30a7a1cd779f65b1`)
    4. `har_preds_T0_h1_v2.csv` (`e340a7ba8effd60c8c6d7b85b073aded`)
    5. `har_preds_T0_h10_v2.csv` (`43baf149f5f60be52a80b962ab93e481`)
    6. `CV_Predictions_V4.csv` (`6251bfb6ce543ffcd746e5fab98c3a63`)
    7. `GARCH_Predictions_V4.csv` (`5e84fa827fd5cfd13bea445230daf2ee`)

---

## 8. Urutan Tahapan Eksekusi (Stages A - D)
Variabel kontrol `STAGE` diatur pada Cell 2:
- **`STAGE = "A"` (Audit CPU & Reproduksi Benchmark)**:
  - Memvalidasi hash MD5, audit CV4, split data (1165/167/344), benchmark HAR v2, baselines, uji DM, dan GARCH merge (selesai < 10 detik di CPU).
- **`STAGE = "B"` (Light Tuning Hyperparameter pada Train 1165 + Val 167 = 1332)**:
  - Mengevaluasi grid 8 kombinasi TFT & NBEATSx pada feature set `NOSPILL` (2 seed: `[42, 7]`).
  - Menghasilkan `v6_best_hyperparameters.json` dan `v6_tuning_log.csv`.
- **`STAGE = "C"` (Run Utama Multi-Seed 335 Cutoff, Fit 1332 Baris)**:
  - Memvalidasi assert anti-leakage (a), (b), (c).
  - Menjalankan smoke test 335 cutoff dan estimasi batas waktu 9 jam.
  - Melatih 5 seed $\times$ 4 konfigurasi pada 335 test cutoff dengan `use_fitted=True` dan `RestoreBestWeightsCallback`.
  - Menghasilkan 20 berkas `v6_partial_{model}_{fset}_T0_s{seed}.csv` (resumable) dan `v6_es_log.csv`.
- **`STAGE = "D"` (Agregasi, Ensemble & Uji Statistik CPU)**:
  - Menghasilkan tabel metrik per-seed (`v6_metrics_per_seed.csv`), ringkasan statistik benih (`v6_metrics_seed_summary.csv`), prediksi ensemble (`v6_ensemble_predictions.csv`), metrik ensemble (`v6_metrics_ensemble.csv`), uji DM Bonferroni (`v6_dm_ensemble.csv`), dan uji per-seed deskriptif (`v6_dm_per_seed.csv`, `v6_dm_per_seed_summary.csv`).
  - Mengompresi seluruh berkas luaran ke `v6_outputs.zip` beserta manifest MD5 `v6_manifest.csv`.
- **`STAGE = "ALL"`**:
  - Menjalankan seluruh tahapan A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ D secara berurutan.

---

## 9. Konfirmasi Manajemen Direktori Sementara
Perintah `rm -rf ./v6_outputs` **HANYA** menghapus direktori keluaran sementara dari run v6 itu sendiri. Perintah ini tidak memengaruhi atau menghapus direktori proyek existing mana pun (`results/`, `results/v4/`, `results/v5/`, `results/v5_sens/`, `inputs/v5/`, `data/clean/`, dll.).
