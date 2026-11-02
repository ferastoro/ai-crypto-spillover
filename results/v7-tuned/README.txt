================================================================================
RSIC Volatility Forecasting Audit (v7)
Loss-Matched Retraining & 60-Day Periodic Refitting
================================================================================
Execution Timestamp  : 2026-10-30 08:28:36 UTC
Active STAGE         : ALL
Hardware Device      : Tesla T4 (CUDA=True)
NeuralForecast Ver   : 3.3.0
Evaluated Seeds      : [42, 7, 13, 21, 99]
Test Cutoffs Target  : 335 cutoffs
Runtime Budget Limit : 10.0 hours
Total Elapsed Time   : 0.55 hours
Source Notebook Copy : NOT FOUND! Generated notebook_source_MISSING.txt

Actual Stage Elapsed Times:
  - Stage A               : Not Run
  - Stage A_TUNED         : 1238.14 seconds (20.64 minutes)
  - Stage B               : 700.43 seconds (11.67 minutes)
  - Stage POSTPROCESSING  : 2.89 seconds (0.05 minutes)

Hyperparameter Tuning Grid Specifications & Settings:
  - v7 NBEATSx Grid (12 combos) : input_size in {20, 50, 100} x learning_rate in {1e-3, 5e-4, 3e-4, 1e-4} | MSE loss only
  - v7 TFT Grid (8 combos)      : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4} | MAE loss only
  - Tuning Ranking Metric       : Mean validation MSE over seeds 42 and 7 (averaged over h=1 and h=10)
  
Early Stopping & Verification Settings:
  - NBEATSx Tuning & Winner : val_check_steps = 5 (v6 used 25) | patience = 25 checks (v6 used 5)
                              Total patience in steps = 125 steps (5 * 25 = 125, identical to v6: 25 * 5 = 125).
                              Restore-best weights: Enabled.
  - TFT Tuning & Winner     : val_check_steps = 25 | patience = 5 checks (125 steps). Restore-best: Enabled.

Grid Edge Verification:
  - NBEATSx  Winner Edge Status : ON THE EDGE (learning_rate=1e-4 (lower bound))
  - TFT      Winner Edge Status : ON THE EDGE ()

v6 Stage B Empirical Tuning Results for NBEATSx (from results/v6/v6_tuning_log.csv):
  +--------------------+------------+--------------+------------------+------------------+
  | Hyperparameters    | input_size | lr           | MAE Train ValMSE | MSE Train ValMSE |
  +--------------------+------------+--------------+------------------+------------------+
  | Combo 1 (cfg1/2)   | 20         | 0.0010       | 1.774417e-06     | 9.471862e-06     |
  | Combo 2 (cfg3/4)*  | 20         | 0.0003       | 1.772739e-06 (*) | 5.141590e-06     |
  | Combo 3 (cfg5/6)   | 50         | 0.0010       | 2.338505e-06     | 3.573679e-06     |
  | Combo 4 (cfg7/8)   | 50         | 0.0003       | 2.277654e-06     | 2.836472e-06     |
  +--------------------+------------+--------------+------------------+------------------+
  (*) Winner in v6 Stage B: NBEATSx_cfg3 (MAE loss, input 20, lr 0.0003).

Run Completion Summary:
  - Total Expected Runs : 44
  - Finished Runs Count : 44
  - Missing Runs Count  : 0

Finished Runs:
  + A_TUNED_NBEATSx_MSEtuned_SPILL_s42
  + A_TUNED_NBEATSx_MSEtuned_SPILL_s7
  + A_TUNED_NBEATSx_MSEtuned_SPILL_s13
  + A_TUNED_NBEATSx_MSEtuned_SPILL_s21
  + A_TUNED_NBEATSx_MSEtuned_SPILL_s99
  + A_TUNED_NBEATSx_MSEtuned_NOSPILL_s42
  + A_TUNED_NBEATSx_MSEtuned_NOSPILL_s7
  + A_TUNED_NBEATSx_MSEtuned_NOSPILL_s13
  + A_TUNED_NBEATSx_MSEtuned_NOSPILL_s21
  + A_TUNED_NBEATSx_MSEtuned_NOSPILL_s99
  + A_TUNED_TFT_MAEtuned_SPILL_s42
  + A_TUNED_TFT_MAEtuned_SPILL_s7
  + A_TUNED_TFT_MAEtuned_SPILL_s13
  + A_TUNED_TFT_MAEtuned_SPILL_s21
  + A_TUNED_TFT_MAEtuned_SPILL_s99
  + A_TUNED_TFT_MAEtuned_NOSPILL_s42
  + A_TUNED_TFT_MAEtuned_NOSPILL_s7
  + A_TUNED_TFT_MAEtuned_NOSPILL_s13
  + A_TUNED_TFT_MAEtuned_NOSPILL_s21
  + A_TUNED_TFT_MAEtuned_NOSPILL_s99
  + B_TFT_MSE_SPILL_blk1
  + B_TFT_MSE_SPILL_blk2
  + B_TFT_MSE_SPILL_blk3
  + B_TFT_MSE_SPILL_blk4
  + B_TFT_MSE_SPILL_blk5
  + B_TFT_MSE_SPILL_blk6
  + B_TFT_MSE_NOSPILL_blk1
  + B_TFT_MSE_NOSPILL_blk2
  + B_TFT_MSE_NOSPILL_blk3
  + B_TFT_MSE_NOSPILL_blk4
  + B_TFT_MSE_NOSPILL_blk5
  + B_TFT_MSE_NOSPILL_blk6
  + B_NBEATSx_MAE_SPILL_blk1
  + B_NBEATSx_MAE_SPILL_blk2
  + B_NBEATSx_MAE_SPILL_blk3
  + B_NBEATSx_MAE_SPILL_blk4
  + B_NBEATSx_MAE_SPILL_blk5
  + B_NBEATSx_MAE_SPILL_blk6
  + B_NBEATSx_MAE_NOSPILL_blk1
  + B_NBEATSx_MAE_NOSPILL_blk2
  + B_NBEATSx_MAE_NOSPILL_blk3
  + B_NBEATSx_MAE_NOSPILL_blk4
  + B_NBEATSx_MAE_NOSPILL_blk5
  + B_NBEATSx_MAE_NOSPILL_blk6

Missing Runs:
  (None - All expected runs completed successfully)

Measured Fit Timing:
  - Average Fit Time : 30.49 seconds per fit

================================================================================
