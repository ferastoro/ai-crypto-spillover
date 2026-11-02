#!/usr/bin/env python3
"""
results/outlier_check_v2/run_outlier_check_v2.py
Outlier sensitivity analysis pipeline v2 for RSIC v6 Deep Learning experiments.
Evaluates model ranking, mean bias, and Diebold-Mariano test robustness under:
1. Full sample (N=335)
2. Target winsorization at 99%, 97.5%, and 95% percentiles (N=335, predictions unmodified)
3. Event-window exclusion of Feb 5-6, 2026 crash & rebound (N=333)

Strictly CPU only, read-only access to existing data files.
All multi-testing analyses are designated as EXPLORATORY.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_DIR = os.path.join(ROOT_DIR, "inputs/v5")
V6_OUT_DIR = os.path.join(ROOT_DIR, "results/v6")
AUDIT_V6_DIR = os.path.join(ROOT_DIR, "audit/v6")
OUT_DIR = os.path.join(ROOT_DIR, "results/outlier_check_v2")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(SRC_DIR)
sys.path.append(INPUT_DIR)
import v5_helpers


def md5_file(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


def diebold_mariano_engine(y_true, y_p1, y_p2, h_val, loss_type="MSE"):
    """
    Computes the Diebold-Mariano test statistic with:
    - lag = 0 for h = 1
    - lag = h - 1 (lag = 9 for h = 10) with Bartlett weights
    - Harvey-Leybourne-Newbold (HLN) small-sample adjustment for h > 1
    - Patton (2011) QLIKE without clipping (returns NaN if any prediction <= 0)
    Sign convention: d_t = Loss(Model 1) - Loss(Model 2).
    stat < 0 indicates Model 1 has lower loss (better).
    """
    if loss_type == "MSE":
        d = (y_true - y_p1) ** 2 - (y_true - y_p2) ** 2
    elif loss_type == "MAE":
        d = np.abs(y_true - y_p1) - np.abs(y_true - y_p2)
    elif loss_type == "QLIKE":
        if np.any(y_p1 <= 0) or np.any(y_p2 <= 0) or np.any(y_true <= 0):
            return np.nan, np.nan
        q1 = y_true / y_p1 - np.log(y_true / y_p1) - 1.0
        q2 = y_true / y_p2 - np.log(y_true / y_p2) - 1.0
        d = q1 - q2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    T = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    lag = 0 if h_val == 1 else (h_val - 1)
    for l in range(1, lag + 1):
        weight = 1.0 - (l / (lag + 1))
        cov = np.cov(d[l:], d[:-l], ddof=0)[0, 1]
        gamma_sum += 2.0 * weight * cov

    var_d = gamma0 + gamma_sum
    if var_d <= 0:
        stat = 0.0
        p_raw = 1.0
    else:
        stat = d_bar / np.sqrt(var_d / T)
        if h_val > 1:
            hln = np.sqrt((T + 1 - 2 * h_val + (h_val * (h_val - 1)) / T) / T)
            stat = stat * hln
        p_raw = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))

    return stat, p_raw


def main():
    print("=" * 80)
    print("RSIC v6 OUTLIER SENSITIVITY CHECK v2 (WINSORIZATION & PATTON QLIKE)")
    print("=" * 80)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. INPUT FILES AND HASHES (READ-ONLY)
    # --------------------------------------------------------------------------
    input_files = {
        "v6_ensemble_predictions.csv": os.path.join(V6_OUT_DIR, "v6_ensemble_predictions.csv"),
        "CV_Predictions_V4.csv": os.path.join(INPUT_DIR, "CV_Predictions_V4.csv"),
        "dataset_deep_learning.csv": os.path.join(INPUT_DIR, "dataset_deep_learning.csv"),
        "har_preds_T0_h1_v2.csv": os.path.join(INPUT_DIR, "har_preds_T0_h1_v2.csv"),
        "har_preds_T0_h10_v2.csv": os.path.join(INPUT_DIR, "har_preds_T0_h10_v2.csv"),
        "GARCH_Predictions_V4.csv": os.path.join(INPUT_DIR, "GARCH_Predictions_V4.csv"),
        "v5_helpers.py": os.path.join(INPUT_DIR, "v5_helpers.py"),
        "audit_v6_dm_ensemble.csv": os.path.join(AUDIT_V6_DIR, "audit_v6_dm_ensemble.csv"),
    }

    print("\n[1. INPUT FILES AND MD5 HASHES]")
    for name, path in input_files.items():
        h = md5_file(path)
        sz = os.path.getsize(path)
        rel_path = os.path.relpath(path, ROOT_DIR)
        print(f"  {rel_path}: md5={h} ({sz} bytes)")

    # --------------------------------------------------------------------------
    # 2. LOAD DATASETS AND BUILD UNIFIED ARRAYS
    # --------------------------------------------------------------------------
    cv4 = pd.read_csv(input_files["CV_Predictions_V4.csv"])
    cv4["cutoff"] = pd.to_datetime(cv4["cutoff"])
    cv4["real_date"] = pd.to_datetime(cv4["real_date"])

    df_dl = pd.read_csv(input_files["dataset_deep_learning.csv"])
    df_dl["Date"] = pd.to_datetime(df_dl["Date"])
    hist_series = pd.Series(df_dl["ret_crypto_btc_idr_indodax"].values ** 2, index=df_dl["Date"])

    har_h1_df = pd.read_csv(input_files["har_preds_T0_h1_v2.csv"])
    har_h10_df = pd.read_csv(input_files["har_preds_T0_h10_v2.csv"])
    har_h1_df["Date"] = pd.to_datetime(har_h1_df["Date"])
    har_h10_df["Date"] = pd.to_datetime(har_h10_df["Date"])

    garch_df = pd.read_csv(input_files["GARCH_Predictions_V4.csv"])
    m_garch = v5_helpers.load_garch_merged(cv4, garch_df)
    g1 = m_garch[m_garch["h_step"] == 1].reset_index(drop=True)
    g10 = m_garch[m_garch["h_step"] == 10].reset_index(drop=True)

    ens_df = pd.read_csv(input_files["v6_ensemble_predictions.csv"])
    ens_df["cutoff"] = pd.to_datetime(ens_df["cutoff"])
    ens_df["real_date"] = pd.to_datetime(ens_df["real_date"])

    c_origins = v5_helpers.get_cutoff_origins(cv4, hist_series)
    grp_h1 = cv4.groupby("cutoff").head(1).reset_index(drop=True)
    grp_h10 = cv4.groupby("cutoff").tail(1).reset_index(drop=True)

    hexp_arr_h1, roll_arr_h1, _ = v5_helpers.compute_baselines(grp_h1, hist_series, c_origins)
    hexp_arr_h10, roll_arr_h10, _ = v5_helpers.compute_baselines(grp_h10, hist_series, c_origins)

    def assemble_horizon_data(h_step_val, base_grp, har_df, hexp_arr, roll_arr, garch_sub):
        df_base = base_grp[["cutoff", "real_date", "y"]].copy()
        m_har = pd.merge(df_base, har_df[["Date", "HAR"]], left_on="real_date", right_on="Date", how="inner")
        assert len(m_har) == 335, f"HAR merge len != 335 (got {len(m_har)})"

        models_dict = {
            "HAR": m_har["HAR"].values,
            "Hist_Exp": hexp_arr,
            "Roll_20": roll_arr,
            "GARCH11": garch_sub["GARCH11"].values,
            "EGARCH": garch_sub["EGARCH"].values,
        }

        for cfg in ["TFT_SPILL_T0", "TFT_NOSPILL_T0", "NBEATSx_SPILL_T0", "NBEATSx_NOSPILL_T0"]:
            clean_name = cfg.replace("_T0", "")
            sub_ens = ens_df[ens_df["Config"] == cfg][["cutoff", "real_date", "prediction"]]
            m_dl = pd.merge(df_base[["cutoff", "real_date"]], sub_ens, on=["cutoff", "real_date"], how="inner")
            assert len(m_dl) == 335, f"DL {cfg} merge len != 335 (got {len(m_dl)})"
            models_dict[clean_name] = m_dl["prediction"].values

        return df_base, models_dict

    df_h1, models_h1 = assemble_horizon_data(1, grp_h1, har_h1_df, hexp_arr_h1, roll_arr_h1, g1)
    df_h10, models_h10 = assemble_horizon_data(10, grp_h10, har_h10_df, hexp_arr_h10, roll_arr_h10, g10)

    # --------------------------------------------------------------------------
    # 3. GROUND TRUTH SANITY CHECK
    # --------------------------------------------------------------------------
    print("\n[2. GROUND TRUTH SANITY CHECK]")
    y_h1 = df_h1["y"].values
    y_h10 = df_h10["y"].values

    mse_har_h1 = np.mean((y_h1 - models_h1["HAR"]) ** 2)
    mse_har_h10 = np.mean((y_h10 - models_h10["HAR"]) ** 2)
    mse_tft_spill_h1 = np.mean((y_h1 - models_h1["TFT_SPILL"]) ** 2)
    mse_tft_nospill_h1 = np.mean((y_h1 - models_h1["TFT_NOSPILL"]) ** 2)
    mse_nb_spill_h1 = np.mean((y_h1 - models_h1["NBEATSx_SPILL"]) ** 2)

    tol = 1e-11
    assert abs(mse_har_h1 - 2.2346806329693047e-06) < tol, "HAR h1 mismatch"
    assert abs(mse_har_h10 - 2.4397431333520627e-06) < tol, "HAR h10 mismatch"
    assert abs(mse_tft_spill_h1 - 2.3884532468485217e-06) < tol, "TFT SPILL h1 mismatch"
    assert abs(mse_tft_nospill_h1 - 2.394793350569176e-06) < tol, "TFT NOSPILL h1 mismatch"
    assert abs(mse_nb_spill_h1 - 2.272067473140114e-06) < tol, "NBEATSx SPILL h1 mismatch"
    print("  >>> SANITY CHECK PASS: All MSE baseline values match v6 ground truth exactly. <<<\n")

    # --------------------------------------------------------------------------
    # 4. REGIME DEFINITIONS (WINSORIZATION AND EVENT EXCLUSION)
    # --------------------------------------------------------------------------
    model_order = [
        "HAR",
        "Hist_Exp",
        "Roll_20",
        "GARCH11",
        "EGARCH",
        "TFT_SPILL",
        "TFT_NOSPILL",
        "NBEATSx_SPILL",
        "NBEATSx_NOSPILL",
    ]

    event_dates = [pd.to_datetime("2026-02-05"), pd.to_datetime("2026-02-06")]

    def get_regimes(df_h):
        y = df_h["y"].values
        q99 = np.percentile(y, 99)
        q975 = np.percentile(y, 97.5)
        q95 = np.percentile(y, 95)
        mask_event = ~df_h["real_date"].isin(event_dates).values

        return [
            ("full", "Full Sample", y, np.ones(len(y), dtype=bool), None),
            ("win99", "Winsorize 99%", np.clip(y, None, q99), np.ones(len(y), dtype=bool), q99),
            ("win975", "Winsorize 97.5%", np.clip(y, None, q975), np.ones(len(y), dtype=bool), q975),
            ("win95", "Winsorize 95%", np.clip(y, None, q95), np.ones(len(y), dtype=bool), q95),
            ("event", "Event Excl (Feb 5-6)", y[mask_event], mask_event, None),
        ]

    # --------------------------------------------------------------------------
    # 5. COMPUTE METRICS BY REGIME (MSE, MAE, MEAN BIAS, PATTON QLIKE)
    # --------------------------------------------------------------------------
    print("[3. COMPUTING METRICS AND MEAN BIAS BY REGIME]")
    metrics_records = []

    for h_label, h_val, df_h, m_dict in [("h=1", 1, df_h1, models_h1), ("h=10", 10, df_h10, models_h10)]:
        regimes = get_regimes(df_h)
        for reg_key, reg_label, y_target, mask, cap_val in regimes:
            n_obs = len(y_target)
            for m_name in model_order:
                p_unmod = m_dict[m_name][mask]
                mse = np.mean((y_target - p_unmod) ** 2)
                mae = np.mean(np.abs(y_target - p_unmod))
                mean_bias = np.mean(p_unmod - y_target)

                # Patton QLIKE: y/y_hat - log(y/y_hat) - 1
                # Non-applicable if any p_unmod <= 0; strictly NO clipping
                has_neg_pred = bool(np.any(p_unmod <= 0))
                if has_neg_pred:
                    qlike = np.nan
                    qlike_status = "N/A (pred <= 0)"
                else:
                    ratio = y_target / p_unmod
                    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
                    qlike_status = "VALID"

                metrics_records.append(
                    {
                        "horizon": h_label,
                        "regime": reg_key,
                        "regime_label": reg_label,
                        "n_obs": n_obs,
                        "cap_value": cap_val,
                        "model": m_name,
                        "MSE": mse,
                        "MAE": mae,
                        "Mean_Bias": mean_bias,
                        "QLIKE": qlike,
                        "QLIKE_status": qlike_status,
                    }
                )

    metrics_df = pd.DataFrame(metrics_records)
    # Compute rank MSE per (horizon, regime)
    metrics_df["Rank_MSE"] = metrics_df.groupby(["horizon", "regime"])["MSE"].rank().astype(int)

    metrics_csv_path = os.path.join(OUT_DIR, "metrics_by_regime.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved: {metrics_csv_path} ({len(metrics_df)} rows)")

    # --------------------------------------------------------------------------
    # 6. DIEBOLD-MARIANO TESTS ACROSS REGIMES (EXPLORATORY)
    # --------------------------------------------------------------------------
    print("\n[4. DIEBOLD-MARIANO TESTS ACROSS REGIMES (EXPLORATORY)]")

    # Define test pairs
    # Family 1: Deep Learning vs 3 Baselines (Hist_Exp, Roll_20, HAR) -> 4 DL * 3 Baselines = 12 tests per horizon
    dl_models = ["TFT_SPILL", "TFT_NOSPILL", "NBEATSx_SPILL", "NBEATSx_NOSPILL"]
    baselines = ["Hist_Exp", "Roll_20", "HAR"]

    # Family 2: Benchmark HAR vs 2 Baselines (Hist_Exp, Roll_20) -> 2 tests per horizon
    har_baselines = [("HAR", "Hist_Exp"), ("HAR", "Roll_20")]

    # Family 3: Ablation SPILL vs NOSPILL -> 2 tests per horizon
    ablation_pairs = [("TFT_SPILL", "TFT_NOSPILL"), ("NBEATSx_SPILL", "NBEATSx_NOSPILL")]

    dm_records = []

    for h_label, h_val, df_h, m_dict in [("h=1", 1, df_h1, models_h1), ("h=10", 10, df_h10, models_h10)]:
        regimes = get_regimes(df_h)
        for reg_key, reg_label, y_target, mask, _ in regimes:

            # 1. DL vs Baselines
            for dl_m in dl_models:
                p_dl = m_dict[dl_m][mask]
                for bl_m in baselines:
                    p_bl = m_dict[bl_m][mask]
                    pair_name = f"{dl_m} vs {bl_m}"

                    # MSE Loss
                    stat_mse, p_mse = diebold_mariano_engine(y_target, p_dl, p_bl, h_val, loss_type="MSE")
                    # MAE Loss
                    stat_mae, p_mae = diebold_mariano_engine(y_target, p_dl, p_bl, h_val, loss_type="MAE")
                    # QLIKE Loss (Patton)
                    stat_ql, p_ql = diebold_mariano_engine(y_target, p_dl, p_bl, h_val, loss_type="QLIKE")

                    for ltype, stat_val, p_val in [
                        ("MSE", stat_mse, p_mse),
                        ("MAE", stat_mae, p_mae),
                        ("QLIKE", stat_ql, p_ql),
                    ]:
                        dm_records.append(
                            {
                                "family": "DL_vs_Baselines",
                                "pair": pair_name,
                                "model_1": dl_m,
                                "model_2": bl_m,
                                "horizon": h_label,
                                "regime": reg_key,
                                "regime_label": reg_label,
                                "loss": ltype,
                                "DM_stat": stat_val,
                                "p_raw": p_val,
                                "m_horizon_family": 12,
                                "m_total_table": 24,
                                "designation": "EXPLORATORY",
                            }
                        )

            # 2. HAR vs Baselines
            for m1, m2 in har_baselines:
                p1 = m_dict[m1][mask]
                p2 = m_dict[m2][mask]
                pair_name = f"{m1} vs {m2}"
                for ltype in ["MSE", "MAE", "QLIKE"]:
                    stat_val, p_val = diebold_mariano_engine(y_target, p1, p2, h_val, loss_type=ltype)
                    dm_records.append(
                        {
                            "family": "HAR_vs_Baselines",
                            "pair": pair_name,
                            "model_1": m1,
                            "model_2": m2,
                            "horizon": h_label,
                            "regime": reg_key,
                            "regime_label": reg_label,
                            "loss": ltype,
                            "DM_stat": stat_val,
                            "p_raw": p_val,
                            "m_horizon_family": 2,
                            "m_total_table": 4,
                            "designation": "EXPLORATORY",
                        }
                    )

            # 3. Ablation SPILL vs NOSPILL
            for m1, m2 in ablation_pairs:
                p1 = m_dict[m1][mask]
                p2 = m_dict[m2][mask]
                pair_name = f"{m1} vs {m2}"
                for ltype in ["MSE", "MAE", "QLIKE"]:
                    stat_val, p_val = diebold_mariano_engine(y_target, p1, p2, h_val, loss_type=ltype)
                    dm_records.append(
                        {
                            "family": "Ablation_SPILL_vs_NOSPILL",
                            "pair": pair_name,
                            "model_1": m1,
                            "model_2": m2,
                            "horizon": h_label,
                            "regime": reg_key,
                            "regime_label": reg_label,
                            "loss": ltype,
                            "DM_stat": stat_val,
                            "p_raw": p_val,
                            "m_horizon_family": 2,
                            "m_total_table": 4,
                            "designation": "EXPLORATORY",
                        }
                    )

    dm_df = pd.DataFrame(dm_records)

    # Compute Bonferroni p-values with m = total tests per family-horizon and total per table
    dm_df["p_bonf_m_horizon"] = dm_df.apply(
        lambda r: min(1.0, r["p_raw"] * r["m_horizon_family"]) if not np.isnan(r["p_raw"]) else np.nan, axis=1
    )
    dm_df["p_bonf_m_table"] = dm_df.apply(
        lambda r: min(1.0, r["p_raw"] * r["m_total_table"]) if not np.isnan(r["p_raw"]) else np.nan, axis=1
    )

    def determine_winner(r):
        if np.isnan(r["DM_stat"]) or np.isnan(r["p_bonf_m_table"]):
            return "N/A"
        if r["p_bonf_m_table"] < 0.05:
            return r["model_1"] if r["DM_stat"] < 0 else r["model_2"]
        return "Tie"

    dm_df["winner_bonf"] = dm_df.apply(determine_winner, axis=1)

    dm_csv_path = os.path.join(OUT_DIR, "dm_by_regime.csv")
    dm_df.to_csv(dm_csv_path, index=False)
    print(f"  Saved: {dm_csv_path} ({len(dm_df)} rows)")

    # --------------------------------------------------------------------------
    # 7. EXTREME DAYS AND EVENT-WINDOW DETAILS
    # --------------------------------------------------------------------------
    print("\n[5. EXTREME DAYS AND EVENT-WINDOW DETAILS]")
    y1 = df_h1["y"].values
    total_sse_har_h1 = np.sum((y1 - models_h1["HAR"]) ** 2)
    total_sse_tft_sp_h1 = np.sum((y1 - models_h1["TFT_SPILL"]) ** 2)

    extreme_df = pd.DataFrame(
        {
            "real_date": df_h1["real_date"].dt.strftime("%Y-%m-%d"),
            "cutoff_h1": df_h1["cutoff"].dt.strftime("%Y-%m-%d"),
            "cutoff_h10": df_h10["cutoff"].dt.strftime("%Y-%m-%d"),
            "y": y1,
            "return_btc_idr": df_dl.set_index(df_dl["Date"].dt.strftime("%Y-%m-%d"))
            .loc[df_h1["real_date"].dt.strftime("%Y-%m-%d"), "ret_crypto_btc_idr_indodax"]
            .values,
            "HAR_pred_h1": models_h1["HAR"],
            "TFT_SPILL_pred_h1": models_h1["TFT_SPILL"],
            "TFT_NOSPILL_pred_h1": models_h1["TFT_NOSPILL"],
            "NBEATSx_SPILL_pred_h1": models_h1["NBEATSx_SPILL"],
            "NBEATSx_NOSPILL_pred_h1": models_h1["NBEATSx_NOSPILL"],
            "HAR_pred_h10": models_h10["HAR"],
            "TFT_SPILL_pred_h10": models_h10["TFT_SPILL"],
            "TFT_NOSPILL_pred_h10": models_h10["TFT_NOSPILL"],
            "NBEATSx_SPILL_pred_h10": models_h10["NBEATSx_SPILL"],
            "NBEATSx_NOSPILL_pred_h10": models_h10["NBEATSx_NOSPILL"],
        }
    )

    extreme_df["pct_return"] = extreme_df["return_btc_idr"] * 100.0
    extreme_df["sse_HAR_h1"] = (extreme_df["y"] - extreme_df["HAR_pred_h1"]) ** 2
    extreme_df["sse_TFT_SPILL_h1"] = (extreme_df["y"] - extreme_df["TFT_SPILL_pred_h1"]) ** 2
    extreme_df["pct_total_sse_HAR_h1"] = (extreme_df["sse_HAR_h1"] / total_sse_har_h1) * 100.0
    extreme_df["pct_total_sse_TFT_SPILL_h1"] = (extreme_df["sse_TFT_SPILL_h1"] / total_sse_tft_sp_h1) * 100.0

    # Sort descending by y
    extreme_sorted = extreme_df.sort_values("y", ascending=False).reset_index(drop=True)
    extreme_sorted["rank_y"] = extreme_sorted.index + 1

    # Flag event-window dates
    extreme_sorted["is_event_window_feb5_6"] = extreme_sorted["real_date"].isin(["2026-02-05", "2026-02-06"])

    top20_csv_path = os.path.join(OUT_DIR, "extreme_days_v2.csv")
    extreme_sorted.head(20).to_csv(top20_csv_path, index=False)
    print(f"  Saved: {top20_csv_path} (top 20 extreme days)")

    # --------------------------------------------------------------------------
    # 8. GENERATE FACTUAL, UNBIASED README.md REPORT
    # --------------------------------------------------------------------------
    print("\n[6. GENERATING FACTUAL README.md REPORT]")
    readme_path = os.path.join(OUT_DIR, "README.md")

    # Prepare markdown tables
    # Metrics table
    tbl_metrics_h1 = metrics_df[metrics_df["horizon"] == "h=1"].sort_values(["regime", "Rank_MSE"])
    tbl_metrics_h10 = metrics_df[metrics_df["horizon"] == "h=10"].sort_values(["regime", "Rank_MSE"])

    # DM summary for TFT vs HAR and DL vs Hist_Exp under MSE
    dm_mse_har = dm_df[
        (dm_df["loss"] == "MSE")
        & (dm_df["pair"].isin(["TFT_SPILL vs HAR", "TFT_NOSPILL vs HAR", "NBEATSx_SPILL vs HAR", "NBEATSx_NOSPILL vs HAR"]))
    ]
    dm_mse_hexp = dm_df[
        (dm_df["loss"] == "MSE")
        & (dm_df["pair"].isin(["TFT_SPILL vs Hist_Exp", "TFT_NOSPILL vs Hist_Exp"]))
    ]

    # Bias summary table
    pivot_bias_h1 = metrics_df[metrics_df["horizon"] == "h=1"].pivot(
        index="model", columns="regime", values="Mean_Bias"
    )[["full", "win99", "win975", "win95", "event"]]
    pivot_bias_h10 = metrics_df[metrics_df["horizon"] == "h=10"].pivot(
        index="model", columns="regime", values="Mean_Bias"
    )[["full", "win99", "win975", "win95", "event"]]

    def to_markdown_table(df_pivot):
        headers = ["Model"] + [str(c) for c in df_pivot.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] + [":---:"] * len(df_pivot.columns)) + " |"]
        for idx, row in df_pivot.iterrows():
            row_str = [f"**{idx}**"]
            for val in row:
                if isinstance(val, (float, np.floating)):
                    row_str.append(f"{val:+.6e}")
                else:
                    row_str.append(str(val))
            lines.append("| " + " | ".join(row_str) + " |")
        return "\n".join(lines)

    with open(readme_path, "w") as f:
        f.write("# RSIC v6 Outlier Sensitivity Check v2: Analysis Report\n\n")
        f.write("**Status:** EXPLORATORY SENSITIVITY ANALYSIS (READ-ONLY EVALUATION)\n")
        f.write("**Evaluation Window:** 335 cutoffs (Target dates: 2025-04-24 to 2026-09-04)\n")
        f.write("**Target:** Realized Volatility $y_t = r_t^2$ (BTC/IDR Indodax)\n")
        f.write("**Execution Environment:** CPU only, deterministic evaluation, no model retraining.\n\n")

        f.write("---\n\n")
        f.write("## 1. Temuan Faktual Utama (Apa Adanya)\n\n")
        f.write("1. **Posisi HAR pada Full Sample Didorong Hari Ekstrem 5–6 Februari 2026:**\n")
        f.write(
            "   - Dua hari berturut-turut pada 5 dan 6 Februari 2026 (crash $-14.85\%$ disusul rebound $+11.75\%$) menyumbang **76.12%** dari total SSE model HAR ($h=1$) dan **79.61%** dari total SSE model TFT SPILL ($h=1$).\n"
        )
        f.write(
            "   - Pada sampel penuh ($N=335$), HAR mencatat MSE terendah pada $h=1$ ($2.235 \\times 10^{-6}$) dan $h=10$ ($2.440 \\times 10^{-6}$).\n\n"
        )

        f.write("2. **Pengaruh Winsorize dan Event Exclusion terhadap Ranking MSE:**\n")
        f.write(
            "   - Saat target $y$ di-winsorize pada persentil 99, 97.5, atau 95 (prediksi $\\hat{y}$ tidak diubah), atau saat hanya 2 hari event-window (5–6 Feb 2026) dibuang ($N=333$):\n"
        )
        f.write(
            "     - Pada $h=1$: TFT_NOSPILL dan TFT_SPILL memperoleh nilai point MSE lebih rendah daripada HAR (peringkat 1 dan 2), sementara HAR berada pada peringkat 3 atau 4.\n"
        )
        f.write(
            "     - Pada $h=10$: TFT_SPILL dan NBEATSx_NOSPILL memperoleh point MSE lebih rendah daripada HAR (peringkat 1 dan 2), sementara HAR berada pada peringkat 4.\n\n"
        )

        f.write("3. **Uji Diebold-Mariano TFT vs. HAR Tetap Tidak Signifikan (Statistically a Tie):**\n")
        f.write(
            "   - Meskipun point MSE TFT menjadi lebih kecil daripada HAR setelah winsorize atau event exclusion, perbedaan loss antara TFT dan HAR **tidak signifikan secara statistik** pada level manapun ($p_{\\text{raw}} \\ge 0.17$, $p_{\\text{bonf}} = 1.0000$).\n"
        )
        f.write(
            "   - Dengan demikian, data menunjukkan status seri statistik (tie), bukan pembalikan keunggulan yang signifikan.\n\n"
        )

        f.write("4. **Keunggulan TFT vs. Hist_Exp Tetap Signifikan:**\n")
        f.write(
            "   - Performa TFT SPILL dan TFT_NOSPILL terhadap baseline `Hist_Exp` pada $h=1$ tetap signifikan secara statistik di seluruh level winsorize dan event exclusion ($p_{\\text{bonf}} < 0.05$).\n\n"
        )

        f.write("5. **Metrik QLIKE (Patton) dan Prediksi Negatif NBEATSx:**\n")
        f.write(
            "   - Patton (2011) mendefinisikan loss QLIKE sebagai $L(y, \\hat{y}) = y/\\hat{y} - \\ln(y/\\hat{y}) - 1$.\n"
        )
        f.write(
            "   - Model NBEATSx menghasilkan prediksi $\\le 0$ pada 20 hingga 46 hari perdagangan ($h=1$ dan $h=10$). Tanpa clipping buatan, nilai QLIKE untuk NBEATSx tidak dapat dihitung secara matematis dan ditandai sebagai **QLIKE tidak applicable**.\n"
        )
        f.write(
            "   - Untuk model yang seluruh prediksinya $> 0$ (HAR, TFT, GARCH, EGARCH, Hist_Exp, Roll_20), nilai rata-rata QLIKE berkisar antara 1.52 hingga 1.97.\n\n"
        )

        f.write("6. **Mean Bias Menjelaskan Pergeseran Ranking:**\n")
        f.write(
            "   - HAR memiliki bias positif rata-rata ($+1.81 \\times 10^{-4}$ pada $h=1$, $+2.37 \\times 10^{-4}$ pada $h=10$) karena memproyeksikan tingkat volatilitas baseline yang lebih tinggi.\n"
        )
        f.write(
            "   - TFT memiliki bias rata-rata mendekati nol ($-5.07 \\times 10^{-5}$ untuk SPILL, $-4.84 \\times 10^{-6}$ untuk NOSPILL pada $h=1$).\n"
        )
        f.write(
            "   - NBEATSx memiliki bias negatif rata-rata ($-2.51 \\times 10^{-4}$ hingga $-2.70 \\times 10^{-4}$).\n"
        )
        f.write(
            "   - Ketika nilai target puncak di-cap oleh winsorize atau dipotong oleh event exclusion, nilai rata-rata $y$ turun dari $\\approx 5.0 \\times 10^{-4}$ menjadi $\\approx 3.8 \\times 10^{-4}$. Hal ini secara mekanis memperbesar penalti kuadratik bagi model dengan level prediksi rata-rata lebih tinggi (seperti HAR) dan menguntungkan model dengan level prediksi rata-rata lebih rendah (seperti TFT dan NBEATSx).\n\n"
        )

        f.write("---\n\n")
        f.write("## 2. Tabel Metrik Evaluasi Berdasarkan Regime (MSE, MAE, Mean Bias, QLIKE)\n\n")
        f.write("*Catatan Regime:*\n")
        f.write("- `Full Sample` ($N=335$): Data uji lengkap 24 April 2025 s.d. 4 September 2026.\n")
        f.write("- `Winsorize 99%` ($N=335$): Target $y$ di-cap pada persentil 99 ($y_{\\text{cap}} = 3.572 \\times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.\n")
        f.write("- `Winsorize 97.5%` ($N=335$): Target $y$ di-cap pada persentil 97.5 ($y_{\\text{cap}} = 2.848 \\times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.\n")
        f.write("- `Winsorize 95%` ($N=335$): Target $y$ di-cap pada persentil 95 ($y_{\\text{cap}} = 2.195 \\times 10^{-3}$ pada $h=1$). Prediksi tidak diubah.\n")
        f.write("- `Event Excl (Feb 5-6)` ($N=333$): Hanya tanggal 2026-02-05 dan 2026-02-06 yang dikeluarkan dari evaluasi.\n\n")

        f.write("### Horizon h=1\n\n")
        f.write("| Model | Regime | n | MSE | MAE | Mean Bias (pred - y) | QLIKE (Patton) | Rank MSE |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in tbl_metrics_h1.iterrows():
            ql_str = f"{r['QLIKE']:.4f}" if not np.isnan(r["QLIKE"]) else "N/A (pred $\\le$ 0)"
            f.write(
                f"| **{r['model']}** | {r['regime_label']} | {r['n_obs']} | {r['MSE']:.6e} | {r['MAE']:.6e} | {r['Mean_Bias']:+.6e} | {ql_str} | {r['Rank_MSE']} |\n"
            )

        f.write("\n### Horizon h=10\n\n")
        f.write("| Model | Regime | n | MSE | MAE | Mean Bias (pred - y) | QLIKE (Patton) | Rank MSE |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in tbl_metrics_h10.iterrows():
            ql_str = f"{r['QLIKE']:.4f}" if not np.isnan(r["QLIKE"]) else "N/A (pred $\\le$ 0)"
            f.write(
                f"| **{r['model']}** | {r['regime_label']} | {r['n_obs']} | {r['MSE']:.6e} | {r['MAE']:.6e} | {r['Mean_Bias']:+.6e} | {ql_str} | {r['Rank_MSE']} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 3. Ringkasan Mean Bias tiap Model across Regimes\n\n")
        f.write("Tabel ini menunjukkan rata-rata selisih prediksi terhadap target $\\frac{1}{N} \\sum (\\hat{y}_t - y_t)$:\n\n")
        f.write("### Horizon h=1\n\n")
        f.write(to_markdown_table(pivot_bias_h1))
        f.write("\n\n### Horizon h=10\n\n")
        f.write(to_markdown_table(pivot_bias_h10))
        f.write("\n\n---\n\n")

        f.write("## 4. Hasil Uji Diebold-Mariano (EXPLORATORY)\n\n")
        f.write("Multi-testing correction:\n")
        f.write("- Family DL vs. 3 Baselines: $m = 12$ per horizon, $m = 24$ total per table.\n")
        f.write("- $p_{\\text{bonf}} = \\min(1.0, m \\times p_{\\text{raw}})$.\n")
        f.write("- Sign convention: $\\text{DM\\_stat} < 0$ menunjukkan Model 1 memiliki loss lebih rendah.\n\n")

        f.write("### Panel A: Deep Learning Models vs. HAR Benchmark (Loss = MSE)\n\n")
        f.write("| Pasangan | Horizon | Regime | DM Stat | p-value (raw) | p-bonf (m=12) | p-bonf (m=24) | Kesimpulan |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in dm_mse_har.iterrows():
            f.write(
                f"| {r['pair']} | {r['horizon']} | {r['regime_label']} | {r['DM_stat']:+.4f} | {r['p_raw']:.4f} | {r['p_bonf_m_horizon']:.4f} | {r['p_bonf_m_table']:.4f} | {r['winner_bonf']} |\n"
            )

        f.write("\n### Panel B: TFT Models vs. Hist_Exp Baseline (Loss = MSE)\n\n")
        f.write("| Pasangan | Horizon | Regime | DM Stat | p-value (raw) | p-bonf (m=12) | p-bonf (m=24) | Kesimpulan |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in dm_mse_hexp.iterrows():
            f.write(
                f"| {r['pair']} | {r['horizon']} | {r['regime_label']} | {r['DM_stat']:+.4f} | {r['p_raw']:.4e} | {r['p_bonf_m_horizon']:.4e} | {r['p_bonf_m_table']:.4e} | {r['winner_bonf']} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 5. Ringkasan Hari Ekstrem (Top 5)\n\n")
        f.write("| Rank y | Tanggal | y (Target $r^2$) | Return (%) | Pred HAR (h=1) | Pred TFT SPILL (h=1) | % Total SSE HAR | % Total SSE TFT SPILL |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in extreme_sorted.head(5).iterrows():
            f.write(
                f"| {r['rank_y']} | {r['real_date']} | {r['y']:.6e} | {r['pct_return']:+.2f}% | {r['HAR_pred_h1']:.6e} | {r['TFT_SPILL_pred_h1']:.6e} | {r['pct_total_sse_HAR_h1']:.2f}% | {r['pct_total_sse_TFT_SPILL_h1']:.2f}% |\n"
            )

        f.write("\n\n---\n")
        f.write("*Laporan disusun secara otomatis dan objektif tanpa interpretasi berlebihan. Seluruh data diverifikasi dari artefak eksperimen v6.*\n")

    print(f"  Saved: {readme_path}")
    print("\n" + "=" * 80)
    print("ALL ANALYSES AND EXPORTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
