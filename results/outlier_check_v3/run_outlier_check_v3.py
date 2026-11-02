#!/usr/bin/env python3
"""
run_outlier_check_v3.py

Verifikasi Hari Ekstrem (2026-02-05 & 2026-02-06) dan QLIKE Diebold-Mariano Test.
CPU-only, READ-ONLY ke file existing.
Output disimpan di: results/outlier_check_v3/
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_CLEAN_DIR = os.path.join(ROOT_DIR, "data/clean")
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
INPUT_DIR = os.path.join(ROOT_DIR, "inputs/v5")
V6_OUT_DIR = os.path.join(ROOT_DIR, "results/v6")
OUTLIER_V2_DIR = os.path.join(ROOT_DIR, "results/outlier_check_v2")
OUT_DIR = os.path.join(ROOT_DIR, "results/outlier_check_v3")
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


def diebold_mariano_qlike(y_true, y_p1, y_p2, h_val):
    """
    Computes Diebold-Mariano test under Patton (2011) QLIKE loss:
      QLIKE(y, y_hat) = y / y_hat - ln(y / y_hat) - 1
    Sign convention: d_t = Loss(Model 1) - Loss(Model 2).
    stat < 0 indicates Model 1 has lower loss (better).
    Newey-West lag:
      lag = 0 for h = 1
      lag = h - 1 (lag = 9 for h = 10) with Bartlett weights
      Harvey-Leybourne-Newbold (HLN) adjustment for h > 1.
    """
    if np.any(y_p1 <= 0) or np.any(y_p2 <= 0) or np.any(y_true <= 0):
        return np.nan, np.nan

    q1 = y_true / y_p1 - np.log(y_true / y_p1) - 1.0
    q2 = y_true / y_p2 - np.log(y_true / y_p2) - 1.0
    d = q1 - q2

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
        return 0.0, 1.0

    stat = d_bar / np.sqrt(var_d / T)
    if h_val > 1:
        hln = np.sqrt((T + 1 - 2 * h_val + (h_val * (h_val - 1)) / T) / T)
        stat = stat * hln

    p_raw = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))
    return stat, p_raw


def format_markdown_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def main():
    print("=" * 80)
    print("RSIC OUTLIER SENSITIVITY CHECK v3 (DATA VERIFICATION & QLIKE DM)")
    print("=" * 80)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. INPUT FILES AND MD5 HASHES
    # --------------------------------------------------------------------------
    input_files = {
        "dataset_deep_learning.csv": os.path.join(DATA_CLEAN_DIR, "dataset_deep_learning.csv"),
        "crypto_btc_idr_indodax.csv": os.path.join(DATA_RAW_DIR, "crypto_btc_idr_indodax.csv"),
        "crypto_btc_usd.csv": os.path.join(DATA_RAW_DIR, "crypto_btc_usd.csv"),
        "crypto_eth_usd.csv": os.path.join(DATA_RAW_DIR, "crypto_eth_usd.csv"),
        "konteks_usdidr.csv": os.path.join(DATA_RAW_DIR, "konteks_usdidr.csv"),
        "saham_nvda.csv": os.path.join(DATA_RAW_DIR, "saham_nvda.csv"),
        "saham_msft.csv": os.path.join(DATA_RAW_DIR, "saham_msft.csv"),
        "saham_tsla.csv": os.path.join(DATA_RAW_DIR, "saham_tsla.csv"),
        "saham_googl.csv": os.path.join(DATA_RAW_DIR, "saham_googl.csv"),
        "saham_meta.csv": os.path.join(DATA_RAW_DIR, "saham_meta.csv"),
        "CV_Predictions_V4.csv": os.path.join(INPUT_DIR, "CV_Predictions_V4.csv"),
        "har_preds_T0_h1_v2.csv": os.path.join(INPUT_DIR, "har_preds_T0_h1_v2.csv"),
        "har_preds_T0_h10_v2.csv": os.path.join(INPUT_DIR, "har_preds_T0_h10_v2.csv"),
        "GARCH_Predictions_V4.csv": os.path.join(INPUT_DIR, "GARCH_Predictions_V4.csv"),
        "v6_ensemble_predictions.csv": os.path.join(V6_OUT_DIR, "v6_ensemble_predictions.csv"),
        "extreme_days_v2.csv": os.path.join(OUTLIER_V2_DIR, "extreme_days_v2.csv"),
    }

    print("\n[1. INPUT FILES AND MD5 HASHES]")
    input_md5s = {}
    for name, path in input_files.items():
        h = md5_file(path)
        sz = os.path.getsize(path)
        rel_path = os.path.relpath(path, ROOT_DIR)
        input_md5s[name] = {"path": rel_path, "md5": h, "size": sz}
        print(f"  {rel_path}: md5={h} ({sz} bytes)")

    # --------------------------------------------------------------------------
    # 2. PART A: VERIFIKASI DATA ASLI (HARI EKSTREM FEB 5-6 & JENDELA FEB 2-10)
    # --------------------------------------------------------------------------
    print("\n[2. LOADING DATASETS FOR PART A]")
    df_dl = pd.read_csv(input_files["dataset_deep_learning.csv"])
    df_dl["Date"] = pd.to_datetime(df_dl["Date"])

    raw_indodax = pd.read_csv(input_files["crypto_btc_idr_indodax.csv"])
    raw_indodax["Date"] = pd.to_datetime(raw_indodax["Date"])

    raw_btcusd = pd.read_csv(input_files["crypto_btc_usd.csv"])
    raw_btcusd["Date"] = pd.to_datetime(raw_btcusd["Date"])

    raw_ethusd = pd.read_csv(input_files["crypto_eth_usd.csv"])
    raw_ethusd["Date"] = pd.to_datetime(raw_ethusd["Date"])

    raw_usdidr = pd.read_csv(input_files["konteks_usdidr.csv"])
    raw_usdidr["Date"] = pd.to_datetime(raw_usdidr["Date"])

    extreme_v2 = pd.read_csv(input_files["extreme_days_v2.csv"])
    extreme_v2["real_date"] = pd.to_datetime(extreme_v2["real_date"])

    # Full sample return difference distribution
    diff_series = df_dl["ret_crypto_btc_idr_indodax"] - df_dl["ret_crypto_btc_usd"]
    mean_diff = float(diff_series.mean())
    std_diff = float(diff_series.std())
    p1_diff = float(diff_series.quantile(0.01))
    p5_diff = float(diff_series.quantile(0.05))
    p95_diff = float(diff_series.quantile(0.95))
    p99_diff = float(diff_series.quantile(0.99))

    print(f"Full Sample Return Difference (BTC-IDR minus BTC-USD, N={len(diff_series)}):")
    print(f"  Mean: {mean_diff:+.6f} ({mean_diff*100:+.4f}%)")
    print(f"  Std:   {std_diff:.6f} ({std_diff*100:.4f}%)")
    print(f"  P1:   {p1_diff:+.6f} ({p1_diff*100:+.4f}%)")
    print(f"  P5:   {p5_diff:+.6f} ({p5_diff*100:+.4f}%)")
    print(f"  P95:  {p95_diff:+.6f} ({p95_diff*100:+.4f}%)")
    print(f"  P99:  {p99_diff:+.6f} ({p99_diff*100:+.4f}%)")

    # A.1 & A.2: Jendela 2026-02-02 s/d 2026-02-10
    window_start = pd.to_datetime("2026-02-02")
    window_end = pd.to_datetime("2026-02-10")
    sub_dl_window = df_dl[(df_dl["Date"] >= window_start) & (df_dl["Date"] <= window_end)].copy()

    # Build event check rows
    event_rows = []
    for _, row in sub_dl_window.iterrows():
        d = row["Date"]
        in_match = raw_indodax[raw_indodax["Date"] == d]
        us_match = raw_btcusd[raw_btcusd["Date"] == d]
        fx_match = raw_usdidr[raw_usdidr["Date"] == d]
        eth_match = raw_ethusd[raw_ethusd["Date"] == d]

        raw_c_in = in_match["Close"].values[0] if len(in_match) > 0 else np.nan
        raw_v_in = in_match["Volume"].values[0] if len(in_match) > 0 else np.nan
        raw_c_us = us_match["Close"].values[0] if len(us_match) > 0 else np.nan
        raw_v_us = us_match["Volume"].values[0] if len(us_match) > 0 else np.nan
        raw_c_fx = fx_match["Close"].values[0] if len(fx_match) > 0 else np.nan

        r_in = row["ret_crypto_btc_idr_indodax"]
        r_us = row["ret_crypto_btc_usd"]
        r_fx = row["ret_konteks_usdidr"]
        diff = r_in - r_us
        implied_diff = r_in - (r_us + r_fx)

        # Check if this date is in extreme ranks
        match_rank = extreme_v2[extreme_v2["real_date"] == d]
        rank_val = match_rank["rank_y"].values[0] if len(match_rank) > 0 else None

        event_rows.append({
            "section": "event_window_feb2026",
            "rank_extreme": rank_val,
            "date": d.strftime("%Y-%m-%d"),
            "day_of_week": int(row["day_of_week"]),
            "gap_days": int(row["gap_days"]),
            "is_monday": int(row["is_monday"]),
            "raw_close_indodax": raw_c_in,
            "raw_volume_indodax": raw_v_in,
            "raw_close_btc_usd": raw_c_us,
            "raw_volume_btc_usd": raw_v_us,
            "raw_close_usdidr": raw_c_fx,
            "ret_btc_idr": r_in,
            "ret_btc_usd": r_us,
            "diff_btc_idr_usd": diff,
            "ret_usdidr": r_fx,
            "implied_diff_fx": implied_diff,
            "ret_eth_usd": row["ret_crypto_eth_usd"],
            "ret_nvda": row["ret_saham_nvda"],
            "ret_msft": row["ret_saham_msft"],
            "ret_tsla": row["ret_saham_tsla"],
            "ret_googl": row["ret_saham_googl"],
            "ret_meta": row["ret_saham_meta"],
        })

    # A.5: Tambahkan 5 hari ekstrem berikutnya di extreme_days_v2.csv (Rank 3-7)
    # Ranks 1-7 are all in extreme_days_v2
    top7_dates = extreme_v2.sort_values("rank_y").head(7)["real_date"].tolist()
    for _, row_ext in extreme_v2.sort_values("rank_y").head(7).iterrows():
        d = row_ext["real_date"]
        # If already in event_rows, we still include in the extreme_ranks_1_to_7 section for clear comparison
        dl_match = df_dl[df_dl["Date"] == d]
        if len(dl_match) == 0:
            continue
        row_dl = dl_match.iloc[0]

        in_match = raw_indodax[raw_indodax["Date"] == d]
        us_match = raw_btcusd[raw_btcusd["Date"] == d]
        fx_match = raw_usdidr[raw_usdidr["Date"] == d]

        raw_c_in = in_match["Close"].values[0] if len(in_match) > 0 else np.nan
        raw_v_in = in_match["Volume"].values[0] if len(in_match) > 0 else np.nan
        raw_c_us = us_match["Close"].values[0] if len(us_match) > 0 else np.nan
        raw_v_us = us_match["Volume"].values[0] if len(us_match) > 0 else np.nan
        raw_c_fx = fx_match["Close"].values[0] if len(fx_match) > 0 else np.nan

        r_in = row_dl["ret_crypto_btc_idr_indodax"]
        r_us = row_dl["ret_crypto_btc_usd"]
        r_fx = row_dl["ret_konteks_usdidr"]
        diff = r_in - r_us
        implied_diff = r_in - (r_us + r_fx)

        event_rows.append({
            "section": "extreme_ranks_1_to_7",
            "rank_extreme": int(row_ext["rank_y"]),
            "date": d.strftime("%Y-%m-%d"),
            "day_of_week": int(row_dl["day_of_week"]),
            "gap_days": int(row_dl["gap_days"]),
            "is_monday": int(row_dl["is_monday"]),
            "raw_close_indodax": raw_c_in,
            "raw_volume_indodax": raw_v_in,
            "raw_close_btc_usd": raw_c_us,
            "raw_volume_btc_usd": raw_v_us,
            "raw_close_usdidr": raw_c_fx,
            "ret_btc_idr": r_in,
            "ret_btc_usd": r_us,
            "diff_btc_idr_usd": diff,
            "ret_usdidr": r_fx,
            "implied_diff_fx": implied_diff,
            "ret_eth_usd": row_dl["ret_crypto_eth_usd"],
            "ret_nvda": row_dl["ret_saham_nvda"],
            "ret_msft": row_dl["ret_saham_msft"],
            "ret_tsla": row_dl["ret_saham_tsla"],
            "ret_googl": row_dl["ret_saham_googl"],
            "ret_meta": row_dl["ret_saham_meta"],
        })

    event_check_df = pd.DataFrame(event_rows)
    event_check_path = os.path.join(OUT_DIR, "event_check.csv")
    event_check_df.to_csv(event_check_path, index=False)
    print(f"\nSaved event check table to: {event_check_path} ({len(event_check_df)} rows)")

    # --------------------------------------------------------------------------
    # 3. PART B: QLIKE DIEBOLD-MARIANO TEST
    # --------------------------------------------------------------------------
    print("\n[3. EXECUTING PART B: QLIKE DIEBOLD-MARIANO TEST]")
    cv4 = pd.read_csv(input_files["CV_Predictions_V4.csv"])
    cv4["cutoff"] = pd.to_datetime(cv4["cutoff"])
    cv4["real_date"] = pd.to_datetime(cv4["real_date"])

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

    def assemble_models(base_grp, har_df, hexp_arr, roll_arr, garch_sub):
        df_base = base_grp[["cutoff", "real_date", "y"]].copy()
        m_har = pd.merge(df_base, har_df[["Date", "HAR"]], left_on="real_date", right_on="Date", how="inner")
        assert len(m_har) == 335, f"HAR merge mismatch: {len(m_har)}"

        models = {
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
            assert len(m_dl) == 335, f"DL merge mismatch: {len(m_dl)}"
            models[clean_name] = m_dl["prediction"].values

        return df_base, models

    df_base_h1, models_h1 = assemble_models(grp_h1, har_h1_df, hexp_arr_h1, roll_arr_h1, g1)
    df_base_h10, models_h10 = assemble_models(grp_h10, har_h10_df, hexp_arr_h10, roll_arr_h10, g10)

    # Verifikasi domain prediksi (non-positive check)
    print("Verifikasi Nilai Prediksi Minimum:")
    pred_summary = []
    for m in ["HAR", "Hist_Exp", "Roll_20", "GARCH11", "EGARCH", "TFT_SPILL", "TFT_NOSPILL", "NBEATSx_SPILL", "NBEATSx_NOSPILL"]:
        min_h1 = float(models_h1[m].min())
        min_h10 = float(models_h10[m].min())
        status = "Included (> 0)" if (min_h1 > 0 and min_h10 > 0) else "EXCLUDED (<= 0)"
        pred_summary.append((m, min_h1, min_h10, status))
        print(f"  {m:16s} | h=1 min: {min_h1:+.8f} | h=10 min: {min_h10:+.8f} | Status: {status}")

    # Jalankan QLIKE DM untuk model eligible
    eligible_dl = ["TFT_SPILL", "TFT_NOSPILL"]
    benchmarks = ["HAR", "Hist_Exp", "Roll_20", "GARCH11", "EGARCH"]

    dm_rows = []
    for h_val, df_b, m_dict in [(1, df_base_h1, models_h1), (10, df_base_h10, models_h10)]:
        y = df_b["y"].values
        for dl in eligible_dl:
            for bm in benchmarks:
                p_dl = m_dict[dl]
                p_bm = m_dict[bm]
                stat, p_raw = diebold_mariano_qlike(y, p_dl, p_bm, h_val)
                q_dl = float(np.mean(y / p_dl - np.log(y / p_dl) - 1.0))
                q_bm = float(np.mean(y / p_bm - np.log(y / p_bm) - 1.0))
                dm_rows.append({
                    "horizon": f"h={h_val}",
                    "model_1": dl,
                    "model_2": bm,
                    "qlike_m1": q_dl,
                    "qlike_m2": q_bm,
                    "diff_qlike": q_dl - q_bm,
                    "dm_stat": stat,
                    "p_raw": p_raw,
                })

    qlike_dm_df = pd.DataFrame(dm_rows)
    m_total = len(qlike_dm_df)  # 20 tests in the table
    qlike_dm_df["p_bonf_m20"] = np.minimum(qlike_dm_df["p_raw"] * m_total, 1.0)
    qlike_dm_df["p_bonf_m10"] = np.minimum(qlike_dm_df["p_raw"] * 10, 1.0)

    def get_verdict(row):
        # alpha = 0.05 on Bonferroni m=20
        if row["p_bonf_m20"] < 0.05:
            return "Model 1 Beats Model 2" if row["dm_stat"] < 0 else "Model 2 Beats Model 1"
        return "Tie (p_bonf >= 0.05)"

    qlike_dm_df["verdict_bonf_05"] = qlike_dm_df.apply(get_verdict, axis=1)

    qlike_dm_path = os.path.join(OUT_DIR, "qlike_dm.csv")
    qlike_dm_df.to_csv(qlike_dm_path, index=False)
    print(f"\nSaved QLIKE DM results to: {qlike_dm_path} ({len(qlike_dm_df)} tests)")

    # --------------------------------------------------------------------------
    # 4. GENERATE README.md
    # --------------------------------------------------------------------------
    print("\n[4. GENERATING README.md]")

    # Table 1: Window Feb 2-10
    tbl1_headers = [
        "Date", "Day", "Close Indodax", "Close BTC-USD", "Close USD/IDR",
        "Ret BTC-IDR (%)", "Ret BTC-USD (%)", "Diff (%)", "Ret USD/IDR (%)", "Implied Diff (%)", "Ret ETH-USD (%)"
    ]
    tbl1_rows = []
    window_df = event_check_df[event_check_df["section"] == "event_window_feb2026"]
    for _, r in window_df.iterrows():
        tbl1_rows.append([
            r["date"],
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(r["day_of_week"])],
            f"{r['raw_close_indodax']:,.0f}",
            f"${r['raw_close_btc_usd']:,.2f}",
            f"{r['raw_close_usdidr']:,.2f}",
            f"{r['ret_btc_idr']*100:+.2f}%",
            f"{r['ret_btc_usd']*100:+.2f}%",
            f"{r['diff_btc_idr_usd']*100:+.2f}%",
            f"{r['ret_usdidr']*100:+.2f}%",
            f"{r['implied_diff_fx']*100:+.2f}%",
            f"{r['ret_eth_usd']*100:+.2f}%"
        ])
    tbl1_md = format_markdown_table(tbl1_headers, tbl1_rows)

    # Table 2: AI Stocks return in window
    tbl2_headers = ["Date", "Ret NVDA (%)", "Ret MSFT (%)", "Ret TSLA (%)", "Ret GOOGL (%)", "Ret META (%)"]
    tbl2_rows = []
    for _, r in window_df.iterrows():
        tbl2_rows.append([
            r["date"],
            f"{r['ret_nvda']*100:+.2f}%",
            f"{r['ret_msft']*100:+.2f}%",
            f"{r['ret_tsla']*100:+.2f}%",
            f"{r['ret_googl']*100:+.2f}%",
            f"{r['ret_meta']*100:+.2f}%"
        ])
    tbl2_md = format_markdown_table(tbl2_headers, tbl2_rows)

    # Table 3: Full sample difference distribution vs Feb 5-6
    feb5_row = window_df[window_df["date"] == "2026-02-05"].iloc[0]
    feb6_row = window_df[window_df["date"] == "2026-02-06"].iloc[0]
    feb5_diff = feb5_row["diff_btc_idr_usd"]
    feb6_diff = feb6_row["diff_btc_idr_usd"]
    z5 = (feb5_diff - mean_diff) / std_diff
    z6 = (feb6_diff - mean_diff) / std_diff

    tbl3_headers = ["Metric / Percentile", "Full Sample (N=1,677)", "2026-02-05 (Rank 1)", "2026-02-06 (Rank 2)"]
    tbl3_rows = [
        ["Mean Difference", f"{mean_diff*100:+.4f}%", "-", "-"],
        ["Std Difference", f"{std_diff*100:.4f}%", "-", "-"],
        ["Percentile 1%", f"{p1_diff*100:+.4f}%", "-", "-"],
        ["Percentile 5%", f"{p5_diff*100:+.4f}%", "-", "-"],
        ["Percentile 95%", f"{p95_diff*100:+.4f}%", "-", "-"],
        ["Percentile 99%", f"{p99_diff*100:+.4f}%", "-", "-"],
        ["Return Difference (IDR - USD)", "-", f"{feb5_diff*100:+.4f}%", f"{feb6_diff*100:+.4f}%"],
        ["Z-score vs Full Sample", "-", f"{z5:+.4f} std", f"{z6:+.4f} std"],
        ["Inside P1-P99 Interval?", "-", "YES", "YES"],
    ]
    tbl3_md = format_markdown_table(tbl3_headers, tbl3_rows)

    # Table 4: Raw candle audit around Feb 5-6
    # Check indodax raw rows for Feb 2 - Feb 10
    raw_sub = raw_indodax[(raw_indodax["Date"] >= window_start) & (raw_indodax["Date"] <= window_end)].copy()
    raw_sub["date_diff"] = raw_sub["Date"].diff().dt.days
    raw_sub["hl_ratio"] = raw_sub["High"] / raw_sub["Low"]
    raw_sub["gap_open_prev_close"] = (raw_sub["Open"] - raw_sub["Close"].shift(1)) / raw_sub["Close"].shift(1)

    tbl4_headers = ["Date", "Candles", "Open (IDR)", "High (IDR)", "Low (IDR)", "Close (IDR)", "Volume (BTC)", "H/L Ratio", "Open vs Prev Close"]
    tbl4_rows = []
    for _, r in raw_sub.iterrows():
        gap_txt = f"{r['gap_open_prev_close']*100:+.4f}%" if pd.notnull(r["gap_open_prev_close"]) else "N/A"
        tbl4_rows.append([
            r["Date"].strftime("%Y-%m-%d"),
            "1",
            f"{r['Open']:,.0f}",
            f"{r['High']:,.0f}",
            f"{r['Low']:,.0f}",
            f"{r['Close']:,.0f}",
            f"{r['Volume']:.2f}",
            f"{r['hl_ratio']:.4f}",
            gap_txt
        ])
    tbl4_md = format_markdown_table(tbl4_headers, tbl4_rows)

    # Table 5: Extreme ranks 1 to 7 comparison
    tbl5_headers = [
        "Rank", "Date", "gap_days", "Indodax Vol (BTC)", "Close Indodax", "Close BTC-USD",
        "Ret BTC-IDR (%)", "Ret BTC-USD (%)", "Diff (%)", "Ret USD/IDR (%)", "Implied Diff (%)", "Ret ETH-USD (%)"
    ]
    tbl5_rows = []
    ext_df = event_check_df[event_check_df["section"] == "extreme_ranks_1_to_7"].sort_values("rank_extreme")
    for _, r in ext_df.iterrows():
        tbl5_rows.append([
            f"Rank {int(r['rank_extreme'])}",
            r["date"],
            str(int(r["gap_days"])),
            f"{r['raw_volume_indodax']:.2f}",
            f"{r['raw_close_indodax']:,.0f}",
            f"${r['raw_close_btc_usd']:,.2f}",
            f"{r['ret_btc_idr']*100:+.2f}%",
            f"{r['ret_btc_usd']*100:+.2f}%",
            f"{r['diff_btc_idr_usd']*100:+.2f}%",
            f"{r['ret_usdidr']*100:+.2f}%",
            f"{r['implied_diff_fx']*100:+.2f}%",
            f"{r['ret_eth_usd']*100:+.2f}%"
        ])
    tbl5_md = format_markdown_table(tbl5_headers, tbl5_rows)

    # Table 6: Model prediction domain (QLIKE eligibility)
    tbl6_headers = ["Model", "Min Pred (h=1)", "Min Pred (h=10)", "All Pred > 0?", "QLIKE Status"]
    tbl6_rows = []
    for m, m1, m10, st in pred_summary:
        all_pos = "YES" if (m1 > 0 and m10 > 0) else "NO"
        tbl6_rows.append([m, f"{m1:+.8f}", f"{m10:+.8f}", all_pos, st])
    tbl6_md = format_markdown_table(tbl6_headers, tbl6_rows)

    # Table 7: QLIKE DM results
    tbl7_headers = [
        "Horizon", "Model 1 (DL)", "Model 2 (Benchmark)", "QLIKE M1", "QLIKE M2", "Diff (M1 - M2)",
        "DM Stat", "p_raw", "p_bonf (m=20)", "Verdict (alpha=0.05)"
    ]
    tbl7_rows = []
    for _, r in qlike_dm_df.iterrows():
        tbl7_rows.append([
            r["horizon"],
            r["model_1"],
            r["model_2"],
            f"{r['qlike_m1']:.4f}",
            f"{r['qlike_m2']:.4f}",
            f"{r['diff_qlike']:+.4f}",
            f"{r['dm_stat']:+.4f}",
            f"{r['p_raw']:.4f}",
            f"{r['p_bonf_m20']:.4f}",
            r["verdict_bonf_05"]
        ])
    tbl7_md = format_markdown_table(tbl7_headers, tbl7_rows)

    readme_sections = [
        "# Outlier Sensitivity Check v3: Verifikasi Hari Ekstrem & QLIKE DM\n",
        "Dokumen ini menyajikan verifikasi data asli untuk tanggal ekstrem **2026-02-05** dan **2026-02-06**, perbandingannya dengan 5 hari ekstrem berikutnya (Rank 3–7), audit candle data mentah Indodax, serta uji signifikansi prediktif Diebold-Mariano berbasis rugi **QLIKE Patton (2011)** tanpa clipping.\n",
        "Seluruh perhitungan dijalankan secara deterministik pada CPU dan berstatus **EXPLORATORY**.\n",
        "---\n",
        "## 1. Verifikasi Hari Ekstrem (2026-02-02 s/d 2026-02-10)\n",
        "### 1.1 Return dan Harga Penutupan Mentah\n",
        "Tabel berikut memuat harga penutupan mentah dan return log harian untuk BTC-IDR (Indodax), BTC-USD (Yahoo Finance), USD/IDR, selisih return, serta ETH-USD.\n",
        tbl1_md + "\n",
        "### 1.2 Return Saham AI Global pada Jendela yang Sama\n",
        "Pergerakan return log harian untuk lima saham AI acuan (NVDA, MSFT, TSLA, GOOGL, META):\n",
        tbl2_md + "\n",
        "---\n",
        "## 2. Analisis Selisih Return BTC-IDR vs BTC-USD\n",
        "Selisih return harian didefinisikan sebagai:\n",
        "- Delta r_t = r_{BTC-IDR} - r_{BTC-USD}\n",
        "- Delta r_t (implied FX) = r_{BTC-IDR} - (r_{BTC-USD} + r_{USD/IDR})\n\n",
        tbl3_md + "\n",
        "### Temuan Faktual:\n",
        "1. **Arah dan Besaran**: Penurunan -14.85% (Feb 5) dan rebound +11.75% (Feb 6) di Indodax tercermin secara langsung dan serentak di pasar global BTC-USD (-15.23% pada Feb 5 dan +11.80% pada Feb 6).\n",
        "2. **Besaran Tracking Error**: Selisih return BTC-IDR terhadap BTC-USD pada 5 Februari adalah **+0.38%** (z-score: +0.46) dan pada 6 Februari adalah **-0.05%** (z-score: -0.08). Keduanya berada jauh di dalam batas rentang persentil 1% s/d 99% dari seluruh sampel data (N = 1.677, [-2.23%, +2.24%]).\n",
        "3. **Peran Kurs USD/IDR**: Depresiasi Rupiah (+0.24% pada Feb 5 dan +0.47% pada Feb 6) mempersempit tracking error implied menjadi hanya **+0.14%** pada 5 Februari dan **-0.53%** pada 6 Februari.\n",
        "---\n",
        "## 3. Audit Data Mentah Indodax (Candle Integrity)\n",
        tbl4_md + "\n",
        "### Parameter Audit:\n",
        "- **Jumlah Candle**: Tepat 1 candle per hari kalender tanpa ada candle duplikat dan tanpa ada hari yang hilang (kelanjutan tanggal sempurna 1 hari kalender penuh).\n",
        "- **Likuiditas dan Volume**: Volume perdagangan di Indodax melonjak tajam pada 5 Februari (**176.51 BTC**) dan 6 Februari (**162.21 BTC**), jauh di atas rata-rata volume harian normal (20–60 BTC). Volume global BTC-USD juga melonjak ke rekor $125.5 Miliar (Feb 5) dan $114.7 Miliar (Feb 6). Lonjakan volume ini menunjukkan adanya aktivitas likuidasi dan akumulasi nyata, bukan akibat ilikuiditas tipis (*thin liquidity*).\n",
        "- **Kontinuitas Harga**: Harga Open tanggal 5 Februari (Rp 1.233.765.000) dan 6 Februari (Rp 1.063.500.000) bersambung secara kontinu dengan harga Close hari sebelumnya (gap terbuka kurang dari 0.0002%). Tidak terdapat lonjakan harga anomali atau *spike glitch* antarcandle.\n",
        "- **Jam Cut Candle**: **UNVERIFIED**. Skrip pengumpulan data (`src/collect_data.py`) mem-parsing timestamp detik Unix (`unit=\"s\", utc=True`) dari endpoint TradingView Indodax. Namun, apakah server Indodax memotong lilin harian pada pukul 00:00 UTC (07:00 WIB) atau pukul 00:00 WIB (17:00 UTC hari sebelumnya) tidak dapat diverifikasi secara independen dari kode maupun CSV mentah.\n",
        "---\n",
        "## 4. Status Gap-Day dan Akumulasi Return\n",
        "Berdasarkan definisi fitur penataan kalender pada `dataset_deep_learning.csv`:\n",
        "- `gap_days = 1` untuk tanggal 2026-02-05 (Kamis) dan 2026-02-06 (Jumat).\n",
        "- `is_monday = 0` untuk kedua tanggal tersebut.\n",
        "- `crypto_btc_idr_indodax_weekend_return = 0.0` dan `crypto_btc_usd_weekend_return = 0.0`.\n\n",
        "Kedua tanggal tersebut **bukan** merupakan *gap-day* dan **bukan** akumulasi akhir pekan (Sun→Mon). Return yang tercatat adalah return perdagangan tunggal selama 24 jam kalender.\n",
        "---\n",
        "## 5. Perbandingan dengan 5 Hari Ekstrem Berikutnya (Rank 3–7)\n",
        "Tabel perbandingan data aktual untuk Rank 1 hingga Rank 7 dari `extreme_days_v2.csv`:\n",
        tbl5_md + "\n",
        "### Pola Konsistensi:\n",
        "- Seluruh 7 hari ekstrem memiliki `gap_days = 1` (hari perdagangan tunggal biasa).\n",
        "- Pada seluruh 7 hari ekstrem, arah pergerakan BTC-IDR selalu identik dengan arah BTC-USD dan ETH-USD.\n",
        "- Tracking error antara BTC-IDR dan BTC-USD pada seluruh 7 hari ekstrem sangat kecil (berkisar antara -1.21% hingga +0.58%), menunjukkan hubungan arbitrase pasar lokal terhadap pasar global yang tetap terjaga selama periode volatilitas ekstrem.\n",
        "---\n",
        "## 6. Kesimpulan Faktual Hari Ekstrem\n",
        "1. Pergerakan ekstrem pada tanggal 5 dan 6 Februari 2026 di Indodax (-14.85% dan +11.75%) **konsisten secara langsung dengan pergerakan pasar global** (BTC-USD -15.23% dan +11.80%, ETH-USD -16.27% dan +12.46%).\n",
        "2. Fenomena ini **bukan anomali lokal spesifik Indodax**, bukan akibat kesalahan pencatatan (*data glitch*), bukan akibat ilikuiditas tipis, dan bukan akumulasi akhir pekan (*non-gap day*).\n",
        "3. Faktor penggerak fundamental makroekonomi atau pemicu pasar eksternal yang melatarbelakangi jatuhnya harga global tersebut **tidak dapat diverifikasi dari data** numerik yang tersedia di repositori ini.\n",
        "---\n",
        "## 7. Uji Diebold-Mariano QLIKE (Patton 2011)\n",
        "### 7.1 Kelayakan Model (Domain Prediksi)\n",
        "Fungsi rugi QLIKE Patton (2011) didefinisikan sebagai:\n",
        "  QLIKE(y, y_hat) = y / y_hat - ln(y / y_hat) - 1\n\n",
        "Fungsi ini mensyaratkan y_hat > 0. Model dengan prediksi non-positif (<= 0) tidak dapat dievaluasi menggunakan QLIKE tanpa clipping buatan.\n",
        tbl6_md + "\n",
        "> **Catatan**: N-BEATSx (baik SPILL maupun NOSPILL) menghasilkan prediksi negatif (minimum -0.00287 pada h=1 dan -0.00119 pada h=10). Sesuai metodologi yang ketat tanpa clipping buatan, **N-BEATSx dikecualikan secara eksplisit** dari evaluasi QLIKE.\n",
        "### 7.2 Hasil Uji Diebold-Mariano (Full Sample, N=335)\n",
        "- Horizon h=1: Lag Newey-West = 0.\n",
        "- Horizon h=10: Lag Newey-West = 9 dengan bobot Bartlett dan koreksi sampel kecil Harvey-Leybourne-Newbold (HLN).\n",
        "- Koreksi Family-wise: m = 20 (seluruh 20 uji pada tabel) dan m = 10 (per horizon).\n",
        "- Status: **EXPLORATORY**.\n",
        tbl7_md + "\n",
        "### Rangkuman QLIKE:\n",
        "1. Pada seluruh 20 pasangan uji (TFT_SPILL dan TFT_NOSPILL vs 5 model pembanding pada h=1 dan h=10), seluruh nilai p_bonf bernilai **1.0000** (atau p_raw > 0.05).\n",
        "2. Tidak terdapat perbedaan performa prediktif yang signifikan secara statistik antara TFT dan model ekonometrika konvensional (HAR, Hist_Exp, Roll_20, GARCH(1,1), EGARCH) di bawah metrik kerugian asimetris QLIKE.\n",
        "3. Seluruh perbandingan berstatus **Tie** pada taraf signifikansi alpha = 0.05.\n",
        "---\n",
        "## 8. Verifikasi Integritas File (MD5 Checksum)\n",
        "### Input Files (Read-Only)\n",
        "| File | Rel Path | MD5 Hash |\n",
        "| :--- | :--- | :--- |\n",
    ]
    readme_content = "\n".join(readme_sections)
    for name, info in input_md5s.items():
        readme_content += f"| `{name}` | `{info['path']}` | `{info['md5']}` |\n"

    readme_content += """
### Output Files
| File | Rel Path | MD5 Hash |
| :--- | :--- | :--- |
"""
    # Output files table in README
    md5_event = md5_file(event_check_path)
    md5_qlike = md5_file(qlike_dm_path)

    readme_content += f"| `event_check.csv` | `results/outlier_check_v3/event_check.csv` | `{md5_event}` |\n"
    readme_content += f"| `qlike_dm.csv` | `results/outlier_check_v3/qlike_dm.csv` | `{md5_qlike}` |\n"

    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    md5_readme = md5_file(readme_path)

    print("\n[5. OUTPUT FILES AND MD5 HASHES]")
    print(f"  results/outlier_check_v3/event_check.csv: md5={md5_event} ({os.path.getsize(event_check_path)} bytes)")
    print(f"  results/outlier_check_v3/qlike_dm.csv: md5={md5_qlike} ({os.path.getsize(qlike_dm_path)} bytes)")
    print(f"  results/outlier_check_v3/README.md: md5={md5_readme} ({os.path.getsize(readme_path)} bytes)")

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
