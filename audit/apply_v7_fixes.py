#!/usr/bin/env python3
"""
Script to apply the required 6 fixes in-place to notebooks/08_dl_v7_audit_loss_refit.ipynb.
"""

import os
import json

NB_PATH = os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb')

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

# ------------------------------------------------------------------------------
# 1. Cell 2: RUN_TFT_MAE_TUNED default = True, tracking globals
# ------------------------------------------------------------------------------
cell2_src = ''.join(nb['cells'][2]['source'])

old_c2_target = 'RUN_TFT_MAE_TUNED = False  # Set True to tune TFT with MAE loss alongside NBEATSx with MSE loss'
new_c2_target = '''RUN_TFT_MAE_TUNED = True  # Set True to tune TFT with MAE loss alongside NBEATSx with MSE loss

# Global tracking structures for audit provenance and stage timers
STAGE_TIMINGS = {
    'A': None,
    'A_TUNED': None,
    'B': None,
    'POSTPROCESSING': None
}
MISSING_BASELINES = []
PARTIAL_ENSEMBLES = []'''

assert old_c2_target in cell2_src, "Target in Cell 2 not found!"
cell2_src = cell2_src.replace(old_c2_target, new_c2_target)
nb['cells'][2]['source'] = [line + '\n' for line in cell2_src.splitlines()]

# ------------------------------------------------------------------------------
# 2. Cell 6: Enhance find_file to search /kaggle/input and subdirs recursively
# ------------------------------------------------------------------------------
cell6_src = ''.join(nb['cells'][6]['source'])

old_c6_find = '''def find_file(filename, candidate_dirs=None):
    if candidate_dirs:
        for d in candidate_dirs:
            if d and os.path.exists(d):
                p = os.path.join(d, filename)
                if os.path.exists(p):
                    return os.path.abspath(p)
    if os.path.exists('/kaggle/input'):
        matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    for base in ['.', '..', '../..']:
        for sub in ['inputs/v5', 'data/clean', 'scratch/final', 'rsic-v5-inputs']:
            p = os.path.join(base, sub, filename)
            if os.path.exists(p):
                return os.path.abspath(p)
        matches = glob.glob(f'{base}/**/{filename}', recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    raise FileNotFoundError(f'File {filename} tidak ditemukan di /kaggle/input maupun folder lokal!')'''

new_c6_find = '''def find_file(filename, candidate_dirs=None):
    if candidate_dirs:
        for d in candidate_dirs:
            if d and os.path.exists(d):
                p = os.path.join(d, filename)
                if os.path.exists(p):
                    return os.path.abspath(p)
    if os.path.exists('/kaggle/input'):
        matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    for base in ['.', '..', '../..']:
        for sub in ['inputs/v5', 'data/clean', 'scratch/final', 'rsic-v5-inputs', 'results/v6', 'v6_outputs', 'results']:
            p = os.path.join(base, sub, filename)
            if os.path.exists(p):
                return os.path.abspath(p)
        matches = glob.glob(f'{base}/**/{filename}', recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    raise FileNotFoundError(f'File {filename} tidak ditemukan di /kaggle/input maupun folder lokal!')'''

assert old_c6_find in cell6_src, "find_file target in Cell 6 not found!"
cell6_src = cell6_src.replace(old_c6_find, new_c6_find)
nb['cells'][6]['source'] = [line + '\n' for line in cell6_src.splitlines()]

# ------------------------------------------------------------------------------
# 3. Cell 10 (Stage A): Record stage elapsed time
# ------------------------------------------------------------------------------
cell10_src = ''.join(nb['cells'][10]['source'])
old_c10_start = '''if STAGE in ['A', 'SMOKE']:
    print('=' * 80)
    print(f'STARTING EXPERIMENT A: UNTUNED LOSS-MATCHED RETRAINING ({len(exp_a_configs)} configs x {len(exp_a_seeds)} seeds)')
    print('=' * 80)'''

new_c10_start = '''if STAGE in ['A', 'SMOKE']:
    t0_stage_a = time.time()
    print('=' * 80)
    print(f'STARTING EXPERIMENT A: UNTUNED LOSS-MATCHED RETRAINING ({len(exp_a_configs)} configs x {len(exp_a_seeds)} seeds)')
    print('=' * 80)'''

assert old_c10_start in cell10_src, "Target start in Cell 10 not found!"
cell10_src = cell10_src.replace(old_c10_start, new_c10_start)

# Add elapsed timing recording at the end of cell 10
cell10_src = cell10_src.rstrip() + "\n    STAGE_TIMINGS['A'] = time.time() - t0_stage_a\n"
nb['cells'][10]['source'] = [line + '\n' for line in cell10_src.splitlines()]

# ------------------------------------------------------------------------------
# 4. Cell 12 (Stage A_TUNED): Timing, header comments, write header into CSV
# ------------------------------------------------------------------------------
cell12_src = ''.join(nb['cells'][12]['source'])

old_c12_comment = "# 8 combos per model, spanning identical ranges [input_size: 20-50, lr: 3e-4 to 1e-3]"
new_c12_comment = """# v7 A_TUNED Grid (8 combos per model): input_size in {20, 50} x lr in {1e-3, 7.5e-4, 5e-4, 3e-4}
# Loss fixed to MSE for NBEATSx, MAE for TFT.
# Difference from v6 Stage B: v6 varied loss {MAE, MSE} across 2 lr levels {1e-3, 3e-4};
# v7 fixes loss to match target evaluation loss and densifies lr exploration (4 rates vs 2 rates in v6).
# v7 is a loss-matched densified grid, NOT the same grid as v6."""

assert old_c12_comment in cell12_src, "Grid comment in Cell 12 not found!"
cell12_src = cell12_src.replace(old_c12_comment, new_c12_comment)

old_c12_start = '''if STAGE in ['A_TUNED', 'ALL', 'SMOKE']:
    print('=' * 80)'''

new_c12_start = '''if STAGE in ['A_TUNED', 'ALL', 'SMOKE']:
    t0_stage_atuned = time.time()
    print('=' * 80)'''

assert old_c12_start in cell12_src, "Start block in Cell 12 not found!"
cell12_src = cell12_src.replace(old_c12_start, new_c12_start)

# Save grid with documentation header into a_tuned_grid_evaluation.csv
old_c12_save = '''        df_grid = pd.DataFrame(grid_records)
        atomic_save_csv(df_grid, grid_eval_file)'''

new_c12_save = '''        df_grid = pd.DataFrame(grid_records)
        grid_header_comment = """# RSIC v7 A_TUNED Hyperparameter Tuning Grid Evaluation
# v7 Grid Definition : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4} | Loss fixed to MSE (NBEATSx) / MAE (TFT) (8 combos per model)
# v6 Grid Definition : input_size in {20, 50} x learning_rate in {3e-4, 1e-3} x loss in {MAE, MSE} (8 combos per model)
# Grid Difference vs v6: v7 fixes loss to match target evaluation loss and densifies learning rates across 4 levels spanning [3e-4, 1e-3], whereas v6 varied loss over 2 learning rates. v7 is a loss-matched densified grid, NOT the same grid as v6.
"""
        tmp_grid_path = f"{grid_eval_file}.tmp_{os.getpid()}_{random.randint(1000, 9999)}"
        with open(tmp_grid_path, 'w') as f_g:
            f_g.write(grid_header_comment)
            df_grid.to_csv(f_g, index=False)
        os.replace(tmp_grid_path, grid_eval_file)'''

assert old_c12_save in cell12_src, "atomic_save_csv(df_grid) in Cell 12 not found!"
cell12_src = cell12_src.replace(old_c12_save, new_c12_save)

# Add elapsed timing recording at the end of cell 12
cell12_src = cell12_src.rstrip() + "\n    STAGE_TIMINGS['A_TUNED'] = time.time() - t0_stage_atuned\n"
nb['cells'][12]['source'] = [line + '\n' for line in cell12_src.splitlines()]

# ------------------------------------------------------------------------------
# 5. Cell 14 (Stage B): Record stage elapsed time
# ------------------------------------------------------------------------------
cell14_src = ''.join(nb['cells'][14]['source'])
old_c14_start = '''if STAGE in ['B', 'ALL', 'SMOKE']:
    print('=' * 80)
    print(f'STARTING EXPERIMENT B: 60-DAY EXPANDING-WINDOW REFIT ({len(blocks)} blocks, Seed 42)')
    print('=' * 80)'''

new_c14_start = '''if STAGE in ['B', 'ALL', 'SMOKE']:
    t0_stage_b = time.time()
    print('=' * 80)
    print(f'STARTING EXPERIMENT B: 60-DAY EXPANDING-WINDOW REFIT ({len(blocks)} blocks, Seed 42)')
    print('=' * 80)'''

assert old_c14_start in cell14_src, "Start block in Cell 14 not found!"
cell14_src = cell14_src.replace(old_c14_start, new_c14_start)

# Add elapsed timing recording at the end of cell 14
cell14_src = cell14_src.rstrip() + "\n    STAGE_TIMINGS['B'] = time.time() - t0_stage_b\n"
nb['cells'][14]['source'] = [line + '\n' for line in cell14_src.splitlines()]

# ------------------------------------------------------------------------------
# 6. Cell 18: Build A & A_TUNED ensembles only when all 5 seeds exist, else partial
# ------------------------------------------------------------------------------
cell18_src = ''.join(nb['cells'][18]['source'])

old_c18_expa = '''    if len(seed_dfs) > 0:
        base_df = seed_dfs[0][['cutoff', 'real_date', 'unique_id', 'ds', 'y']].copy()
        pred_matrix = np.column_stack([sdf[m_name].values for sdf in seed_dfs])
        base_df['prediction'] = pred_matrix.mean(axis=1)
        base_df['Config'] = cfg_label
        base_df['Stage'] = 'A'
        ensemble_rows.append(base_df)
        print(f"  Ensemble Exp A: {cfg_label:<24} averaged across {len(seed_dfs)} seeds ({len(base_df)} rows)")'''

new_c18_expa = '''    if len(seed_dfs) > 0:
        base_df = seed_dfs[0][['cutoff', 'real_date', 'unique_id', 'ds', 'y']].copy()
        pred_matrix = np.column_stack([sdf[m_name].values for sdf in seed_dfs])
        base_df['prediction'] = pred_matrix.mean(axis=1)
        if len(seed_dfs) == 5:
            base_df['Config'] = cfg_label
            print(f"  Ensemble Exp A: {cfg_label:<24} averaged across 5 seeds ({len(base_df)} rows)")
        else:
            partial_cfg_name = f"{cfg_label}_partial{len(seed_dfs)}seeds"
            base_df['Config'] = partial_cfg_name
            PARTIAL_ENSEMBLES.append(f"{partial_cfg_name} ({len(seed_dfs)}/5 seeds completed)")
            print(f"  ⚠️ Partial Ensemble Exp A: {partial_cfg_name:<30} averaged across {len(seed_dfs)}/5 seeds ({len(base_df)} rows)")
        base_df['Stage'] = 'A'
        ensemble_rows.append(base_df)'''

assert old_c18_expa in cell18_src, "Exp A block in Cell 18 not found!"
cell18_src = cell18_src.replace(old_c18_expa, new_c18_expa)

old_c18_expatuned = '''        if len(seed_dfs) > 0:
            base_df = seed_dfs[0][['cutoff', 'real_date', 'unique_id', 'ds', 'y']].copy()
            pred_matrix = np.column_stack([sdf[m_name].values for sdf in seed_dfs])
            base_df['prediction'] = pred_matrix.mean(axis=1)
            base_df['Config'] = cfg_label
            base_df['Stage'] = 'A_TUNED'
            ensemble_rows.append(base_df)
            print(f"  Ensemble Exp A_TUNED: {cfg_label:<24} averaged across {len(seed_dfs)} seeds ({len(base_df)} rows)")'''

new_c18_expatuned = '''        if len(seed_dfs) > 0:
            base_df = seed_dfs[0][['cutoff', 'real_date', 'unique_id', 'ds', 'y']].copy()
            pred_matrix = np.column_stack([sdf[m_name].values for sdf in seed_dfs])
            base_df['prediction'] = pred_matrix.mean(axis=1)
            if len(seed_dfs) == 5:
                base_df['Config'] = cfg_label
                print(f"  Ensemble Exp A_TUNED: {cfg_label:<24} averaged across 5 seeds ({len(base_df)} rows)")
            else:
                partial_cfg_name = f"{cfg_label}_partial{len(seed_dfs)}seeds"
                base_df['Config'] = partial_cfg_name
                PARTIAL_ENSEMBLES.append(f"{partial_cfg_name} ({len(seed_dfs)}/5 seeds completed)")
                print(f"  ⚠️ Partial Ensemble Exp A_TUNED: {partial_cfg_name:<30} averaged across {len(seed_dfs)}/5 seeds ({len(base_df)} rows)")
            base_df['Stage'] = 'A_TUNED'
            ensemble_rows.append(base_df)'''

assert old_c18_expatuned in cell18_src, "Exp A_TUNED block in Cell 18 not found!"
cell18_src = cell18_src.replace(old_c18_expatuned, new_c18_expatuned)
nb['cells'][18]['source'] = [line + '\n' for line in cell18_src.splitlines()]

# ------------------------------------------------------------------------------
# 7. Cell 20: Baseline discovery via find_file, loud banner, MISSING BASELINE in README, error on full STAGE
# ------------------------------------------------------------------------------
cell20_src = ''.join(nb['cells'][20]['source'])

old_c20_load = '''ensemble_csv = os.path.join(OUTPUT_DIR, 'v7_ensemble_predictions.csv')

if os.path.exists(ensemble_csv):
    df_ens = pd.read_csv(ensemble_csv)
    df_ens['cutoff'] = df_ens['cutoff'].astype(str)
    df_ens['real_date'] = df_ens['real_date'].astype(str)
    df_ens['h'] = df_ens.groupby(['cutoff', 'Config']).cumcount() + 1
    
    # Load benchmarks
    har_h1_file = find_file('har_preds_T0_h1_v2.csv', [FALLBACK_INPUT_DIR, DATA_DIR])
    har_h10_file = find_file('har_preds_T0_h10_v2.csv', [FALLBACK_INPUT_DIR, DATA_DIR])
    garch_file = find_file('GARCH_Predictions_V4.csv', [FALLBACK_INPUT_DIR, DATA_DIR])
    
    har_h1 = pd.read_csv(har_h1_file).sort_values('Date').reset_index(drop=True)
    har_h10 = pd.read_csv(har_h10_file).sort_values('Date').reset_index(drop=True)
    garch_df = pd.read_csv(garch_file)
    garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
    garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
    garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)
    
    # Optional v6 baseline predictions
    v6_file = None
    for cand_v6 in [os.path.join(BASE_DIR, 'results', 'v6', 'v6_ensemble_predictions.csv'), os.path.join(FALLBACK_INPUT_DIR, 'v6_ensemble_predictions.csv')]:
        if os.path.exists(cand_v6):
            v6_file = cand_v6
            break
    df_v6 = pd.read_csv(v6_file) if v6_file else None
    if df_v6 is not None:
        df_v6['cutoff'] = df_v6['cutoff'].astype(str)
        df_v6['real_date'] = df_v6['real_date'].astype(str)
        df_v6['h'] = df_v6.groupby(['cutoff', 'Config']).cumcount() + 1'''

new_c20_load = '''t0_stage_post = time.time()
ensemble_csv = os.path.join(OUTPUT_DIR, 'v7_ensemble_predictions.csv')

if os.path.exists(ensemble_csv):
    df_ens = pd.read_csv(ensemble_csv)
    df_ens['cutoff'] = df_ens['cutoff'].astype(str)
    df_ens['real_date'] = df_ens['real_date'].astype(str)
    df_ens['h'] = df_ens.groupby(['cutoff', 'Config']).cumcount() + 1
    
    # Discover required baseline files via find_file (searching configured dirs and /kaggle/input)
    candidate_baseline_dirs = [DATA_DIR, FALLBACK_INPUT_DIR, os.path.join(BASE_DIR, 'results', 'v6'), os.path.join(BASE_DIR, 'inputs', 'v5')]
    baseline_targets = {
        'har_h1': 'har_preds_T0_h1_v2.csv',
        'har_h10': 'har_preds_T0_h10_v2.csv',
        'garch': 'GARCH_Predictions_V4.csv',
        'v6': 'v6_ensemble_predictions.csv'
    }
    baseline_paths = {}
    
    for b_key, b_fname in baseline_targets.items():
        try:
            p = find_file(b_fname, candidate_baseline_dirs)
            baseline_paths[b_key] = p
            print(f"✅ Found baseline {b_fname:<30}: {p}")
        except FileNotFoundError:
            baseline_paths[b_key] = None
            if b_fname not in MISSING_BASELINES:
                MISSING_BASELINES.append(b_fname)
            print(f"❌ MISSING baseline {b_fname:<30}: Not found!")

    if MISSING_BASELINES:
        print("!" * 88)
        print("CRITICAL ERROR: MISSING BASELINE PREDICTION FILES REQUIRED FOR POST-PROCESSING")
        for mbf in MISSING_BASELINES:
            print(f"  - MISSING BASELINE: {mbf}")
        print("!" * 88)
        
        # In full stages, stop immediately before post-processing starts (no silent df_v6 = None)
        if STAGE in ['A_TUNED', 'B', 'ALL']:
            raise FileNotFoundError(f"Missing required baseline files for post-processing in STAGE {STAGE}: {MISSING_BASELINES}. Ensure all 4 baseline files are uploaded to Kaggle dataset.")

    har_h1 = pd.read_csv(baseline_paths['har_h1']).sort_values('Date').reset_index(drop=True) if baseline_paths['har_h1'] else None
    har_h10 = pd.read_csv(baseline_paths['har_h10']).sort_values('Date').reset_index(drop=True) if baseline_paths['har_h10'] else None
    
    if baseline_paths['garch']:
        garch_df = pd.read_csv(baseline_paths['garch'])
        garch_df['h'] = garch_df.groupby('cutoff').cumcount() + 1
        garch_h1 = garch_df[garch_df['h'] == 1].sort_values('real_date').reset_index(drop=True)
        garch_h10 = garch_df[garch_df['h'] == 10].sort_values('real_date').reset_index(drop=True)
    else:
        garch_h1 = garch_h10 = None
        
    df_v6 = pd.read_csv(baseline_paths['v6']) if baseline_paths['v6'] else None
    if df_v6 is not None:
        df_v6['cutoff'] = df_v6['cutoff'].astype(str)
        df_v6['real_date'] = df_v6['real_date'].astype(str)
        df_v6['h'] = df_v6.groupby(['cutoff', 'Config']).cumcount() + 1'''

assert old_c20_load in cell20_src, "Baseline load block in Cell 20 not found!"
cell20_src = cell20_src.replace(old_c20_load, new_c20_load)

# Add elapsed timing recording at the end of cell 20
cell20_src = cell20_src.rstrip() + "\n    STAGE_TIMINGS['POSTPROCESSING'] = time.time() - t0_stage_post\n"
nb['cells'][20]['source'] = [line + '\n' for line in cell20_src.splitlines()]

# ------------------------------------------------------------------------------
# 8. Cell 22: Notebook source copy, notebook_source_MISSING.txt, README details, stage timings
# ------------------------------------------------------------------------------
cell22_src = ''.join(nb['cells'][22]['source'])

old_c22_all = '''# 1. WRITE README.TXT FIRST (Before Manifest)
readme_path = os.path.join(OUTPUT_DIR, 'README.txt')
readme_content = f"""================================================================================
RSIC Volatility Forecasting Audit (v7)
Loss-Matched Retraining & 60-Day Periodic Refitting
================================================================================
Execution Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
Active STAGE         : {STAGE}
Hardware Device      : {gpu_name} (CUDA={gpu_available})
NeuralForecast Ver   : {nf_ver}
Evaluated Seeds      : {SEEDS_BANNER}
Test Cutoffs Target  : {N_TEST_CUTOFFS} cutoffs
Runtime Budget Limit : {MAX_RUNTIME_HOURS} hours
Total Elapsed Time   : {(time.time() - T_GLOBAL_START)/3600:.2f} hours

Run Completion Summary:
  - Total Expected Runs : {len(expected_runs)}
  - Finished Runs Count : {len(finished_runs)}
  - Missing Runs Count  : {len(missing_runs)}

Finished Runs:
{chr(10).join(['  + ' + r for r in finished_runs]) if finished_runs else '  (None)'}

Missing Runs:
{chr(10).join(['  - ' + r for r in missing_runs]) if missing_runs else '  (None - All expected runs completed successfully)'}

Measured Fit Timing:
  - Average Fit Time : {avg_fit_sec:.2f} seconds per fit
{smoke_estimates_text}
================================================================================
"""
with open(readme_path, 'w') as f_rm:
    f_rm.write(readme_content)
print(f"✅ README.txt ditulis pertama: {readme_path}")

# 2. Save copy of notebook source into output directory
notebook_copy_path = os.path.join(OUTPUT_DIR, 'notebook_source.ipynb')
candidate_notebooks = [
    os.path.abspath('08_dl_v7_audit_loss_refit.ipynb'),
    os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb'),
    os.path.abspath('../08_dl_v7_audit_loss_refit.ipynb')
]
nb_source_found = False
for c_nb in candidate_notebooks:
    if os.path.exists(c_nb):
        shutil.copy2(c_nb, notebook_copy_path)
        nb_source_found = True
        break
if not nb_source_found:
    with open(notebook_copy_path, 'w') as f_dummy:
        f_dummy.write('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}')
print(f"✅ Salinan notebook sumber disimpan: {notebook_copy_path}")'''

new_c22_all = '''# 1. Search and copy notebook source into output directory (Fix 1)
notebook_copy_path = os.path.join(OUTPUT_DIR, 'notebook_source.ipynb')
candidate_notebooks = [
    os.path.abspath('08_dl_v7_audit_loss_refit.ipynb'),
    os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb'),
    os.path.abspath('../08_dl_v7_audit_loss_refit.ipynb'),
    os.path.abspath('__notebook_source__.ipynb'),
    os.path.abspath('.virtual_documents/__notebook_source__.ipynb'),
    '/kaggle/working/__notebook_source__.ipynb',
    '/kaggle/working/.virtual_documents/__notebook_source__.ipynb'
]
nb_source_found = False
nb_source_file_used = None
for c_nb in candidate_notebooks:
    if os.path.exists(c_nb):
        shutil.copy2(c_nb, notebook_copy_path)
        nb_source_found = True
        nb_source_file_used = c_nb
        break

if nb_source_found:
    notebook_status_readme = f"Source Notebook Copy : Saved to notebook_source.ipynb (copied from {nb_source_file_used})"
    print(f"✅ Salinan notebook sumber disimpan: {notebook_copy_path} (from {nb_source_file_used})")
else:
    missing_txt_path = os.path.join(OUTPUT_DIR, 'notebook_source_MISSING.txt')
    with open(missing_txt_path, 'w') as f_miss:
        f_miss.write("WARNING: Notebook source file not found during packaging.\\nChecked paths:\\n" + "\\n".join(['  - ' + p for p in candidate_notebooks]) + "\\n")
    notebook_status_readme = "Source Notebook Copy : NOT FOUND! Generated notebook_source_MISSING.txt"
    print(f"⚠️ Notebook source tidak ditemukan! Dibuat: {missing_txt_path}")

# Format Stage Timings (Fix 6)
timing_lines = []
for stg_k in ['A', 'A_TUNED', 'B', 'POSTPROCESSING']:
    stg_t = STAGE_TIMINGS.get(stg_k)
    if stg_t is not None:
        timing_lines.append(f"  - Stage {stg_k:<15} : {stg_t:.2f} seconds ({stg_t/60.0:.2f} minutes)")
    else:
        timing_lines.append(f"  - Stage {stg_k:<15} : Not Run")
timing_text = "\\n".join(timing_lines)

# Format Grid Specifications (Fix 4)
grid_doc_text = """Hyperparameter Tuning Grid Specifications:
  - v7 A_TUNED Grid    : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4}
                         Loss fixed to MSE (NBEATSx) / MAE (TFT) (8 combos per model)
  - v6 Stage B Grid    : input_size in {20, 50} x learning_rate in {3e-4, 1e-3} x loss in {MAE, MSE} (8 combos per model)
  - Grid Difference    : v7 fixes the loss to match target evaluation loss and tests 4 learning rate levels
                         spanning [3e-4, 1e-3], whereas v6 varied loss across 2 learning rate levels.
                         v7 is a loss-matched densified grid, NOT the same grid as v6."""

# Missing Baselines & Partial Ensembles sections (Fix 2 & Fix 3)
missing_baselines_text = ""
if MISSING_BASELINES:
    missing_baselines_text = "\\nMissing Baseline Files for Post-Processing:\\n" + "\\n".join([f"  - MISSING BASELINE: {b}" for b in MISSING_BASELINES]) + "\\n"

partial_ensembles_text = ""
if PARTIAL_ENSEMBLES:
    partial_ensembles_text = "\\nPartial Seed Ensembles (<5 seeds):\\n" + "\\n".join([f"  - {pe}" for pe in PARTIAL_ENSEMBLES]) + "\\n"

# 2. WRITE README.TXT FIRST (Before Manifest)
readme_path = os.path.join(OUTPUT_DIR, 'README.txt')
readme_content = f"""================================================================================
RSIC Volatility Forecasting Audit (v7)
Loss-Matched Retraining & 60-Day Periodic Refitting
================================================================================
Execution Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
Active STAGE         : {STAGE}
Hardware Device      : {gpu_name} (CUDA={gpu_available})
NeuralForecast Ver   : {nf_ver}
Evaluated Seeds      : {SEEDS_BANNER}
Test Cutoffs Target  : {N_TEST_CUTOFFS} cutoffs
Runtime Budget Limit : {MAX_RUNTIME_HOURS} hours
Total Elapsed Time   : {(time.time() - T_GLOBAL_START)/3600:.2f} hours
{notebook_status_readme}

Actual Stage Elapsed Times:
{timing_text}

{grid_doc_text}
{missing_baselines_text}{partial_ensembles_text}
Run Completion Summary:
  - Total Expected Runs : {len(expected_runs)}
  - Finished Runs Count : {len(finished_runs)}
  - Missing Runs Count  : {len(missing_runs)}

Finished Runs:
{chr(10).join(['  + ' + r for r in finished_runs]) if finished_runs else '  (None)'}

Missing Runs:
{chr(10).join(['  - ' + r for r in missing_runs]) if missing_runs else '  (None - All expected runs completed successfully)'}

Measured Fit Timing:
  - Average Fit Time : {avg_fit_sec:.2f} seconds per fit
{smoke_estimates_text}
================================================================================
"""
with open(readme_path, 'w') as f_rm:
    f_rm.write(readme_content)
print(f"✅ README.txt ditulis pertama: {readme_path}")'''

assert old_c22_all in cell22_src, "README and notebook copy block in Cell 22 not found!"
cell22_src = cell22_src.replace(old_c22_all, new_c22_all)
nb['cells'][22]['source'] = [line + '\n' for line in cell22_src.splitlines()]

# Save modified notebook IN PLACE
with open(NB_PATH, 'w') as f_out:
    json.dump(nb, f_out, indent=1)

print(f"SUCCESS: Successfully updated {NB_PATH} in place.")
