#!/usr/bin/env python3
"""
Update script for notebooks/08_dl_v7_audit_loss_refit.ipynb:
1. Update NBEATSx grid to 12 combos (input {20, 50, 100} x lr {1e-3, 5e-4, 3e-4, 1e-4}, MSE only).
2. Rank A_TUNED by mean validation MSE across seeds [42, 7] (both NBEATSx and TFT).
3. Set NBEATSx tuning and winner val_check_steps = 5, patience = 25 (scaled for 125 steps patience).
4. Print grid edge status (input 100 or lr 1e-4) and record in README.txt.
5. Write v6 Stage B 8 NBEATSx combos side-by-side into README.txt.
6. Update runtime estimates for 12 NBEATSx combos + 8 TFT combos x 2 seeds.
7. Ensure candidate_baseline_dirs includes /kaggle/input/datasets/ferastoro/spillover-v5-inputs.
"""

import os
import json

NB_PATH = os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb')

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

# ------------------------------------------------------------------------------
# 1. Update Cell 12: Grid definitions, 2-seed evaluation, val_check_steps=5 for NBEATSx, edge check
# ------------------------------------------------------------------------------
cell12_src = ''.join(nb['cells'][12]['source'])

old_grid_block = """TUNING_GRIDS = {
    'NBEATSx': [
        {'input_size': 20, 'learning_rate': 1e-3, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 7.5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 3e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 1e-3, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 7.5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 3e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
    ],
    'TFT': [
        {'input_size': 20, 'learning_rate': 1e-3, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 7.5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 3e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 1e-3, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 7.5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 3e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
    ]
}"""

new_grid_block = """# v7 A_TUNED Grids:
# - NBEATSx (12 combos): input_size {20, 50, 100} x lr {1e-3, 5e-4, 3e-4, 1e-4}, MSE loss only.
# - TFT (8 combos)    : input_size {20, 50} x lr {1e-3, 7.5e-4, 5e-4, 3e-4}, MAE loss only.
# Both models ranked on 2-seed validation MSE average (seeds 42 and 7, over h=1 and h=10).
TUNING_GRIDS = {
    'NBEATSx': [
        {'input_size': 20, 'learning_rate': 1e-3, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 3e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 20, 'learning_rate': 1e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 1e-3, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 3e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 50, 'learning_rate': 1e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 100, 'learning_rate': 1e-3, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 100, 'learning_rate': 5e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 100, 'learning_rate': 3e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
        {'input_size': 100, 'learning_rate': 1e-4, 'mlp_units': [[256, 256], [256, 256], [256, 256]], 'loss_name': 'MSE'},
    ],
    'TFT': [
        {'input_size': 20, 'learning_rate': 1e-3, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 7.5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 20, 'learning_rate': 3e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 1e-3, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 7.5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 5e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
        {'input_size': 50, 'learning_rate': 3e-4, 'hidden_size': 64, 'loss_name': 'MAE'},
    ]
}"""

assert old_grid_block in cell12_src, "Grid block in Cell 12 not found!"
cell12_src = cell12_src.replace(old_grid_block, new_grid_block)

# Update the tuning loop in Cell 12 to evaluate seeds [42, 7] and apply val_check_steps=5 for NBEATSx
old_tuning_loop = '''    grid_records = []
    best_tuned_hp = {}
    
    for m_name in a_tuned_models:
        grid = TUNING_GRIDS[m_name]
        if STAGE == 'SMOKE':
            grid = [grid[0]]  # Single combo dry-run in smoke
            
        fset_cols = FEATURE_SETS['NOSPILL']
        print(f"\\n--- Tuning Grid for {m_name} ({len(grid)} combos, Seed 42) ---")
        
        min_grid_input = min(c['input_size'] for c in grid)
        max_grid_input = max(c['input_size'] for c in grid)
        min_grid_lr = min(c['learning_rate'] for c in grid)
        max_grid_lr = max(c['learning_rate'] for c in grid)
        
        for idx, hp_cand in enumerate(grid):
            combo_id = f"{m_name}_tuned_cfg{idx+1}"
            loss_choice = hp_cand['loss_name']
            extra_kw = {k: v for k, v in hp_cand.items() if k not in ['input_size', 'loss_name']}
            curve_file = os.path.join(LOGS_DIR, f'val_curve_tuning_{combo_id}.csv')
            
            set_all_seeds(42)
            model = get_model_instance(
                model_name=m_name,
                fset_cols=fset_cols,
                h=H,
                input_size=hp_cand['input_size'],
                max_steps=MAX_STEPS_BANNER,
                val_check_steps=VAL_CHECK_STEPS_BANNER,
                patience=PATIENCE_BANNER,
                seed=42,
                loss_name=loss_choice,
                extra_kwargs=extra_kw,
                log_curve_file=curve_file
            )
            
            nf_sub = df_train_val[['unique_id', 'ds', 'y'] + fset_cols].copy()
            nf = NeuralForecast(models=[model], freq='D')
            
            t0 = time.time()
            nf.fit(df=nf_sub, val_size=n_val)
            t_fit = time.time() - t0
            a_tuned_measured_fit_times.append(t_fit)
            
            m_fitted = nf.models[0]
            chk_before = get_weights_checksum(m_fitted)
            tr_list = [cb for cb in m_fitted.trainer_kwargs.get('callbacks', []) if isinstance(cb, RestoreBestWeightsCallback)]
            tr = tr_list[0] if tr_list else None
            
            n_val_windows = n_val - H + 1
            cv_val = nf.cross_validation(
                df=nf_sub,
                n_windows=n_val_windows,
                step_size=1,
                use_fitted=True
            ).reset_index(drop=True)
            
            chk_after = get_weights_checksum(m_fitted)
            assert chk_before == chk_after, f"ASSERT STOP: Weights changed during cv_val in {combo_id}!"
            
            grp_v1 = cv_val.groupby('cutoff').head(1)
            grp_v10 = cv_val.groupby('cutoff').tail(1)
            mse_v1 = float(mean_squared_error(grp_v1['y'], grp_v1[m_name]))
            mse_v10 = float(mean_squared_error(grp_v10['y'], grp_v10[m_name]))
            mse_val_avg = (mse_v1 + mse_v10) / 2.0
            
            val_mean_target = float(cv_val['y'].mean())
            val_mean_pred = float(cv_val[m_name].mean())
            val_ratio = val_mean_pred / val_mean_target if val_mean_target > 0 else np.nan
            
            ratio_flag = "OK"
            if val_ratio > 2.0:
                ratio_flag = "FLAG_OVERSHOOT (>2x)"
            elif val_ratio < 0.5:
                ratio_flag = "FLAG_UNDERSHOOT (<0.5x)"
                
            rec = {
                'model': m_name,
                'combo_id': combo_id,
                'input_size': hp_cand['input_size'],
                'learning_rate': hp_cand['learning_rate'],
                'loss_name': loss_choice,
                'hyperparameters': json.dumps(hp_cand),
                'best_step': tr.best_step if tr else np.nan,
                'stop_step': tr.actual_final_step if tr else np.nan,
                'val_mse_h1': mse_v1,
                'val_mse_h10': mse_v10,
                'val_mse_avg': mse_val_avg,
                'val_mean_target': val_mean_target,
                'val_mean_pred': val_mean_pred,
                'val_forecast_to_target_ratio': val_ratio,
                'ratio_flag': ratio_flag,
                'fit_seconds': round(t_fit, 2)
            }
            grid_records.append(rec)
            print(f"  [{combo_id}] Inp={hp_cand['input_size']} LR={hp_cand['learning_rate']:.1e} | BestStep={tr.best_step:<4} | ValMSE={mse_val_avg:.6e} | Ratio={val_ratio:.2f} ({ratio_flag})")
            
        df_grid = pd.DataFrame(grid_records)
        grid_header_comment = """# RSIC v7 A_TUNED Hyperparameter Tuning Grid Evaluation
# v7 Grid Definition : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4} | Loss fixed to MSE (NBEATSx) / MAE (TFT) (8 combos per model)
# v6 Grid Definition : input_size in {20, 50} x learning_rate in {3e-4, 1e-3} x loss in {MAE, MSE} (8 combos per model)
# Grid Difference vs v6: v7 fixes loss to match target evaluation loss and densifies learning rates across 4 levels spanning [3e-4, 1e-3], whereas v6 varied loss over 2 learning rates. v7 is a loss-matched densified grid, NOT the same grid as v6.
"""
        tmp_grid_path = f"{grid_eval_file}.tmp_{os.getpid()}_{random.randint(1000, 9999)}"
        with open(tmp_grid_path, 'w') as f_g:
            f_g.write(grid_header_comment)
            df_grid.to_csv(f_g, index=False)
        os.replace(tmp_grid_path, grid_eval_file)
        
        df_m = df_grid[df_grid['model'] == m_name].sort_values('val_mse_avg').reset_index(drop=True)
        winner = df_m.iloc[0]
        winner_hp = json.loads(winner['hyperparameters'])
        best_tuned_hp[m_name] = winner_hp
        
        is_edge_input = (winner_hp['input_size'] in [min_grid_input, max_grid_input])
        is_edge_lr = (winner_hp['learning_rate'] in [min_grid_lr, max_grid_lr])
        is_edge = is_edge_input or is_edge_lr
        
        print(f"\\n🏆 Winner for {m_name}: {winner['combo_id']}")
        print(f"   Hyperparameters: {winner_hp}")
        print(f"   Val MSE Avg    : {winner['val_mse_avg']:.6e} | Ratio: {winner['val_forecast_to_target_ratio']:.2f}")
        print(f"   Grid Edge Check: {'ON THE EDGE' if is_edge else 'INTERIOR'} (input_size={winner_hp['input_size']} [range: {min_grid_input}-{max_grid_input}, edge={is_edge_input}], lr={winner_hp['learning_rate']} [range: {min_grid_lr}-{max_grid_lr}, edge={is_edge_lr}])")'''

new_tuning_loop = '''    grid_records = []
    best_tuned_hp = {}
    winner_edge_notes = {}
    
    stage_b_tuning_seeds = [42, 7] if STAGE != 'SMOKE' else [42]
    
    for m_name in a_tuned_models:
        grid = TUNING_GRIDS[m_name]
        if STAGE == 'SMOKE':
            grid = [grid[0]]  # Single combo dry-run in smoke
            
        fset_cols = FEATURE_SETS['NOSPILL']
        
        # Hyperparameter check steps and patience configuration:
        # For NBEATSx: val_check_steps = 5 (v6 used 25).
        # Scaled patience = 25 checks (25 * 5 = 125 steps), keeping the exact same 125-step patience as v6 (5 * 25 = 125).
        # For TFT: keeps default VAL_CHECK_STEPS_BANNER (25) and PATIENCE_BANNER (5).
        if m_name == 'NBEATSx':
            tune_val_check_steps = 5
            tune_patience = 25 if STAGE != 'SMOKE' else 4
        else:
            tune_val_check_steps = VAL_CHECK_STEPS_BANNER
            tune_patience = PATIENCE_BANNER
            
        print(f"\\n--- Tuning Grid for {m_name} ({len(grid)} combos, Seeds {stage_b_tuning_seeds}, val_check_steps={tune_val_check_steps}, patience={tune_patience}) ---")
        
        min_grid_input = min(c['input_size'] for c in grid)
        max_grid_input = max(c['input_size'] for c in grid)
        min_grid_lr = min(c['learning_rate'] for c in grid)
        max_grid_lr = max(c['learning_rate'] for c in grid)
        
        for idx, hp_cand in enumerate(grid):
            combo_id = f"{m_name}_tuned_cfg{idx+1}"
            loss_choice = hp_cand['loss_name']
            extra_kw = {k: v for k, v in hp_cand.items() if k not in ['input_size', 'loss_name']}
            
            seed_val_mses = []
            seed_val_ratios = []
            seed_best_steps = []
            seed_stop_steps = []
            combo_fit_secs = 0.0
            
            for t_seed in stage_b_tuning_seeds:
                curve_file = os.path.join(LOGS_DIR, f'val_curve_tuning_{combo_id}_s{t_seed}.csv')
                set_all_seeds(t_seed)
                model = get_model_instance(
                    model_name=m_name,
                    fset_cols=fset_cols,
                    h=H,
                    input_size=hp_cand['input_size'],
                    max_steps=MAX_STEPS_BANNER,
                    val_check_steps=tune_val_check_steps,
                    patience=tune_patience,
                    seed=t_seed,
                    loss_name=loss_choice,
                    extra_kwargs=extra_kw,
                    log_curve_file=curve_file
                )
                
                nf_sub = df_train_val[['unique_id', 'ds', 'y'] + fset_cols].copy()
                nf = NeuralForecast(models=[model], freq='D')
                
                t0 = time.time()
                nf.fit(df=nf_sub, val_size=n_val)
                t_fit = time.time() - t0
                combo_fit_secs += t_fit
                a_tuned_measured_fit_times.append(t_fit)
                
                m_fitted = nf.models[0]
                chk_before = get_weights_checksum(m_fitted)
                tr_list = [cb for cb in m_fitted.trainer_kwargs.get('callbacks', []) if isinstance(cb, RestoreBestWeightsCallback)]
                tr = tr_list[0] if tr_list else None
                
                n_val_windows = n_val - H + 1
                cv_val = nf.cross_validation(
                    df=nf_sub,
                    n_windows=n_val_windows,
                    step_size=1,
                    use_fitted=True
                ).reset_index(drop=True)
                
                chk_after = get_weights_checksum(m_fitted)
                assert chk_before == chk_after, f"ASSERT STOP: Weights changed during cv_val in {combo_id}_s{t_seed}!"
                
                grp_v1 = cv_val.groupby('cutoff').head(1)
                grp_v10 = cv_val.groupby('cutoff').tail(1)
                mse_v1 = float(mean_squared_error(grp_v1['y'], grp_v1[m_name]))
                mse_v10 = float(mean_squared_error(grp_v10['y'], grp_v10[m_name]))
                mse_val_seed_avg = (mse_v1 + mse_v10) / 2.0
                
                val_mean_target = float(cv_val['y'].mean())
                val_mean_pred = float(cv_val[m_name].mean())
                val_ratio = val_mean_pred / val_mean_target if val_mean_target > 0 else np.nan
                
                seed_val_mses.append(mse_val_seed_avg)
                seed_val_ratios.append(val_ratio)
                seed_best_steps.append(tr.best_step if tr else np.nan)
                seed_stop_steps.append(tr.actual_final_step if tr else np.nan)
            
            # 2-Seed Mean Validation MSE (averaging over seeds 42 and 7 as in v6 Stage B)
            val_mse_avg_2seeds = float(np.mean(seed_val_mses))
            mean_val_ratio = float(np.mean(seed_val_ratios))
            
            ratio_flag = "OK"
            if mean_val_ratio > 2.0:
                ratio_flag = "FLAG_OVERSHOOT (>2x)"
            elif mean_val_ratio < 0.5:
                ratio_flag = "FLAG_UNDERSHOOT (<0.5x)"
                
            rec = {
                'model': m_name,
                'combo_id': combo_id,
                'input_size': hp_cand['input_size'],
                'learning_rate': hp_cand['learning_rate'],
                'loss_name': loss_choice,
                'val_check_steps': tune_val_check_steps,
                'patience': tune_patience,
                'hyperparameters': json.dumps(hp_cand),
                'best_step_s42': seed_best_steps[0] if len(seed_best_steps) > 0 else np.nan,
                'best_step_s7': seed_best_steps[1] if len(seed_best_steps) > 1 else np.nan,
                'val_mse_s42': seed_val_mses[0] if len(seed_val_mses) > 0 else np.nan,
                'val_mse_s7': seed_val_mses[1] if len(seed_val_mses) > 1 else np.nan,
                'val_mse_avg': val_mse_avg_2seeds,
                'val_forecast_to_target_ratio': mean_val_ratio,
                'ratio_flag': ratio_flag,
                'fit_seconds_total': round(combo_fit_secs, 2)
            }
            grid_records.append(rec)
            print(f"  [{combo_id}] Inp={hp_cand['input_size']} LR={hp_cand['learning_rate']:.1e} | 2-Seed ValMSE={val_mse_avg_2seeds:.6e} (s42={rec['val_mse_s42']:.6e}, s7={rec['val_mse_s7']:.6e}) | Ratio={mean_val_ratio:.2f} ({ratio_flag})")
            
        df_grid = pd.DataFrame(grid_records)
        grid_header_comment = """# RSIC v7 A_TUNED Hyperparameter Tuning Grid Evaluation
# v7 NBEATSx Grid : input_size in {20, 50, 100} x learning_rate in {1e-3, 5e-4, 3e-4, 1e-4} | Loss fixed to MSE (12 combos)
# v7 TFT Grid     : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4} | Loss fixed to MAE (8 combos)
# Selection Metric: Ranked by mean validation MSE across seeds 42 and 7 (averaged over h=1 and h=10).
# v6 Grid Diff    : v6 used {20, 50} x {3e-4, 1e-3} x {MAE, MSE}; v7 fixes loss and explores larger/finer grid with val_check_steps=5, patience=25 for NBEATSx.
"""
        tmp_grid_path = f"{grid_eval_file}.tmp_{os.getpid()}_{random.randint(1000, 9999)}"
        with open(tmp_grid_path, 'w') as f_g:
            f_g.write(grid_header_comment)
            df_grid.to_csv(f_g, index=False)
        os.replace(tmp_grid_path, grid_eval_file)
        
        df_m = df_grid[df_grid['model'] == m_name].sort_values('val_mse_avg').reset_index(drop=True)
        winner = df_m.iloc[0]
        winner_hp = json.loads(winner['hyperparameters'])
        # Store tuned trainer kwargs into winner hyperparameters
        winner_hp['val_check_steps'] = tune_val_check_steps
        winner_hp['patience'] = tune_patience
        best_tuned_hp[m_name] = winner_hp
        
        # Edge checks (specifically for input 100 or lr 1e-4)
        is_edge_input_100 = (winner_hp['input_size'] == 100)
        is_edge_lr_1e4 = (winner_hp['learning_rate'] == 1e-4)
        is_edge_input = (winner_hp['input_size'] in [min_grid_input, max_grid_input])
        is_edge_lr = (winner_hp['learning_rate'] in [min_grid_lr, max_grid_lr])
        is_edge = is_edge_input or is_edge_lr
        
        edge_notes = []
        if is_edge_input_100:
            edge_notes.append("input_size=100 (upper bound)")
        elif winner_hp['input_size'] == min_grid_input:
            edge_notes.append(f"input_size={min_grid_input} (lower bound)")
        if is_edge_lr_1e4:
            edge_notes.append("learning_rate=1e-4 (lower bound)")
        elif winner_hp['learning_rate'] == max_grid_lr:
            edge_notes.append(f"learning_rate={max_grid_lr} (upper bound)")
            
        edge_status_str = f"ON THE EDGE ({', '.join(edge_notes)})" if is_edge else "INTERIOR (neither input 100 nor lr 1e-4)"
        winner_edge_notes[m_name] = edge_status_str
        
        print(f"\\n🏆 Winner for {m_name}: {winner['combo_id']}")
        print(f"   Hyperparameters: {winner_hp}")
        print(f"   2-Seed Val MSE Avg: {winner['val_mse_avg']:.6e} | Ratio: {winner['val_forecast_to_target_ratio']:.2f}")
        print(f"   Grid Edge Check   : {edge_status_str}")'''

assert old_tuning_loop in cell12_src, "Tuning loop in Cell 12 not found!"
cell12_src = cell12_src.replace(old_tuning_loop, new_tuning_loop)

# In the 5-seed final runs of Cell 12, use winner's tuned val_check_steps and patience
old_5seed_instantiation = '''            model = get_model_instance(
                model_name=m_name,
                fset_cols=fset_cols,
                h=H,
                input_size=hp_winner['input_size'],
                max_steps=MAX_STEPS_BANNER,
                val_check_steps=VAL_CHECK_STEPS_BANNER,
                patience=PATIENCE_BANNER,
                seed=seed,
                loss_name=loss_choice,
                extra_kwargs=extra_kw,
                log_curve_file=curve_file
            )'''

new_5seed_instantiation = '''            # Winner final run uses the same val_check_steps and patience as tuning
            w_val_check_steps = hp_winner.get('val_check_steps', VAL_CHECK_STEPS_BANNER)
            w_patience = hp_winner.get('patience', PATIENCE_BANNER)
            model = get_model_instance(
                model_name=m_name,
                fset_cols=fset_cols,
                h=H,
                input_size=hp_winner['input_size'],
                max_steps=MAX_STEPS_BANNER,
                val_check_steps=w_val_check_steps,
                patience=w_patience,
                seed=seed,
                loss_name=loss_choice,
                extra_kwargs=extra_kw,
                log_curve_file=curve_file
            )'''

assert old_5seed_instantiation in cell12_src, "5-seed instantiation in Cell 12 not found!"
cell12_src = cell12_src.replace(old_5seed_instantiation, new_5seed_instantiation)

# Also update log_run_metrics in 5-seed runs to record w_val_check_steps and w_patience
old_5seed_log = "'max_steps': MAX_STEPS_BANNER,\n                'val_check_steps': VAL_CHECK_STEPS_BANNER,\n                'patience': PATIENCE_BANNER,"
new_5seed_log = "'max_steps': MAX_STEPS_BANNER,\n                'val_check_steps': w_val_check_steps,\n                'patience': w_patience,"

assert old_5seed_log in cell12_src, "5-seed log_run_metrics in Cell 12 not found!"
cell12_src = cell12_src.replace(old_5seed_log, new_5seed_log)

nb['cells'][12]['source'] = [line + '\n' for line in cell12_src.splitlines()]

# ------------------------------------------------------------------------------
# 2. Update Cell 20: Add /kaggle/input/datasets/ferastoro/spillover-v5-inputs to discovery
# ------------------------------------------------------------------------------
cell20_src = ''.join(nb['cells'][20]['source'])

old_c20_dirs = "candidate_baseline_dirs = [DATA_DIR, FALLBACK_INPUT_DIR, os.path.join(BASE_DIR, 'results', 'v6'), os.path.join(BASE_DIR, 'inputs', 'v5')]"
new_c20_dirs = "candidate_baseline_dirs = ['/kaggle/input/datasets/ferastoro/spillover-v5-inputs', '/kaggle/input/datasets/ferastoro/spillover-v5-inputs/rsic-v5-inputs', DATA_DIR, FALLBACK_INPUT_DIR, os.path.join(BASE_DIR, 'results', 'v6'), os.path.join(BASE_DIR, 'inputs', 'v5')]"

assert old_c20_dirs in cell20_src, "candidate_baseline_dirs in Cell 20 not found!"
cell20_src = cell20_src.replace(old_c20_dirs, new_c20_dirs)
nb['cells'][20]['source'] = [line + '\n' for line in cell20_src.splitlines()]

# ------------------------------------------------------------------------------
# 3. Update Cell 22: README documentation for 12-combo grid, v6 side-by-side, val_check_steps=5, runtime estimate
# ------------------------------------------------------------------------------
cell22_src = ''.join(nb['cells'][22]['source'])

old_smoke_est = '''smoke_estimates_text = ""
if STAGE == 'SMOKE':
    est_atuned_hours = (8 * avg_fit_sec * 20 + 10 * avg_fit_sec * 20) / 3600.0 if avg_fit_sec > 0 else 0.5
    est_b_hours = (24 * avg_fit_sec * 20) / 3600.0 if avg_fit_sec > 0 else 1.0
    smoke_estimates_text = f"""
Runtime Estimates from SMOKE (T4 GPU Extrapolation):
  - Measured Time per Fit (SMOKE): {avg_fit_sec:.2f} seconds
  - Estimated Total Runtime for STAGE A_TUNED: ~{est_atuned_hours:.2f} hours
  - Estimated Total Runtime for STAGE B       : ~{est_b_hours:.2f} hours
"""'''

new_smoke_est = '''smoke_estimates_text = ""
if STAGE == 'SMOKE':
    # Updated runtime estimation:
    # A_TUNED: (12 NBEATSx combos * 2 seeds + 8 TFT combos * 2 seeds = 40 tuning fits) + (20 winner 5-seed fits) = 60 fits
    # Stage B: 4 configs * 6 blocks = 24 fits
    est_atuned_hours = (40 * avg_fit_sec * 20 + 20 * avg_fit_sec * 20) / 3600.0 if avg_fit_sec > 0 else 0.6
    est_b_hours = (24 * avg_fit_sec * 20) / 3600.0 if avg_fit_sec > 0 else 0.4
    smoke_estimates_text = f"""
Runtime Estimates from SMOKE (T4 GPU Extrapolation for New Grids):
  - Measured Time per Fit (SMOKE): {avg_fit_sec:.2f} seconds
  - Estimated Total Runtime for STAGE A_TUNED (60 fits total): ~{est_atuned_hours:.2f} hours
  - Estimated Total Runtime for STAGE B       (24 fits total): ~{est_b_hours:.2f} hours
  - Estimated Total Runtime for STAGE ALL     (84 fits total): ~{est_atuned_hours + est_b_hours:.2f} hours
"""'''

assert old_smoke_est in cell22_src, "old_smoke_est in Cell 22 not found!"
cell22_src = cell22_src.replace(old_smoke_est, new_smoke_est)

# Format grid documentation in README including v6 side-by-side table and edge check
old_c22_grid_doc = '''# Format Grid Specifications (Fix 4)
grid_doc_text = """Hyperparameter Tuning Grid Specifications:
  - v7 A_TUNED Grid    : input_size in {20, 50} x learning_rate in {1e-3, 7.5e-4, 5e-4, 3e-4}
                         Loss fixed to MSE (NBEATSx) / MAE (TFT) (8 combos per model)
  - v6 Stage B Grid    : input_size in {20, 50} x learning_rate in {3e-4, 1e-3} x loss in {MAE, MSE} (8 combos per model)
  - Grid Difference    : v7 fixes the loss to match target evaluation loss and tests 4 learning rate levels
                         spanning [3e-4, 1e-3], whereas v6 varied loss across 2 learning rate levels.
                         v7 is a loss-matched densified grid, NOT the same grid as v6."""'''

new_c22_grid_doc = '''# Edge check summary from tuning
edge_info_lines = []
if 'winner_edge_notes' in locals():
    for m_k, e_v in winner_edge_notes.items():
        edge_info_lines.append(f"  - {m_k:<8} Winner Edge Status : {e_v}")
edge_info_text = "\\n".join(edge_info_lines) if edge_info_lines else "  - (Tuning edge check not executed in this run)"

grid_doc_text = f"""Hyperparameter Tuning Grid Specifications & Settings:
  - v7 NBEATSx Grid (12 combos) : input_size in {{20, 50, 100}} x learning_rate in {{1e-3, 5e-4, 3e-4, 1e-4}} | MSE loss only
  - v7 TFT Grid (8 combos)      : input_size in {{20, 50}} x learning_rate in {{1e-3, 7.5e-4, 5e-4, 3e-4}} | MAE loss only
  - Tuning Ranking Metric       : Mean validation MSE over seeds 42 and 7 (averaged over h=1 and h=10)
  
Early Stopping & Verification Settings:
  - NBEATSx Tuning & Winner : val_check_steps = 5 (v6 used 25) | patience = 25 checks (v6 used 5)
                              Total patience in steps = 125 steps (5 * 25 = 125, identical to v6: 25 * 5 = 125).
                              Restore-best weights: Enabled.
  - TFT Tuning & Winner     : val_check_steps = 25 | patience = 5 checks (125 steps). Restore-best: Enabled.

Grid Edge Verification:
{edge_info_text}

v6 Stage B Empirical Tuning Results for NBEATSx (from results/v6/v6_tuning_log.csv):
  +--------------------+------------+--------------+------------------+------------------+
  | Hyperparameters    | input_size | lr           | MAE Train ValMSE | MSE Train ValMSE |
  +--------------------+------------+--------------+------------------+------------------+
  | Combo 1 (cfg1/2)   | 20         | 0.0010       | 1.774417e-06     | 9.471862e-06     |
  | Combo 2 (cfg3/4)*  | 20         | 0.0003       | 1.772739e-06 (*) | 5.141590e-06     |
  | Combo 3 (cfg5/6)   | 50         | 0.0010       | 2.338505e-06     | 3.573679e-06     |
  | Combo 4 (cfg7/8)   | 50         | 0.0003       | 2.277654e-06     | 2.836472e-06     |
  +--------------------+------------+--------------+------------------+------------------+
  (*) Winner in v6 Stage B: NBEATSx_cfg3 (MAE loss, input 20, lr 0.0003)."""'''

assert old_c22_grid_doc in cell22_src, "old_c22_grid_doc in Cell 22 not found!"
cell22_src = cell22_src.replace(old_c22_grid_doc, new_c22_grid_doc)
nb['cells'][22]['source'] = [line + '\n' for line in cell22_src.splitlines()]

# Save modified notebook IN PLACE
with open(NB_PATH, 'w') as f_out:
    json.dump(nb, f_out, indent=1)

print(f"SUCCESS: Successfully applied all updates in place to {NB_PATH}")

