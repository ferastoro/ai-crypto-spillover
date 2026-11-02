#!/usr/bin/env python3
"""
Update notebooks/08_dl_v7_audit_loss_refit.ipynb:
In STAGE="SMOKE", make NBEATSx tuning grid run exactly 2 combos:
- First combo in the grid (input_size=20, lr=1e-3, MSE)
- Extreme corner combo (input_size=100, lr=1e-4, MSE)
TFT stays at 1 combo.
"""

import os
import json

NB_PATH = os.path.abspath('notebooks/08_dl_v7_audit_loss_refit.ipynb')

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

cell12_src = ''.join(nb['cells'][12]['source'])

old_target = '''        grid = TUNING_GRIDS[m_name]
        if STAGE == 'SMOKE':
            grid = [grid[0]]  # Single combo dry-run in smoke'''

new_target = '''        grid = TUNING_GRIDS[m_name]
        if STAGE == 'SMOKE':
            if m_name == 'NBEATSx':
                # Exactly 2 combos: first combo (input=20, lr=1e-3) and extreme corner combo (input=100, lr=1e-4)
                corner_combo = [c for c in grid if c['input_size'] == 100 and c['learning_rate'] == 1e-4 and c['loss_name'] == 'MSE']
                grid = [grid[0]] + (corner_combo if corner_combo else [grid[-1]])  # exactly 2 combos
            else:
                grid = [grid[0]]  # TFT stays at 1 combo'''

assert old_target in cell12_src, "Target in Cell 12 not found!"
cell12_src = cell12_src.replace(old_target, new_target)

# Also ensure min/max grid range checks use TUNING_GRIDS[m_name] so the full design bounds [20, 100] and [1e-4, 1e-3] are preserved in SMOKE
old_min_max = '''        min_grid_input = min(c['input_size'] for c in grid)
        max_grid_input = max(c['input_size'] for c in grid)
        min_grid_lr = min(c['learning_rate'] for c in grid)
        max_grid_lr = max(c['learning_rate'] for c in grid)'''

new_min_max = '''        full_grid_spec = TUNING_GRIDS[m_name]
        min_grid_input = min(c['input_size'] for c in full_grid_spec)
        max_grid_input = max(c['input_size'] for c in full_grid_spec)
        min_grid_lr = min(c['learning_rate'] for c in full_grid_spec)
        max_grid_lr = max(c['learning_rate'] for c in full_grid_spec)'''

assert old_min_max in cell12_src, "min/max in Cell 12 not found!"
cell12_src = cell12_src.replace(old_min_max, new_min_max)

nb['cells'][12]['source'] = [line + '\n' for line in cell12_src.splitlines()]

with open(NB_PATH, 'w') as f_out:
    json.dump(nb, f_out, indent=1)

print("SUCCESS: Updated Cell 12 in place for 2-combo NBEATSx smoke grid.")

