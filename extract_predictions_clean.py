import os
import pandas as pd
from glob import glob

# Define subtask info
subtasks = {
    'subtask1': {
        'columns': ['id', 'polarization']
    },
    'subtask2': {
        'columns': ['id', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    },
    'subtask3': {
        'columns': ['id', 'stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']
    }
}

# Output folder

base_output_dir = os.path.abspath('csv_predictions_cleaned')
os.makedirs(base_output_dir, exist_ok=True)

for subtask, info in subtasks.items():
    # Create subfolder for each subtask
    subtask_output_dir = os.path.join(base_output_dir, subtask)
    os.makedirs(subtask_output_dir, exist_ok=True)
    # Only look for CSVs in subtask/subtask/models
    csv_dir = os.path.join(subtask, subtask, 'models')
    if not os.path.exists(csv_dir):
        continue
    for root, _, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                # Determine model name (immediate subfolder under models)
                rel_path = os.path.relpath(csv_path, csv_dir)
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    model_name = parts[0]
                else:
                    # If CSV is directly under models, use 'other' as fallback
                    model_name = 'other'
                # Create model subfolder in output
                model_output_dir = os.path.join(subtask_output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                try:
                    df = pd.read_csv(csv_path)
                    # For subtask2 and subtask3, map *_pred columns to required output columns
                    if subtask in ['subtask2', 'subtask3']:
                        col_map = {}
                        for out_col in info['columns']:
                            if out_col == 'id':
                                col_map['id'] = 'id'
                            else:
                                pred_col = f"{out_col}_pred"
                                if pred_col in df.columns:
                                    col_map[pred_col] = out_col
                        # Only keep columns that exist in df
                        keep_cols = [src for src in col_map.keys() if src in df.columns]
                        df_out = df[keep_cols].rename(columns=col_map)
                    else:
                        # For subtask1, extract 'id' and 'prediction', rename 'prediction' to 'polarization'
                        if 'id' in df.columns and 'prediction' in df.columns:
                            df_out = df[['id', 'prediction']].rename(columns={'prediction': 'polarization'})
                        else:
                            # fallback: keep only columns that exist from the list
                            keep_cols = [col for col in info['columns'] if col in df.columns]
                            df_out = df[keep_cols]
                    # Save to model output folder with original filename
                    out_name = file
                    out_path = os.path.join(model_output_dir, out_name)
                    df_out.to_csv(out_path, index=False)
                    print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
