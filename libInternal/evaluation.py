from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import pandas as pd

def evaluateModel(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, None)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def printMetrics(metrics, model_name="Model"):
    print(f"=== {model_name} Evaluation Metrics ===")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")
    print("=====================================")

def saveEvaliationResults(model_name, metrics, output_file="modelResults.csv"):
    df = pd.DataFrame([{ "Model": model_name, **metrics }])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(project_root, "assets")
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, output_file)

    if os.path.exists(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
    else:
        df.to_csv(full_path, index=False)

    print(f"✅ Model results saved to: {full_path}")
