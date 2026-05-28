"""
XGBoost Model — Training & Evaluation
Predicts ED admission peaks at 24-72h horizon.

Reference: King et al. (Nature npj Digital Medicine, 2022)
           AUROC 0.82-0.90 on 109,465 UK ED visits
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
)
import shap

MODEL_PATH = Path("models/xgboost_ed_predictor.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.txt")


XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}


def train(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> xgb.XGBRegressor:
    """
    Train XGBoost with time-series cross-validation.

    Args:
        X: Feature matrix
        y: Target (daily ED admissions count)
        n_splits: Number of TimeSeriesSplit folds

    Returns:
        Trained XGBRegressor
    """
    logger.info(f"Training XGBoost on {len(X):,} samples, {X.shape[1]} features")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        cv_scores.append({"fold": fold + 1, "mae": mae, "mape": mape})
        logger.info(f"  Fold {fold+1}: MAE={mae:.1f}, MAPE={mape:.1%}")

    mean_mae = np.mean([s["mae"] for s in cv_scores])
    logger.success(f"CV Mean MAE: {mean_mae:.1f} admissions")

    # Final model on full data
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y, verbose=False)

    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text("\n".join(X.columns.tolist()))
    logger.success(f"Model saved to {MODEL_PATH}")

    return final_model


def evaluate(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model and return metrics dict."""
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "mape": mean_absolute_percentage_error(y_test, preds),
        "n_samples": len(y_test),
    }

    logger.info(f"Test MAE: {metrics['mae']:.1f} admissions ({metrics['mape']:.1%})")
    return metrics


def explain(model: xgb.XGBRegressor, X: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Generate SHAP feature importance for model explainability.
    Critical for institutional trust — decision makers need to understand WHY.
    """
    logger.info("Computing SHAP values for explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[:n_samples])

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info(f"Top 5 features:\n{importance.head().to_string()}")
    return importance


def load_model() -> xgb.XGBRegressor:
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def predict(features: pd.DataFrame) -> np.ndarray:
    """Run inference on new features."""
    model = load_model()
    return model.predict(features)


if __name__ == "__main__":
    import typer

    def main(data_path: str = "data/processed/features.parquet"):
        df = pd.read_parquet(data_path)
        target_col = "ed_admissions_next_24h"

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            raise SystemExit(1)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = train(X, y)
        importance = explain(model, X)
        print("\nTop feature importances:")
        print(importance.head(10).to_string(index=False))

    typer.run(main)
