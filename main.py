from src.data_preprocessing import preprocess
from src.models import (
    train_random_forest,
    train_xgboost,
    train_lightgbm
)
from src.mlflow_logging import (
    log_model_with_mlflow,
    setup_mlflow_experiment
)
from pathlib import Path
import logging
from colorama import Fore, Style
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )


def main():
    setup_logging()
    logging.info("🚀 Starting Antibiotic Prediction Experiment...")

    # MLflow setup
    experiment_id = setup_mlflow_experiment("Antibiotic_Prediction_Exp")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "src/data" / "complete_microbiology_cultures_data.csv"
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Preprocessing
    transformer, label_encoder, X_train, X_test, y_train, y_test = preprocess(df, output_dir)

    # === Model 1: Random Forest ===
    rf_model = train_random_forest(X_train, y_train)
    log_model_with_mlflow(
        model=rf_model,
        col_transf=transformer,
        X_test=X_test,
        y_test=y_test,
        model_name="RandomForestClassifier",
        exp_id=experiment_id,
        output_dir=output_dir
    )

    # === Model 2: XGBoost ===
    xgb_model = train_xgboost(X_train, y_train)
    log_model_with_mlflow(
        model=xgb_model,
        col_transf=transformer,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoostClassifier",
        exp_id=experiment_id,
        output_dir=output_dir
    )

    # === Model 3: LightGBM ===
    lgbm_model = train_lightgbm(X_train, y_train)
    log_model_with_mlflow(
        model=lgbm_model,
        col_transf=transformer,
        X_test=X_test,
        y_test=y_test,
        model_name="LightGBMClassifier",
        exp_id=experiment_id,
        output_dir=output_dir
    )

    logging.info("✅ All models trained and logged successfully!")


if __name__ == "__main__":
    main()
