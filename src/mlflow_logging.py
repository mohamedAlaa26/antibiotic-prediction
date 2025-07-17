import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
import joblib
from pathlib import Path
from .evaluation import eval_metrics, plot_confusion_matrix
import logging
import pandas as pd
from mlflow.models.signature import infer_signature


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://localhost:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def log_model_with_mlflow(model, col_transf, X_test, y_test, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")

        mlflow.set_tag("model", model_name)

        pred = model.predict(X_test)
        accuracy, f1 = eval_metrics(y_test, pred)

        # Log model performance metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "score": model.score(X_test, y_test),
            "accuracy": accuracy,
            "f1_macro": f1
        })

        # Plot and log confusion matrix
        plot_confusion_matrix(y_test, pred, output_dir, labels=model.classes_)
        cm_path = output_dir / "confusion_matrix.png"
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Log input dataset
        pd_dataset = mlflow.data.from_pandas(X_test, name="Testing Dataset")
        mlflow.log_input(pd_dataset, context="Testing")

        # Save and log the model
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Save and log transformer and label encoder
        transformer_path = output_dir / "transformer.pkl"
        label_encoder_path = output_dir / "label_encoder.pkl"

        if transformer_path.exists():
            mlflow.log_artifact(str(transformer_path), artifact_path="transformer")

        if label_encoder_path.exists():
            mlflow.log_artifact(str(label_encoder_path), artifact_path="label_encoder")

        # Log the MLflow model itself (with signature)
        signature = infer_signature(X_test, pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=X_test.iloc[[0]],
        )
