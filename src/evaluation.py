from sklearn import metrics
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np

def eval_metrics(actual, pred):
    logging.info("Calculating evaluation metrics...")

    accuracy = metrics.accuracy_score(actual, pred)
    f1_macro = metrics.f1_score(actual, pred, average='macro')

    logging.info(f"Accuracy: {accuracy:.4f}, F1 Score (macro): {f1_macro:.4f}")
    return accuracy, f1_macro

def plot_confusion_matrix(actual, pred, output_dir: Path, labels=None):
    logging.info("Plotting confusion matrix...")

    cm = metrics.confusion_matrix(actual, pred)
    plt.figure(figsize=(10, 8))
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
