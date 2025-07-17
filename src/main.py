from data_preprocessing import preprocess
from pathlib import Path
import logging
from colorama import Fore, Style
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )



def main():
    setup_logging()
    logging.info("Starting data preparation...")

    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "data/complete_microbiology_cultures_data.csv"
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    transf, le, X_train, X_test, y_train, y_test = preprocess(df)

    logging.info("Data preparation completed successfully!")

    # Save transformer
    joblib.dump(transf, "models/transformer.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    # Save sample of X_train
    pd.DataFrame(X_train).head(10).to_csv(output_dir / "X_train_sample.csv", index=False)

    # Save label distribution plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x=y_train)
    plt.title("Class Distribution in y_train")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()



    logging.info("Preprocessed data and plots saved to output folder.")



if __name__ == "__main__":
    main()