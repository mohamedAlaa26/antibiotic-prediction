from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
import joblib
from pathlib import Path


def preprocess(df: pd.DataFrame, output_dir: Path):
    df = df.dropna()

    if "order_proc_id_coded" in df.columns:
        df = df.drop(columns=["order_proc_id_coded"])

    cat_cols = ["culture_description", "age"]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X = df.drop("antibiotic", axis=1)
    y = df["antibiotic"]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Save label encoder
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, output_dir / "label_encoder.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Column transformer
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        remainder="passthrough"
    )

    # Fit and transform
    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    # Save transformer
    joblib.dump(col_transf, output_dir / "transformer.pkl")

    # Wrap into DataFrames (preserve feature names)
    feature_names = col_transf.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    return col_transf, le, X_train_df, X_test_df, y_train, y_test
