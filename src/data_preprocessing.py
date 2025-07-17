from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
import pandas as pd

def preprocess(df):
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Column transformer for features
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        remainder="passthrough"
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    return col_transf, le, X_train, X_test, y_train, y_test
