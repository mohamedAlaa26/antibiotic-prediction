from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging

def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=6,
        max_depth=10,
        max_leaf_nodes=50,
        criterion='gini'
    )
    model = rf.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    logging.info("Training XGBoost model...")
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=len(set(y_train)),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=6
    )
    model = xgb.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    logging.info("Training LightGBM model...")
    lgbm = LGBMClassifier(
        objective='multiclass',
        num_class=len(set(y_train)),
        learning_rate=0.1,
        n_estimators=100,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=6
    )
    model = lgbm.fit(X_train, y_train)
    return model
