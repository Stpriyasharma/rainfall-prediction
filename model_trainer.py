"""
Module for feature selection, model training, and evaluation.

This file contains functions to select important features and to train
and evaluate various machine learning models.
"""

import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
# import catboost as cb

# Import helper functions from the utils module
from utils import run_model

def select_features(data):
    """
    Selects important features using the Random Forest wrapper method.
    """
    print("\n--- Step 5: Feature Selection ---")
    X = data.drop('RainTomorrow', axis=1)
    y = data['RainTomorrow']

    print("--- Using Wrapper Method (Random Forest) ---")
    selector_rf = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0))
    selector_rf.fit(X, y)
    rf_features = X.columns[selector_rf.get_support()].tolist()
    print(f"Features selected by Random Forest: {rf_features}")

    # Use features selected by Random Forest for consistency with the article's approach
    selected_features = rf_features
    selected_features.append('RainTomorrow')
    selected_features = [col for col in selected_features if col in data.columns]
    
    X_final = data[selected_features].drop('RainTomorrow', axis=1)
    y_final = data[selected_features]['RainTomorrow']

    return X_final, y_final

def train_and_evaluate_models(X, y):
    """
    Trains and evaluates multiple machine learning models.
    """
    print("\n--- Step 6: Model Training and Evaluation ---")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12345
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Train and Evaluate Models ---

    # Logistic Regression
    print("\n--- Training Logistic Regression Model ---")
    model_lr = LogisticRegression()
    run_model(model_lr, X_train, y_train, X_test, y_test)

    # Decision Tree Classifier
    print("\n--- Training Decision Tree Classifier Model ---")
    params_dt = {'max_depth': 16, 'max_features': "sqrt"}
    model_dt = DecisionTreeClassifier(**params_dt)
    run_model(model_dt, X_train, y_train, X_test, y_test)

    # Neural Network (MLPClassifier)
    print("\n--- Training Neural Network (MLPClassifier) Model ---")
    params_nn = {'hidden_layer_sizes': (30, 30, 30), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 500}
    model_nn = MLPClassifier(**params_nn)
    run_model(model_nn, X_train, y_train, X_test, y_test)

    # Random Forest Classifier
    print("\n--- Training Random Forest Classifier Model ---")
    params_rf = {'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 12345}
    model_rf = RandomForestClassifier(**params_rf)
    run_model(model_rf, X_train, y_train, X_test, y_test)

    # LightGBM Classifier
    print("\n--- Training LightGBM Classifier Model ---")
    params_lgb = {
        'colsample_bytree': 0.95, 'max_depth': 16, 'min_split_gain': 0.1,
        'n_estimators': 200, 'num_leaves': 50, 'reg_alpha': 1.2,
        'reg_lambda': 1.2, 'subsample': 0.95, 'subsample_freq': 20
    }
    model_lgb = lgb.LGBMClassifier(**params_lgb)
    run_model(model_lgb, X_train, y_train, X_test, y_test)

    # XGBoost Classifier
    print("\n--- Training XGBoost Classifier Model ---")
    params_xgb = {'n_estimators': 500, 'max_depth': 16}
    model_xgb = xgb.XGBClassifier(**params_xgb)
    run_model(model_xgb, X_train, y_train, X_test, y_test)
