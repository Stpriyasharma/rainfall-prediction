"""
Main script to run the rainfall prediction workflow.

This file orchestrates the entire process by importing functions
from other modules and executing them in the correct order.
"""

import pandas as pd
import warnings

# Import functions from custom modules
from data_handler import (
    load_data,
    preprocess_data,
    handle_class_imbalance,
    impute_and_transform,
    remove_outliers
)
from model_trainer import (
    select_features,
    train_and_evaluate_models
)

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    """
    Main function to execute the rainfall prediction pipeline.
    """
    print("--- Starting Rainfall Prediction Pipeline ---")
    
    # Step 1: Data Loading and Exploration
    try:
        full_data = load_data('weatherAUS.csv')
    except FileNotFoundError:
        print("Error: 'weatherAUS.csv' not found. Please ensure the file is in the same directory.")
        return

    # Step 2: Preprocess and Handle Imbalance
    processed_data = preprocess_data(full_data)
    oversampled_data = handle_class_imbalance(processed_data)
    
    # Step 3: Imputation and Transformation
    imputed_data = impute_and_transform(oversampled_data)
    
    # Step 4: Outlier Removal
    cleaned_data = remove_outliers(imputed_data)
    
    # Step 5: Feature Selection
    X, y = select_features(cleaned_data)
    
    # Step 6: Model Training and Evaluation
    print("\n--- Training and Evaluating Models ---")
    train_and_evaluate_models(X, y)
    
    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()