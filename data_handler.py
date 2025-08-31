"""
Module for data handling, preprocessing, and cleaning.

This file contains functions for loading the dataset, preprocessing it,
handling class imbalance, and dealing with missing values and outliers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    print("--- Step 1: Data Loading and Exploration ---")
    full_data = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    print("Initial data shape:", full_data.shape)
    print("\nFirst 5 rows of the data:")
    print(full_data.head())
    print("\nData information:")
    print(full_data.info())
    return full_data

def preprocess_data(data):
    """
    Performs initial data preprocessing, including binary conversion.
    """
    data_copy = data.copy()
    data_copy['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    data_copy['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    print("\n'RainToday' and 'RainTomorrow' columns converted to binary.")
    return data_copy

def handle_class_imbalance(data):
    """
    Handles class imbalance using oversampling of the minority class.
    """
    print("\n--- Step 2: Handling Class Imbalance ---")
    print("Class distribution before oversampling:")
    print(data.RainTomorrow.value_counts(normalize=True))

    fig = plt.figure(figsize=(8, 5))
    data.RainTomorrow.value_counts(normalize=True).plot(
        kind='bar', color=['skyblue', 'navy'], alpha=0.9, rot=0
    )
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
    plt.show()

    no = data[data.RainTomorrow == 0]
    yes = data[data.RainTomorrow == 1]
    yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
    oversampled = pd.concat([no, yes_oversampled])

    print("\nClass distribution after oversampling:")
    print(oversampled.RainTomorrow.value_counts(normalize=True))

    fig = plt.figure(figsize=(8, 5))
    oversampled.RainTomorrow.value_counts(normalize=True).plot(
        kind='bar', color=['skyblue', 'navy'], alpha=0.9, rot=0
    )
    plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')
    plt.show()

    return oversampled

def impute_and_transform(data):
    """
    Imputes missing values and transforms categorical features.
    """
    print("\n--- Step 3: Imputation and Transformation ---")
    print("Missing values heatmap before imputation:")
    sns.heatmap(data.isnull(), cbar=False, cmap='PuBu')
    plt.title('Missing Values Heatmap')
    plt.show()

    # Impute categorical variables with mode
    data_copy = data.copy()
    categorical_cols = data_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data_copy[col].fillna(data_copy[col].mode()[0], inplace=True)

    # Convert categorical features to continuous features with Label Encoding
    lencoders = {}
    for col in data_copy.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        data_copy[col] = lencoders[col].fit_transform(data_copy[col])

    # Impute numeric features using IterativeImputer (MICE)
    mice_imputer = IterativeImputer()
    imputed_data = data_copy.copy(deep=True)
    imputed_data.iloc[:, :] = mice_imputer.fit_transform(data_copy)

    print("\nMissing values heatmap after imputation:")
    sns.heatmap(imputed_data.isnull(), cbar=False, cmap='PuBu')
    plt.title('Missing Values Heatmap after Imputation')
    plt.show()

    return imputed_data

def remove_outliers(data):
    """
    Detects and removes outliers from the dataset using the IQR method.
    """
    print("\n--- Step 4: Outlier Detection and Removal ---")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print("Interquartile Range (IQR) for each feature:")
    print(IQR)

    # Removing outliers
    cleaned_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"\nShape of the dataset after removing outliers: {cleaned_data.shape}")
    return cleaned_data