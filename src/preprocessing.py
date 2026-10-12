"""
Preprocessing functions for the Mental Health in Tech Survey dataset.
All functions are designed to be reusable and modular.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns_to_drop : list
        List of column names to drop
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with specified columns dropped
    """
    df = df.copy()
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    return df


def clean_age(df, min_age=18, max_age=75):
    """
    Remove rows where Age is outside the valid range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    min_age : int
        Minimum valid age (default: 18)
    max_age : int
        Maximum valid age (default: 75)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with invalid ages removed
    """
    df = df.copy()
    df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
    return df


def clean_gender(df):
    """
    Clean and standardize Gender column to 2 categories: Male, Female.
    Drops rows with non-binary, queer, enby, and other non-standard gender values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Gender' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned Gender column (only Male/Female rows)
    """
    df = df.copy()
    
    # Male keywords (including common misspellings)
    male_keywords = ['male', 'm', 'man', 'cis male', 'maile', 'mal', 'male (cis)', 
                     'make', 'male ', 'msle', 'mail', 'malr', 'cis man', 'guy']
    
    # Female keywords (including common misspellings)
    female_keywords = ['female', 'f', 'woman', 'cis female', 'femake', 'femail', 
                       'cis-female/femme', 'female (cis)', 'femmale', 'female ']
    
    def standardize_gender(gender):
        if pd.isna(gender):
            return None  # Will be dropped
        gender_str = str(gender).strip().lower()
        if gender_str in male_keywords:
            return 'Male'
        elif gender_str in female_keywords:
            return 'Female'
        else:
            return None  # Drop non-standard values (enby, queer, non-binary, etc.)
    
    df['Gender'] = df['Gender'].apply(standardize_gender)
    
    # Drop rows where Gender is None (non-standard values)
    initial_rows = len(df)
    df = df.dropna(subset=['Gender'])
    print(f"  Dropped {initial_rows - len(df)} rows with non-standard gender values")
    
    return df


def impute_missing_values(df):
    """
    Impute missing values with domain-specific defaults.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    df = df.copy()
    
    # self_employed: fill NaN with 'No' (safe assumption)
    if 'self_employed' in df.columns:
        df['self_employed'] = df['self_employed'].fillna('No')
    
    # work_interfere: fill NaN with "Don't know"
    if 'work_interfere' in df.columns:
        df['work_interfere'] = df['work_interfere'].fillna("Don't know")
    
    # Any remaining NaNs: fill with mode for each column
    for col in df.columns:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    return df


def encode_ordinal_columns(df):
    """
    Encode ordinal columns with manual mappings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with ordinal columns encoded
    """
    df = df.copy()
    
    # work_interfere mapping
    work_interfere_map = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Often': 3,
        "Don't know": -1
    }
    if 'work_interfere' in df.columns:
        df['work_interfere'] = df['work_interfere'].map(work_interfere_map)
    
    # no_employees mapping (use midpoints)
    no_employees_map = {
        '1-5': 3,
        '6-25': 15,
        '26-100': 60,
        '100-500': 250,
        '500-1000': 750,
        'More than 1000': 1500
    }
    if 'no_employees' in df.columns:
        df['no_employees'] = df['no_employees'].map(no_employees_map)
    
    # leave mapping
    leave_map = {
        'Very easy': 4,
        'Somewhat easy': 3,
        "Don't know": 2,
        'Somewhat difficult': 1,
        'Very difficult': 0
    }
    if 'leave' in df.columns:
        df['leave'] = df['leave'].map(leave_map)
    
    return df


def encode_binary_columns(df):
    """
    Encode binary columns (Yes/No) using LabelEncoder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with binary columns encoded
    """
    df = df.copy()
    
    # Common binary columns in the dataset
    binary_cols = [
        'self_employed', 'family_history', 'treatment', 'remote_work', 
        'tech_company', 'obs_consequence', 'mental_vs_physical', 
        'mental_health_interview', 'phys_health_interview'
    ]
    
    le = LabelEncoder()
    
    for col in binary_cols:
        if col in df.columns:
            # Check if column is binary (Yes/No)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 3:  # Yes, No, and maybe NaN
                df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def encode_categorical_columns(df, drop_first=True):
    """
    Encode remaining categorical columns using one-hot encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    drop_first : bool
        Whether to drop first column to avoid multicollinearity
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with categorical columns one-hot encoded
    """
    df = df.copy()
    
    # Get categorical columns (object type)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    
    return df


def full_preprocessing_pipeline(df):
    """
    Apply the complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw dataframe
        
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe
    """
    print("Starting preprocessing pipeline...")
    
    # Step 1: Drop columns
    print("Step 1: Dropping columns (Timestamp, comments, state)...")
    df = drop_columns(df, ['Timestamp', 'comments', 'state'])
    
    # Step 2: Clean Age
    print("Step 2: Cleaning Age (removing values < 18 or > 75)...")
    initial_rows = len(df)
    df = clean_age(df)
    print(f"  Removed {initial_rows - len(df)} rows with invalid ages")
    
    # Step 3: Clean Gender
    print("Step 3: Cleaning Gender (keeping only Male/Female, dropping others)...")
    df = clean_gender(df)
    print(f"  Gender distribution: {df['Gender'].value_counts().to_dict()}")
    
    # Step 4: Impute missing values
    print("Step 4: Imputing missing values...")
    df = impute_missing_values(df)
    print(f"  Remaining missing values: {df.isna().sum().sum()}")
    
    # Step 5: Encode ordinal columns
    print("Step 5: Encoding ordinal columns...")
    df = encode_ordinal_columns(df)
    
    # Step 6: Encode binary columns
    print("Step 6: Encoding binary columns...")
    df = encode_binary_columns(df)
    
    # Step 7: Encode categorical columns
    print("Step 7: One-hot encoding remaining categorical columns...")
    df = encode_categorical_columns(df)
    
    print(f"Preprocessing complete! Final shape: {df.shape}")
    
    return df
