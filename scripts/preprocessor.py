"""
Data Preprocessing on data/dataset_original.csv

We following Experiment 1 detailed in the original paper to preprocess our data:
1.  Loading the raw data.
2.  Performing a detailed Exploratory Data Analysis (EDA) with visualizations.
3.  Checking for data quality issues like missing values.
4.  Creating a balanced version of the dataset using random undersampling.
5.  Saving the processed (imbalanced and balanced) datasets as CSV files.

"""

#==============================================================================
# Part 1: Setup and Environment Configuration
#==============================================================================

print("STEP 1: Setting up the environment...")

# Standard library imports
import os
import logging

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn and imbalanced-learn imports
from imblearn.under_sampling import RandomUnderSampler

# --- Global Configurations ---

# Input file path
RAW_DATASET_PATH = os.path.join('data', 'dataset_original.csv')

# Output directory for processed data and plots
OUTPUT_DIR = 'processed_data'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set up logging for clear, informative outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure plot aesthetics for professional-looking reports
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Environment setup complete.\n")

#==============================================================================
# Part 2: Data Loading and Exploratory Data Analysis (EDA)
#==============================================================================

print("STEP 2: Loading and Exploring the Dataset...")

def load_data(filepath):
    """
    Loads the dataset from a specified CSV file with robust error handling.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Dataset successfully loaded from '{filepath}'.")
        return df
    except FileNotFoundError:
        logging.error(f"FATAL: Dataset file not found at '{filepath}'. Please check the file path.")
        exit()

def detailed_eda(df, plots_dir):
    """
    Performs a comprehensive Exploratory Data Analysis on the dataset.
    """
    logging.info("Starting comprehensive Exploratory Data Analysis (EDA)...")
    
    # --- 1. Basic Information ---
    print("\n--- Dataset Head ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    df.info()
    print("\n--- Statistical Summary ---")
    print(df.describe())
    
    # --- 2. Data Quality Check ---
    missing_values = df.isnull().sum().sum()
    logging.info(f"Total number of missing values in the dataset: {missing_values}")
    if missing_values > 0:
        logging.warning("Missing values detected. Further investigation or imputation may be required.")
    else:
        logging.info("Data quality check passed: No missing values found.")

    # --- 3. Class Distribution Analysis ---
    logging.info("Analyzing class distribution (Goodware vs. Malware)...")
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(data=df, x='malware', hue='malware', palette=['#4CAF50', '#F44336'], legend=False)
    
    plt.title('Imbalanced Class Distribution', fontsize=16, weight='bold')
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['0: Goodware', '1: Malware'])
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    plot_path = os.path.join(plots_dir, 'imbalanced_class_distribution.png')
    plt.savefig(plot_path)
    plt.show()
    logging.info(f"Class distribution plot saved to '{plot_path}'")

    if 'malware' in df.columns and 1 in df['malware'].value_counts():
        malware_count = df['malware'].value_counts()[1]
        goodware_count = df['malware'].value_counts().get(0, 0)
        imbalance_ratio = malware_count / goodware_count if goodware_count > 0 else float('inf')
        logging.info(f"Imbalance Ratio: {imbalance_ratio:.2f} malware samples for every 1 goodware sample.")

    # --- 4. API Call Frequency Analysis ---
    logging.info("Analyzing frequency of the Top 30 most common API calls...")
    api_call_columns = [f't_{i}' for i in range(100)]
    api_calls_flat = df[api_call_columns].values.flatten()
    
    plt.figure(figsize=(15, 8))
    pd.Series(api_calls_flat).value_counts().nlargest(30).plot(kind='bar', color='skyblue')
    plt.title('Top 30 Most Frequent API Calls Across All Samples', fontsize=16, weight='bold')
    plt.xlabel('API Call ID Number', fontsize=12)
    plt.ylabel('Total Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'top_30_api_calls_frequency.png')
    plt.savefig(plot_path)
    plt.show()
    logging.info(f"API call frequency plot saved to '{plot_path}'")
    
    logging.info("EDA has been successfully completed.")


#==============================================================================
# Part 3: Data Balancing and Final Export
#==============================================================================

print("\nSTEP 3: Balancing the dataset and preparing for export...")

def balance_and_export_data(df, output_dir, plots_dir):
    """
    Balances the dataset using random undersampling and saves both the original
    and balanced datasets as CSV files.
    """
    logging.info("Separating features (X) and labels (y)...")
    X_imbalanced = df.drop(['hash', 'malware'], axis=1).values
    y_imbalanced = df['malware'].values
    
    logging.info(f"Shape of imbalanced features: {X_imbalanced.shape}")
    logging.info(f"Shape of imbalanced labels: {y_imbalanced.shape}")

    logging.info("Applying Random Undersampling to create a balanced dataset...")
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_imbalanced, y_imbalanced)
    
    logging.info(f"Shape of balanced features after undersampling: {X_balanced.shape}")
    logging.info(f"Shape of balanced labels after undersampling: {y_balanced.shape}")
    
    # --- Visualize the balanced distribution ---
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(x=y_balanced, hue=y_balanced, palette=['#4CAF50', '#F44336'], legend=False)
    
    plt.title('Balanced Class Distribution', fontsize=16, weight='bold')
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['0: Goodware', '1: Malware'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    plot_path = os.path.join(plots_dir, 'balanced_class_distribution.png')
    plt.savefig(plot_path)
    plt.show()
    logging.info(f"Balanced class distribution plot saved to '{plot_path}'")

    # --- Save the processed data to CSV ---
    logging.info("Saving processed datasets to .csv files...")

    # Define column names
    feature_columns = [f't_{i}' for i in range(X_imbalanced.shape[1])]
    all_columns = feature_columns + ['malware']

    # --- Process and Save Imbalanced Dataset ---
    # Combine features and labels into one DataFrame
    imbalanced_df = pd.DataFrame(np.c_[X_imbalanced, y_imbalanced], columns=all_columns)
    # Ensure integer types are preserved for consistency
    imbalanced_df = imbalanced_df.astype(int)
    # Define file path
    imbalanced_csv_path = os.path.join(output_dir, 'imbalanced_dataset.csv')
    # Save to CSV, excluding the default pandas index
    imbalanced_df.to_csv(imbalanced_csv_path, index=False)
    logging.info(f"Saved imbalanced dataset to '{imbalanced_csv_path}'")

    # --- Process and Save Balanced Dataset ---
    # Combine features and labels into one DataFrame
    balanced_df = pd.DataFrame(np.c_[X_balanced, y_balanced], columns=all_columns)
    # Ensure integer types are preserved
    balanced_df = balanced_df.astype(int)
    # Define file path
    balanced_csv_path = os.path.join(output_dir, 'balanced_dataset.csv')
    # Save to CSV, excluding the default pandas index
    balanced_df.to_csv(balanced_csv_path, index=False)
    logging.info(f"Saved balanced dataset to '{balanced_csv_path}'")

    logging.info(f"All processed CSV files have been saved in the '{output_dir}' directory.")


#==============================================================================
# Main Driver
#==============================================================================

if __name__ == "__main__":
    logging.info("--- Starting Data Preprocessing Pipeline ---")
    
    # 1. Load the data
    raw_df = load_data(RAW_DATASET_PATH)
    
    # 2. Perform detailed EDA
    detailed_eda(raw_df, PLOTS_DIR)
    
    # 3. Balance the data and export for model training
    balance_and_export_data(raw_df, OUTPUT_DIR, PLOTS_DIR)
    
    logging.info("--- Data Preprocessing Pipeline Finished Successfully ---")