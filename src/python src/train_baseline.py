import pandas as pd
import numpy as np
# Path 
FILE_PATH = "/Users/jonaslorler/master-thesis-uq-churn/data/raw/kaggle_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Load dataset
df = pd.read_csv(FILE_PATH)
print("Dataset Shape:", df.shape)
print("\n--- Data Types ---")
df.info()

print("\n--- First 5 Rows ---")
display(df.head())
