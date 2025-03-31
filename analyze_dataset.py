import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
print("Loading dataset...")
df = pd.read_excel('Dataset2_Needs.xlsx')

print("\n=== Basic Dataset Information ===")
print(f"Dataset Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0] if any(missing > 0) else "No missing values found")

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== First few rows ===")
print(df.head())

# If there are binary columns, show their value counts
binary_cols = df.select_dtypes(include=['bool', 'int64']).columns
if len(binary_cols) > 0:
    print("\n=== Binary Column Distributions ===")
    for col in binary_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3) * 100, "% of data")

# Save correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print("\n=== Correlation Matrix ===")
print("Saved as 'correlation_matrix.png'") 