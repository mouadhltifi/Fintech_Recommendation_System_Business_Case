import pandas as pd

# Read the original dataset
df = pd.read_excel('Dataset2_Needs.xls')

# Take 20 random rows from the dataset
reduced_df = df.sample(n=20, random_state=42)

# Save the reduced dataset
reduced_df.to_excel('reduced_dataset.xlsx', index=False)

print("Reduced dataset created successfully with 20 rows")
print(f"Original dataset shape: {df.shape}")
print(f"Reduced dataset shape: {reduced_df.shape}") 