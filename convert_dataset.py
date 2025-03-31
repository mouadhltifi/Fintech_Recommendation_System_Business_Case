import pandas as pd

print("Reading original dataset...")
df = pd.read_excel('Dataset2_Needs.xls')

print(f"Converting dataset with shape: {df.shape}")
print("Saving to xlsx format...")
df.to_excel('Dataset2_Needs.xlsx', index=False)

print("Conversion completed successfully!")
print("New file created: Dataset2_Needs.xlsx") 