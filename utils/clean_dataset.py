import pandas as pd

# Read the dataset
print("Reading dataset...")
df = pd.read_csv('dataset/dataset_v2.csv', sep=';')

# Remove rows where text is empty or contains only whitespace
print("Cleaning dataset...")
df = df[df['text'].str.strip().str.len() > 0]

# Save the cleaned dataset
print("Saving cleaned dataset...")
df.to_csv('dataset/dataset_v2_cleaned.csv', sep=';', index=False)

# Print statistics
print("\n=== Dataset Statistics ===")
print(f"Original number of rows: {len(df)}")
print(f"Number of rows after cleaning: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nClass percentages:")
print((df['label'].value_counts() / len(df) * 100).round(2)) 