import pandas as pd

# Read both datasets
df_toxic = pd.read_csv('dataset/islam.csv', sep=';')
df_positive = pd.read_csv('dataset/islam_positive.csv', sep=';')

# Combine the datasets
df_combined = pd.concat([df_toxic, df_positive], ignore_index=True)

# Shuffle the combined dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
df_combined.to_csv('dataset/islam_balanced.csv', sep=';', index=False)

# Print statistics
print("\n=== Dataset Statistics ===")
print(f"Total number of comments: {len(df_combined)}")
print("\nClass distribution:")
print(df_combined['label'].value_counts())
print("\nClass percentages:")
print((df_combined['label'].value_counts() / len(df_combined) * 100).round(2)) 