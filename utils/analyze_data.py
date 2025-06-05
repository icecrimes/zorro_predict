import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('upda_dataset.csv', sep=';')

# Basic statistics
print("\n=== Basic Statistics ===")
print(f"Total number of examples: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nColumns:", df.columns.tolist())

# Class distribution
print("\n=== Class Distribution ===")
class_counts = df['labels'].value_counts()
print("\nClass distribution:")
print(class_counts)
print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

# Text length statistics
print("\n=== Text Length Statistics ===")
df['text_length'] = df['text'].str.len()
print("\nText length statistics (in characters):")
print(df['text_length'].describe())

# Sample examples
print("\n=== Sample Examples ===")
print("\nSample of harassment examples (label=1):")
print(df[df['labels'] == 1]['text'].head(3).to_string())
print("\nSample of non-harassment examples (label=0):")
print(df[df['labels'] == 0]['text'].head(3).to_string())

# Word frequency analysis
print("\n=== Word Frequency Analysis ===")
# Combine all texts and split into words
all_words = ' '.join(df['text'].astype(str)).lower().split()
word_freq = Counter(all_words)
print("\nMost common words:")
print(word_freq.most_common(10))

# Save the analysis results
with open('data_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("=== Dataset Analysis Report ===\n\n")
    f.write(f"Total examples: {len(df)}\n")
    f.write(f"Class distribution:\n{class_counts}\n\n")
    f.write(f"Text length statistics:\n{df['text_length'].describe()}\n\n")
    f.write("Sample harassment examples:\n")
    f.write(df[df['labels'] == 1]['text'].head(3).to_string() + "\n\n")
    f.write("Sample non-harassment examples:\n")
    f.write(df[df['labels'] == 0]['text'].head(3).to_string() + "\n\n")
    f.write("Most common words:\n")
    f.write(str(word_freq.most_common(10)))

print("\nAnalysis complete! Results have been saved to 'data_analysis.txt'") 