import pandas as pd
import random

# Templates for positive comments
positive_templates = [
    "Je respecte la pratique de l'islam en France.",
    "La diversité religieuse enrichit notre société.",
    "Les musulmans contribuent positivement à notre pays.",
    "Le ramadan est une période importante pour les musulmans.",
    "La liberté de culte est un droit fondamental.",
    "Les valeurs de l'islam peuvent être compatibles avec la République.",
    "Les musulmans font partie intégrante de notre société.",
    "Le voile est un choix personnel qui doit être respecté.",
    "La laïcité protège toutes les religions.",
    "Les musulmans ont le droit de pratiquer leur foi.",
    "La diversité culturelle est une richesse.",
    "Les musulmans participent activement à la vie sociale.",
    "Le respect mutuel est essentiel dans notre société.",
    "Les musulmans enrichissent notre culture.",
    "La tolérance religieuse est une valeur importante.",
    "Les musulmans contribuent à l'économie française.",
    "Le dialogue interreligieux est enrichissant.",
    "Les musulmans font partie de notre histoire commune.",
    "La pratique de l'islam est un droit constitutionnel.",
    "Les musulmans participent à la vie démocratique."
]

# Variations to make comments more natural
variations = [
    "Je pense que ",
    "À mon avis, ",
    "Selon moi, ",
    "Je crois que ",
    "Je constate que ",
    "Je remarque que ",
    "Je vois que ",
    "Je comprends que ",
    "Je sais que ",
    "Je trouve que "
]

def generate_positive_comments(num_comments):
    comments = []
    for _ in range(num_comments):
        template = random.choice(positive_templates)
        variation = random.choice(variations)
        comment = variation + template.lower()
        comments.append(comment)
    return comments

# Read the original file to get the number of comments
df_original = pd.read_csv('dataset/islam.csv', sep=';')
num_comments = len(df_original)

# Generate positive comments
positive_comments = generate_positive_comments(num_comments)

# Create DataFrame with positive comments
df_positive = pd.DataFrame({
    'text': positive_comments,
    'label': [0] * num_comments,  # 0 for non-toxic
    'target': ['islam'] * num_comments,
    'source': ['GENERATED'] * num_comments,
    'dataset': ['POSITIVE'] * num_comments
})

# Save to CSV
df_positive.to_csv('dataset/islam_positive.csv', sep=';', index=False)

print(f"Generated {num_comments} positive comments and saved to 'dataset/islam_positive.csv'") 