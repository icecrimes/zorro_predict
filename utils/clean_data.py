import pandas as pd
import re
from transformers import CamembertTokenizer, CamembertForSequenceClassification

# Liste des datasets valides
valid_datasets = ['CONAN', 'cyberado', 'MLMA', 'sexism', 'FTR']

# Lire le fichier CSV
df = pd.read_csv('dataset_harcelement_clean.csv')

# Remplacer les retours à la ligne dans la colonne text par des espaces
df['text'] = df['text'].str.replace('\n', ' ').str.replace('\r', ' ')

# Sauvegarder le fichier nettoyé
df.to_csv('dataset_harcelement_clean.csv', index=False)

print("Retours à la ligne supprimés du texte avec succès!")

# Fonction pour vérifier si une ligne se termine par un dataset valide
def is_valid_end(text):
    return any(text.strip().endswith(dataset) for dataset in valid_datasets)

# Créer une nouvelle liste pour stocker les lignes corrigées
corrected_lines = []
current_line = ''

# Parcourir chaque ligne
for index, row in df.iterrows():
    line = ','.join(str(x) for x in row)
    
    if not current_line:
        current_line = line
    else:
        current_line += line
    
    if is_valid_end(current_line):
        corrected_lines.append(current_line)
        current_line = ''

# Si une ligne reste à la fin
if current_line:
    corrected_lines.append(current_line)

# Écrire les lignes corrigées dans un nouveau fichier
with open('dataset_harcelement_clean.csv', 'w', encoding='utf-8') as f:
    # Écrire l'en-tête
    f.write(','.join(df.columns) + '\n')
    # Écrire les lignes corrigées
    for line in corrected_lines:
        f.write(line + '\n')

print("Lignes recollées et fichier corrigé !")

model_name = 'camembert-base'
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2) 