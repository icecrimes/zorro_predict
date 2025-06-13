import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import pandas as pd
import numpy as np
import os
import sys

EXAMPLES_PATH = 'data/exemples.txt'
MODEL_PATH = f'models/model_multiclasses'
LABEL_MAP_PATH = f'models/model_multiclasses/label_mapping.csv'


# Charger le mapping label -> type
label_map_df = pd.read_csv(LABEL_MAP_PATH)
label2type = dict(zip(label_map_df['label'], label_map_df['type']))

def predict(texts, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    # Déplacer les inputs sur le même device que le modèle
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

if __name__ == "__main__":
    # Déterminer le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Charger le modèle et le tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
    model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)


    # Charger les exemples
    with open(EXAMPLES_PATH, encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    preds, probs = predict(texts, model, tokenizer, device)

    for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
        type_name = label2type[pred]
        print(f"\n{'='*50}")
        print(f"Exemple {i + 1}: {text}")
        print(f"{'='*50}")
        print(f"  → Prédiction: {type_name}")
        print(f"  Scores: { {label2type[j]: float(f'{p:.3f}') for j, p in enumerate(prob)} }\n") 