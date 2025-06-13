import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import pandas as pd
import numpy as np
from pathlib import Path

# Chemins
EXAMPLES_PATH = 'data/exemples.txt'
BINARY_MODEL_PATH = 'models/model_binaire'
CLASS_MODEL_PATH = 'models/model_multiclasses'
LABEL_MAP_PATH = f'{CLASS_MODEL_PATH}/label_mapping.csv'
OUTPUT_CSV = 'data/resultats.csv'

# Charger le mapping label -> type
label_map_df = pd.read_csv(LABEL_MAP_PATH)
label2type = dict(zip(label_map_df['label'], label_map_df['type']))

def predict_binary(texts, model, tokenizer, device):
    model.eval()
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        results.append((pred, float(probs[1]), float(probs[0])))
    return results

def predict_class(texts, model, tokenizer, device):
    model.eval()
    preds, confs = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
        preds.append(pred)
        confs.append(conf)
    return preds, confs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Charger modèles/tokenizers
    binary_tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    binary_model = CamembertForSequenceClassification.from_pretrained(BINARY_MODEL_PATH).to(device)
    class_tokenizer = CamembertTokenizer.from_pretrained(CLASS_MODEL_PATH)
    class_model = CamembertForSequenceClassification.from_pretrained(CLASS_MODEL_PATH).to(device)

    # Charger les exemples
    with open(EXAMPLES_PATH, encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Prédiction binaire
    binary_results = predict_binary(texts, binary_model, binary_tokenizer, device)
    # Pour les harcèlements, prédire le type
    texts_harc = [t for t, (pred, _, _) in zip(texts, binary_results) if pred == 1]
    if texts_harc:
        class_preds, class_confs = predict_class(texts_harc, class_model, class_tokenizer, device)
    else:
        class_preds, class_confs = [], []

    # Construction du DataFrame résultat
    rows = []
    idx_harc = 0
    for text, (pred_bin, conf_bin, conf_non) in zip(texts, binary_results):
        if pred_bin == 1:
            pred_type = label2type[class_preds[idx_harc]]
            conf_type = class_confs[idx_harc]
            idx_harc += 1
        else:
            pred_type = ''
            conf_type = ''
        rows.append({
            'texte': text,
            'is_harassment': pred_bin,
            'confiance_binaire': round(conf_bin, 2),
            'prediction_type': pred_type,
            'confiance_type': round(conf_type, 2) if conf_type != '' else ''
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Résultats enregistrés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 