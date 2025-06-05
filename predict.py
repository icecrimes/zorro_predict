from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import json

model_path = 'results/run_20250605_124448/final_model'

# Print model info
with open(f'{model_path}/config.json') as f:
    config = json.load(f)
print("Model loaded from:", model_path)
print("Original model:", config.get("name_or_path", "unknown"))




def predict_text(text, model, tokenizer):
    # Préparation du texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Récupération des probabilités
    prob_harcelement = predictions[0][1].item()
    prob_non_harcelement = predictions[0][0].item()
    
    # Détermination de la classe
    prediction = 1 if prob_harcelement > prob_non_harcelement else 0
    
    return {
        'prediction': prediction,
        'prob_harcelement': prob_harcelement,
        'prob_non_harcelement': prob_non_harcelement
    }


tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForSequenceClassification.from_pretrained(model_path)


def main():

    text = "jules toi aussi tu pue la merde"

    result = predict_text(text, model, tokenizer)
    
    print(model_path)
    print("\nTexte analysé:", text)
    print("Prédiction:", "Harcèlement" if result['prediction'] == 1 else "Non-harcèlement")
    print(f"Probabilité de harcèlement: {result['prob_harcelement']:.2%}")
    print(f"Probabilité de non-harcèlement: {result['prob_non_harcelement']:.2%}")

if __name__ == "__main__":
    main()
 