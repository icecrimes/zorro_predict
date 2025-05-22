from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

def load_model(model_path='./model_harcelement'):
    """Charge le modèle et le tokenizer"""
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict_harassment(text, model, tokenizer):
    """Prédit si un texte est du harcèlement ou non"""
    # Préparation du texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return {
        'is_harassment': bool(predicted_class),
        'confidence': confidence
    }

def main():
    # Charger le modèle
    print("Chargement du modèle...")
    model, tokenizer = load_model()
    
    # Interface simple pour tester
    print("\nEntrez 'q' pour quitter")
    while True:
        text = input("\nEntrez un message à analyser : ")
        if text.lower() == 'q':
            break
            
        result = predict_harassment(text, model, tokenizer)
        print(f"\nRésultat : {'Harcèlement' if result['is_harassment'] else 'Non harcèlement'}")
        print(f"Confiance : {result['confidence']:.2%}")

if __name__ == "__main__":
    main() 