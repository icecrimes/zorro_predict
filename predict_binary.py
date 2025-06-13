from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import json
import logging
from pathlib import Path

EXAMPLES_PATH = 'data/exemples.txt'
MODEL_PATH = f'models/model_multiclasses'
LABEL_MAP_PATH = f'models/model_multiclasses/label_mapping.csv'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HarassmentPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
            
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        self.model = CamembertForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()
        
        # Chargement et affichage des informations du modèle
        with open(self.model_path / 'config.json') as f:
            config = json.load(f)
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Original model: {config.get('name_or_path', 'unknown')}")

    def predict(self, text):

        # Prepare text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        prob_non_harassment = probabilities[0][0].item()
        prob_harassment = probabilities[0][1].item()
        
        return {
            'prediction': 1 if prob_harassment > prob_non_harassment else 0,
            'prob_harassment': prob_harassment,
            'prob_non_harassment': prob_non_harassment
        }

def main():
    try:
        # Initialize predictor
        model_path = 'models/model_binaire'
        predictor = HarassmentPredictor(model_path)
        
        # Lire les exemples depuis le fichier
        with open('data/exemples.txt', 'r', encoding='utf-8') as f:
            textes = [ligne.strip() for ligne in f if ligne.strip()]
        
        # Analyser chaque texte
        for i, text in enumerate(textes, 1):
            print(f"\n{'='*50}")
            print(f"Exemple {i + 1}: {text}")
            print(f"{'='*50}")
        
            # Get prediction
            result = predictor.predict(text)
            
            # Affichage des résultats
            print("\nTexte analysé:", text)
            print("Prédiction:", "Harcèlement" if result['prediction'] == 1 else "Non-harcèlement")
            print(f"Probabilité de harcèlement: {result['prob_harassment']:.2%}")
            print(f"Probabilité de non-harcèlement: {result['prob_non_harassment']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
 