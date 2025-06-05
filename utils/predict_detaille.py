from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import json
import logging
from pathlib import Path
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_prediction(text, model_path):
    # Chargement du modèle et du tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Tokenization du texte
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    # Affichage des tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    logger.info(f"\nTokens du texte: {tokens}")
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        raw_logits = outputs.logits[0]
        
        # Affichage des logits bruts
        logger.info(f"\nLogits bruts: {raw_logits.tolist()}")
        
        # Calcul des probabilités avec softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Affichage détaillé du calcul
        logits_np = raw_logits.numpy()
        exp_logits = np.exp(logits_np)
        sum_exp = np.sum(exp_logits)
        
        logger.info("\nCalcul détaillé:")
        logger.info(f"1. Logits: {logits_np}")
        logger.info(f"2. exp(logits): {exp_logits}")
        logger.info(f"3. Somme des exp(logits): {sum_exp}")
        logger.info(f"4. Probabilités finales: {probabilities[0].tolist()}")
        
        # Probabilités finales
        prob_non_harcelement = probabilities[0][0].item()
        prob_harcelement = probabilities[0][1].item()
        
        logger.info(f"\nRésultats finaux:")
        logger.info(f"Probabilité non-harcèlement: {prob_non_harcelement:.4f} ({prob_non_harcelement:.2%})")
        logger.info(f"Probabilité harcèlement: {prob_harcelement:.4f} ({prob_harcelement:.2%})")
        
        # Décision
        prediction = 1 if prob_harcelement > prob_non_harcelement else 0
        logger.info(f"\nDécision: {'Harcèlement' if prediction == 1 else 'Non-harcèlement'}")

def main():
    try:
        model_path = 'results/run_20250606_002431/final_model'
        
        # Exemples de textes à analyser
        textes = [
            "jules t'es super laid",
            "bonjour comment vas-tu",
            "tu es vraiment nul",
            "merci pour ton aide"
        ]
        
        for texte in textes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyse du texte: {texte}")
            logger.info(f"{'='*50}")
            analyze_prediction(texte, model_path)
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        raise

if __name__ == "__main__":
    main() 