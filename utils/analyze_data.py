import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from pathlib import Path
import re
from transformers import CamembertTokenizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Fichier de données introuvable: {data_path}")
        
        # Chargement des données
        self.df = pd.read_csv(self.data_path, sep=';')
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        
    def basic_stats(self):
        """Analyse statistique de base"""
        logger.info("\n=== Statistiques de base ===")
        logger.info(f"Nombre total d'exemples: {len(self.df)}")
        
        # Distribution des classes
        class_dist = self.df['label'].value_counts()
        logger.info("\nDistribution des classes:")
        for label, count in class_dist.items():
            percentage = (count / len(self.df)) * 100
            logger.info(f"Classe {label}: {count} exemples ({percentage:.2f}%)")
            
        # Longueur des textes
        self.df['text_length'] = self.df['text'].str.len()
        logger.info("\nStatistiques de longueur des textes:")
        logger.info(self.df['text_length'].describe())
        
    def token_analysis(self):
        """Analyse des tokens"""
        logger.info("\n=== Analyse des tokens ===")
        
        # Tokenization de tous les textes
        all_tokens = []
        for text in self.df['text']:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Statistiques des tokens
        token_counter = Counter(all_tokens)
        logger.info(f"Nombre total de tokens: {len(all_tokens)}")
        logger.info(f"Nombre de tokens uniques: {len(token_counter)}")
        
        # Top 20 tokens les plus fréquents
        logger.info("\nTop 20 tokens les plus fréquents:")
        for token, count in token_counter.most_common(20):
            logger.info(f"{token}: {count}")
            
    def visualize_distributions(self):
        """Visualisation des distributions"""
        # Création du dossier pour les graphiques
        output_dir = Path('results/analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distribution des classes
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='label')
        plt.title('Distribution des Classes')
        plt.savefig(output_dir / 'class_distribution.png')
        plt.close()
        
        # Distribution des longueurs de texte
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='text_length', hue='label', multiple="stack")
        plt.title('Distribution des Longueurs de Texte par Classe')
        plt.savefig(output_dir / 'text_length_distribution.png')
        plt.close()
        
    def analyze_by_class(self):
        """Analyse détaillée par classe"""
        logger.info("\n=== Analyse par classe ===")
        
        for label in self.df['label'].unique():
            class_df = self.df[self.df['label'] == label]
            logger.info(f"\nClasse {label}:")
            logger.info(f"Nombre d'exemples: {len(class_df)}")
            logger.info(f"Longueur moyenne des textes: {class_df['text_length'].mean():.2f}")
            logger.info(f"Longueur médiane des textes: {class_df['text_length'].median():.2f}")
            
            # Top 10 mots les plus fréquents pour cette classe
            all_tokens = []
            for text in class_df['text']:
                tokens = self.tokenizer.tokenize(text)
                all_tokens.extend(tokens)
            
            token_counter = Counter(all_tokens)
            logger.info("\nTop 10 tokens les plus fréquents:")
            for token, count in token_counter.most_common(10):
                logger.info(f"{token}: {count}")

def main():
    try:
        # Initialisation de l'analyseur
        analyzer = DataAnalyzer('data/dataset_v2.csv')
        
        # Exécution des analyses
        analyzer.basic_stats()
        analyzer.token_analysis()
        analyzer.visualize_distributions()
        analyzer.analyze_by_class()
        
        logger.info("\nAnalyse terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        raise

if __name__ == "__main__":
    main() 