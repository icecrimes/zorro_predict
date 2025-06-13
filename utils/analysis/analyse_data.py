import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
import os
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_output_files():
    """Configure les fichiers de sortie pour les résultats"""
    # Créer le dossier analysis s'il n'existe pas
    os.makedirs('analysis', exist_ok=True)
    
    # Créer un fichier texte pour les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'analysis/analysis_results_{timestamp}.txt'
    
    # Configurer le logging pour écrire dans le fichier
    file_handler = logging.FileHandler(results_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return results_file

def analyze_dataset_v2():
    """Analyse du fichier dataset_v2.csv"""
    logger.info("\n" + "="*50)
    logger.info("ANALYSE DU FICHIER DATASET_V2.CSV")
    logger.info("="*50)
    
    # Charger les données
    df = pd.read_csv('data/dataset_v2.csv', sep=';')
    
    # Statistiques de base
    logger.info(f"\nNombre total d'exemples: {len(df)}")
    logger.info(f"Nombre de colonnes: {len(df.columns)}")
    logger.info("\nColonnes présentes:")
    for col in df.columns:
        logger.info(f"- {col}")
    
    # Distribution des labels
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        logger.info("\nDistribution des labels:")
        for label, count in label_counts.items():
            logger.info(f"Label {label}: {count} exemples ({count/len(df)*100:.2f}%)")
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='label')
        plt.title('Distribution des Labels - Dataset V2')
        plt.xlabel('Label')
        plt.ylabel('Nombre d\'exemples')
        plt.savefig('analysis/dataset_v2_label_distribution.png')
        plt.close()
    
    # Statistiques sur les textes
    df['text_length'] = df['text'].str.len()
    logger.info("\nStatistiques sur la longueur des textes:")
    logger.info(f"Longueur moyenne: {df['text_length'].mean():.2f} caractères")
    logger.info(f"Longueur minimale: {df['text_length'].min()} caractères")
    logger.info(f"Longueur maximale: {df['text_length'].max()} caractères")
    
    # Visualisation de la distribution des longueurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Distribution de la Longueur des Textes - Dataset V2')
    plt.xlabel('Longueur (caractères)')
    plt.ylabel('Nombre d\'exemples')
    plt.savefig('analysis/dataset_v2_text_length_distribution.png')
    plt.close()

def analyze_labels_vf():
    """Analyse du fichier labels_vf.csv"""
    logger.info("\n" + "="*50)
    logger.info("ANALYSE DU FICHIER LABELS_VF.CSV")
    logger.info("="*50)
    
    # Charger les données
    df = pd.read_csv('data/labels_vf.csv', sep=';')
    
    # Statistiques de base
    logger.info(f"\nNombre total d'exemples: {len(df)}")
    logger.info(f"Nombre de colonnes: {len(df.columns)}")
    logger.info("\nColonnes présentes:")
    for col in df.columns:
        logger.info(f"- {col}")
    
    # Distribution des labels
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        logger.info("\nDistribution des labels:")
        for label, count in label_counts.items():
            logger.info(f"Label {label}: {count} exemples ({count/len(df)*100:.2f}%)")
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='label')
        plt.title('Distribution des Labels - Labels VF')
        plt.xlabel('Label')
        plt.ylabel('Nombre d\'exemples')
        plt.savefig('analysis/labels_vf_label_distribution.png')
        plt.close()
    
    # Distribution des types
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        logger.info("\nDistribution des types:")
        for type_, count in type_counts.items():
            logger.info(f"Type {type_}: {count} exemples ({count/len(df)*100:.2f}%)")
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='type')
        plt.title('Distribution des Types - Labels VF')
        plt.xlabel('Nombre d\'exemples')
        plt.ylabel('Type')
        plt.tight_layout()
        plt.savefig('analysis/labels_vf_type_distribution.png')
        plt.close()
    
    # Distribution des sources
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        logger.info("\nDistribution des sources:")
        for source, count in source_counts.items():
            logger.info(f"Source {source}: {count} exemples ({count/len(df)*100:.2f}%)")
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='source')
        plt.title('Distribution des Sources - Labels VF')
        plt.xlabel('Nombre d\'exemples')
        plt.ylabel('Source')
        plt.tight_layout()
        plt.savefig('analysis/labels_vf_source_distribution.png')
        plt.close()
    
    # Statistiques sur les textes
    df['text_length'] = df['text'].str.len()
    logger.info("\nStatistiques sur la longueur des textes:")
    logger.info(f"Longueur moyenne: {df['text_length'].mean():.2f} caractères")
    logger.info(f"Longueur minimale: {df['text_length'].min()} caractères")
    logger.info(f"Longueur maximale: {df['text_length'].max()} caractères")
    
    # Visualisation de la distribution des longueurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Distribution de la Longueur des Textes - Labels VF')
    plt.xlabel('Longueur (caractères)')
    plt.ylabel('Nombre d\'exemples')
    plt.savefig('analysis/labels_vf_text_length_distribution.png')
    plt.close()
    
    # Analyse des mots les plus fréquents
    if 'text' in df.columns:
        all_words = ' '.join(df['text'].str.lower()).split()
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20)
        
        logger.info("\n20 mots les plus fréquents:")
        for word, count in most_common:
            logger.info(f"{word}: {count} occurrences")
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        words, counts = zip(*most_common)
        sns.barplot(x=list(counts), y=list(words))
        plt.title('20 Mots les Plus Fréquents - Labels VF')
        plt.xlabel('Nombre d\'occurrences')
        plt.ylabel('Mot')
        plt.tight_layout()
        plt.savefig('analysis/labels_vf_most_common_words.png')
        plt.close()

def main():
    # Configurer les fichiers de sortie
    results_file = setup_output_files()
    
    # Ajouter un en-tête au fichier de résultats
    logger.info("ANALYSE DES DONNÉES DE HARCÈLEMENT")
    logger.info("="*50)
    logger.info(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50 + "\n")
    
    # Analyser les deux fichiers
    analyze_dataset_v2()
    analyze_labels_vf()
    
    # Message de fin
    logger.info("\n" + "="*50)
    logger.info("ANALYSE TERMINÉE")
    logger.info(f"Les résultats ont été sauvegardés dans : {results_file}")
    logger.info("Les graphiques ont été sauvegardés dans le dossier : analysis/")
    logger.info("="*50)

if __name__ == "__main__":
    main() 