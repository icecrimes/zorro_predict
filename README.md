# Zorro Predict - Modèle de prédiction pour la détection de Harcèlement dans le cadre d'un PFE avec l'association Zorro Kids

## Aperçu

Ce projet implémente un système de classification de texte capable de détecter le harcèlement dans les textes en français. Il utilise le modèle CamemBERT, un modèle de langue française basé sur RoBERTa, fine-tuné pour la classification binaire des textes en harcèlement ou non-harcèlement, ainsi que pour la classification multi-classes des types de harcèlement en 5 classes différents : homophobie, injure, physique, racisme, religion et sexisme


## Fonctionnalités

- Modèle CamemBERT fine-tuné pour la classification de texte en français
- Classification binaire (harcèlement vs non-harcèlement)
- Classification multi-classes des types de harcèlement
- Scores de probabilité pour les prédictions
- Arrêt anticipé et sauvegarde des points de contrôle du modèle
- Métriques d'évaluation complètes (précision, F1, rappel)
- Génération automatique de matrices de confusion
- Prédiction combinée avec un seul script (`predict_all.py`)
- Export des résultats au format CSV

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/icecrimes/zorro_predict.git
cd zorro_predict
```

2. Créer et activer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du Projet

```
zorro_predict/
├── data/                      # Jeu de données et fichiers d'exemples
│   └── analysis/              # Analyse des datasets
│   ├── dataset_v2.csv         # Jeu de données binaire
│   ├── labels_vf.csv          # Jeu de données multi-classes
│   └── exemples.txt           # Fichier d'exemples pour les tests
├── models/                    # Points de contrôle du modèle sauvegardés
│   ├── model_binaire/         # Modèle de classification binaire
│   └── model_multiclasses/    # Modèle de classification multi-classes
├── train_binary_model.py      # Script d'entraînement du modèle binaire
├── train_class_model.py       # Script d'entraînement du modèle multi-classes
├── predict_binary.py          # Script de prédiction binaire
├── predict_class.py           # Script de prédiction multi-classes
├── predict_all.py             # Script de prédiction combinée
└── requirements.txt           # Dépendances du projet
```

## Utilisation

### Entraînement des Modèles

Pour entraîner le modèle binaire :
```bash
python train_binary_model.py
```

Pour entraîner le modèle multi-classes :
```bash
python train_class_model.py
```

Les scripts d'entraînement vont :
- Charger et prétraiter le jeu de données
- Diviser les données en ensembles d'entraînement et de validation
- Entraîner le modèle avec arrêt anticipé
- Sauvegarder le meilleur point de contrôle du modèle
- Générer les métriques d'évaluation et la matrice de confusion

### Téléchargement des Modèles

Si l'entraînement est trop long, les modèles sont disponibles sur le drive partagé à l'URL indiqué dans ./models/

Dézipper les dossiers model_binaire/ et model_multiclasses/ dans ./models/

### Faire des Prédictions

### Prédiction Combinée
Le script `predict_all.py` combine les prédictions binaires et multi-classes en une seule étape :

```bash
python predict_all.py
```

Le script génère un fichier `resultats.csv` contenant :
- `texte` : Le texte analysé
- `is_harassment` : Prédiction binaire (0/1)
- `confiance_binaire` : Score de confiance de la prédiction binaire (arrondi à 2 décimales)
- `prediction_type` : Type de harcèlement (si is_harassment = 1)
- `confiance_type` : Score de confiance du type (si is_harassment = 1)

### Prédiction Binaire
Pour uniquement la détection du harcèlement :
```bash
python predict_binary.py 
```

### Prédiction Multi-classes
Pour uniquement la classification des types de harcèlement :
```bash
python predict_class.py 
```

Les scripts de prédiction vont :
- Charger le(s) modèle(s) entraîné(s)
- Traiter le texte d'entrée
- Produire une prédiction pour chaque ligne du fichier d'entrée
- Fournir les scores de probabilité

## Concact
- tranalexan@cy-tech.fr
- raffallilo@cy-tech.fr 
