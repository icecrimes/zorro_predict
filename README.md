# Zorro Predict - Modèle de prédiction pour la détection de Harcèlement dans le cadre d'un PFE avec l'association Zorro Kids

## Aperçu

Ce projet implémente un système de classification de texte capable de détecter le harcèlement dans les textes en français. Il utilise le modèle CamemBERT, un modèle de langue française basé sur RoBERTa, fine-tuné pour la classification binaire des textes en harcèlement ou non-harcèlement.

## Fonctionnalités

- Modèle CamemBERT fine-tuné pour la classification de texte en français
- Classification binaire (harcèlement vs non-harcèlement)
- Scores de probabilité pour les prédictions
- Arrêt anticipé et sauvegarde des points de contrôle du modèle
- Métriques d'évaluation complètes (précision, F1, rappel)

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
├── data/              # Jeu de données et fichiers d'exemples
├── models/            # Points de contrôle du modèle sauvegardés
├── results/           # Résultats d'entraînement et logs
├── utils/             # Fonctions utilitaires
├── train_model.py     # Script d'entraînement du modèle
├── predict.py         # Script de prédiction
└── requirements.txt   # Dépendances du projet
```

## Utilisation

### Entraînement du Modèle

Pour entraîner le modèle :

```bash
python train_model.py
```

Le script d'entraînement va :
- Charger et prétraiter le jeu de données
- Diviser les données en ensembles d'entraînement et de validation
- Entraîner le modèle avec arrêt anticipé
- Sauvegarder le meilleur point de contrôle du modèle
- Générer les métriques d'évaluation

### Téléchargement du Modèle

Si l'entraînement est trop long, un modèle est disponible sur le drive partagé à l'URL indiqué dans ./models/
Dézipper le dossier model_v2/ dans ./models/

### Faire des Prédictions

Pour faire des prédictions sur de nouveaux textes :

```bash
python predict.py
```

Le script de prédiction va :
- Charger le modèle entraîné
- Traiter le texte d'entrée, par défaut : data/exemples.txt (modifiable)
- Produire une prédiction (harcèlement/non-harcèlement) pour chaque ligne du fichier .txt
- Fournir les scores de probabilité pour les deux classes
