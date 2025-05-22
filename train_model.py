import pandas as pd
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Classe pour le dataset PyTorch
class HarassmentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('dataset_harcelement.csv')
    
    # Préparer les données
    texts = df['text'].values
    labels = df['labels'].values
    
    # Diviser en ensembles d'entraînement et de test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialiser le tokenizer et le modèle
    print("Initialisation du modèle CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    
    # Créer les datasets
    train_dataset = HarassmentDataset(train_texts, train_labels, tokenizer)
    val_dataset = HarassmentDataset(val_texts, val_labels, tokenizer)
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
    )
    
    # Initialiser le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    trainer.train()
    
    # Sauvegarder le modèle
    print("Sauvegarde du modèle...")
    model.save_pretrained('./model_harcelement')
    tokenizer.save_pretrained('./model_harcelement')
    
    print("Entraînement terminé !")

if __name__ == "__main__":
    main() 