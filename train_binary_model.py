import pandas as pd
import numpy as np
from transformers import (
    CamembertTokenizer, CamembertForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA availability
logger.info(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class HarassmentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    
    # Calculate class weights based on inverse frequency
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(model.device)
    
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits, labels)
    
    return (loss, outputs) if return_outputs else loss

def main():
    # Create timestamp for unique run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./results/binary/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    df = pd.read_csv('data/dataset_v2.csv', sep=';')
    
    # Basic text preprocessing
    df['text'] = df['text'].str.lower()
    logger.info(f"Total examples: {len(df)}")
    
    texts = df['text'].values
    labels = df['label'].values
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create datasets
    train_dataset = HarassmentDataset(train_texts, train_labels, tokenizer)
    val_dataset = HarassmentDataset(val_texts, val_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,  
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.2,
        weight_decay=0.02,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_total_limit=3,
        seed=42,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )
    
    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate final model
    logger.info("Evaluating final model...")
    final_metrics = trainer.evaluate()
    logger.info("\nFinal metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Génération et sauvegarde de la matrice de confusion
    logger.info("Calcul de la matrice de confusion sur le jeu de validation...")
    # Prédictions sur le jeu de validation
    val_preds_output = trainer.predict(val_dataset)
    y_true = val_preds_output.label_ids
    y_pred = val_preds_output.predictions.argmax(-1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-harcèlement', 'Harcèlement'], yticklabels=['Non-harcèlement', 'Harcèlement'])
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.title('Matrice de confusion (validation)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
    plt.close()

    # Statistiques détaillées
    report = classification_report(y_true, y_pred, target_names=['Non-harcèlement', 'Harcèlement'])
    with open(f'{output_dir}/confusion_matrix_stats.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info("Matrice de confusion et statistiques sauvegardées dans le dossier de sortie.")
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    model_save_path = f'{output_dir}/final_model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 