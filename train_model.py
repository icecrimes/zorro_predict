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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA availability
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

class HarassmentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
    output_dir = f'./results/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    df = pd.read_csv('dataset/dataset_v2.csv', sep=';')
    
    # Analyze class distribution
    logger.info("\n=== Class Distribution Analysis ===")
    class_counts = df['label'].value_counts()
    class_percentages = (class_counts / len(df) * 100).round(2)
    
    logger.info("\nClass counts:")
    for label, count in class_counts.items():
        logger.info(f"Class {label}: {count} examples")
    
    logger.info("\nClass percentages:")
    for label, percentage in class_percentages.items():
        logger.info(f"Class {label}: {percentage}%")
    
    # Calculate class weights for imbalanced dataset
    total_samples = len(df)
    class_weights = {
        label: total_samples / (len(class_counts) * count)
        for label, count in class_counts.items()
    }
    logger.info("\nCalculated class weights for balancing:")
    for label, weight in class_weights.items():
        logger.info(f"Class {label}: {weight:.2f}")
    
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
    model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Changed to 2 labels since we're doing binary classification
    
    # Create datasets
    train_dataset = HarassmentDataset(train_texts, train_labels, tokenizer)
    val_dataset = HarassmentDataset(val_texts, val_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,  
        per_device_train_batch_size=32,  # Reduced from 64 to 32 for better stability
        per_device_eval_batch_size=32,
        learning_rate=2e-5,  # Increased from 1e-5 to 2e-5
        warmup_ratio=0.2,  # Increased from 0.1 to 0.2
        weight_decay=0.02,  # Increased from 0.01 to 0.02
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
        gradient_accumulation_steps=2,  # Increased from 1 to 2
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
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    model_save_path = f'{output_dir}/final_model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 