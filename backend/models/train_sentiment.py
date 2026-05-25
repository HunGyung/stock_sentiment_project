import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# add the project root directory to Python path for relative path recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.preprocessor import clean_news_text

# hyperparameter settings
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "data/model_save"

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # get texts and label
        text = str(self.texts[idx])
        label = self.labels[idx]

        # perfomrs tokenizing and padding/truncation(length limiting)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # transforms output as a 2D tensor to 1D(flatten) and returns it
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
    val_loss = total_loss / len(data_loader)
    acc = accuracy_score(real_values, predictions)
    f1 = f1_score(real_values, predictions, average='macro')
    
    return val_loss, acc, f1

def main():
    print(f"Using device: {DEVICE}")
    
    # loads data and mapping the label
    df = pd.read_csv("data/finance_data.csv")
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label_num'] = df['labels'].map(label_map)
    
    # preprocessing text
    print("Preprocessing text...")
    df['cleaned_sentence'] = df['kor_sentence'].apply(clean_news_text)
    
    # 3. Train / Validation split (80% / 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_num'])
    
    train_texts = train_df['cleaned_sentence'].values
    train_labels = train_df['label_num'].values
    val_texts = val_df['cleaned_sentence'].values
    val_labels = val_df['label_num'].values
    
    # loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # creates Dataset and DataLoader
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # settings WeightedRandomSampler (resolving class imvalance)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    # Validation does not strictly require sampling, so it is loaded in order without a standard shuffle. 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # loads classification model
    print(f"Loading pretrained model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Layer Freezing (freeze lower 6 encoder layer and embedding)
    print("Applying Layer Freezing (Embedding & Encoder Layers 0 to 5)...")
    # freezes Embedding layer
    for param in model.electra.embeddings.parameters():
        param.requires_grad = False
    # freezes lower 6 encoder layer
    for i in range(3):
        for param in model.electra.encoder.layer[i].parameters():
            param.requires_grad = False
            
    # check the number of parameters to be trained
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")
    
    model = model.to(DEVICE)
    
    # Optimizer & Scheduler settings
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training Loop
    best_f1 = 0.0
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print("-" * 50)
        
        # Saves the weight with the best performance based on the Validation F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(SAVE_DIR, exist_ok=True)
            # Saves the tokenizer and the model in Hugging Face style
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"New best model saved to {SAVE_DIR} (F1: {best_f1:.4f})")
            
    print(f"\nTraining Complete! Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
