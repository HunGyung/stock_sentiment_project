import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# add the project root directory to Python path for relative path recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.preprocessor import clean_news_text

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME      = "monologg/koelectra-base-v3-discriminator"
MAX_LEN         = 128
BATCH_SIZE      = 16        # Reduced batch size for local training to prevent OOM
EPOCHS          = 15        # Maximum epochs (early stopping will stop training if no improvement)
PATIENCE        = 3         # Stop if validation F1 does not improve for 3 epochs
LEARNING_RATE   = 2e-5      # Best verified learning rate (1e-5 is insufficient for convergence)
WEIGHT_DECAY    = 0.01      # Regularization to prevent overfitting
LABEL_SMOOTHING = 0.1       # Mitigate overfitting on small dataset
SEED            = 42        # Random seed for reproducibility
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR        = "data/model_save"
DATA_PATH       = "data/finance_data.csv"
F1_GOAL         = 0.85


# ── Set Random Seed ────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    """Set random seeds for python, numpy, and pytorch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure cuDNN deterministic behavior (may slightly reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Random seed fixed: {seed}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class FinancialSentimentDataset(Dataset):
    """PyTorch Dataset to format inputs for the KoELECTRA model."""

    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        max_len: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Train / Eval Functions ─────────────────────────────────────────────────────
def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> float:
    """Train the model for one epoch and return the average train loss."""
    model.train()
    total_loss = 0.0
    loss_fct   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for batch in data_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        model.zero_grad()
        # Compute loss manually on logits to support label smoothing
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fct(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)


def eval_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model on validation data and return val_loss, accuracy, macro_f1."""
    model.eval()
    predictions, real_values = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    val_loss = total_loss / len(data_loader)
    acc      = accuracy_score(real_values, predictions)
    f1       = f1_score(real_values, predictions, average="macro")
    return val_loss, acc, f1


# ── Inference Test ────────────────────────────────────────────────────────────
def run_inference_test(model_dir: str = SAVE_DIR) -> None:
    """Run test inference on saved model (equivalent verification to sentiment.py)."""
    print("\n" + "=" * 60)
    print("Inference Test (sentiment.py equivalence check)")
    print("=" * 60)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    test_sentences = [
        "삼성전자, 역대급 실적 발표에 주가 급등 상한가 기록!",
        "오늘 주식 시장은 별다른 소식 없이 보합세로 마감했습니다.",
        "글로벌 경기 침체 우려로 인해 외국인들이 주식을 대거 매도하며 주가가 급락했습니다.",
    ]

    for sent in test_sentences:
        cleaned = clean_news_text(sent)
        inputs  = tokenizer(
            cleaned, return_tensors="pt",
            max_length=128, padding="max_length", truncation=True,
        ).to(device)

        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1).squeeze(0)

        score = (0.5 * probs[1] + probs[2]).item()
        label = "POSITIVE" if score > 0.6 else ("NEGATIVE" if score < 0.4 else "NEUTRAL")
        print(f"  Sentence : {sent}")
        print(f"  Score    : {score:.4f}  [{label}]")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Execute the full training pipeline."""
    # Ensure reproducibility
    set_seed(SEED)

    print(f"Device        : {DEVICE}")
    print(f"Model         : {MODEL_NAME}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"F1 Goal       : {F1_GOAL}")
    print()

    # ── Load Data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_num"] = df["labels"].map(label_map)

    print("Preprocessing text...")
    df["cleaned_sentence"] = df["kor_sentence"].apply(clean_news_text)

    # ── Train / Validation Split (80 / 20) ────────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    train_texts  = train_df["cleaned_sentence"].values
    train_labels = train_df["label_num"].values
    val_texts    = val_df["cleaned_sentence"].values
    val_labels   = val_df["label_num"].values

    print(f"Train: {len(train_texts):,} samples | Val: {len(val_texts):,} samples")
    print(f"Class distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    # ── Tokenizer & Dataset ───────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset   = FinancialSentimentDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)

    # WeightedRandomSampler to handle class imbalance
    class_weights  = 1.0 / np.bincount(train_labels)
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Full Fine-Tuning -- Total parameters: {total_params:,}")
    model = model.to(DEVICE)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ── Training Loop (with Early Stopping) ───────────────────────────────────
    best_f1        = 0.0
    no_improve_cnt = 0   # Early stopping counter
    history        = []

    print("\n" + "=" * 60)
    print(f"Training Started! (max {EPOCHS} epochs, patience={PATIENCE})")
    print("=" * 60)
    t0 = time.time()

    for epoch in range(EPOCHS):
        te         = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE,
            label_smoothing=LABEL_SMOOTHING,
        )
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, DEVICE)
        elapsed = time.time() - te

        history.append(
            dict(epoch=epoch + 1, train_loss=train_loss,
                 val_loss=val_loss, val_acc=val_acc, val_f1=val_f1)
        )

        improved = val_f1 > best_f1
        tag      = " <-- BEST" if improved else f"  (no improve {no_improve_cnt + 1}/{PATIENCE})"
        print(
            f"Epoch {epoch + 1}/{EPOCHS} ({elapsed:.0f}s)  "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}{tag}"
        )
        print("-" * 60)

        if improved:
            best_f1        = val_f1
            no_improve_cnt = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  -> Best model saved to {SAVE_DIR} (F1={best_f1:.4f})")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= PATIENCE:
                print(f"\n[Early Stopping] {PATIENCE} epochs with no improvement. Stopping training.")
                break

    total_elapsed = time.time() - t0
    print(f"\nTotal training time: {total_elapsed / 60:.1f} minutes")
    print(f"Best Validation Macro F1: {best_f1:.4f}")
    print("=" * 60)

    # ── Target Verification ───────────────────────────────────────────────────
    if best_f1 >= F1_GOAL:
        print(f"[GOAL ACHIEVED] Best F1 ({best_f1:.4f}) >= Target ({F1_GOAL})")
    else:
        print(f"[GOAL NOT MET]  Best F1 ({best_f1:.4f}) < Target ({F1_GOAL})")
        print("  -> Consider increasing EPOCHS or tuning LEARNING_RATE.")

    # ── History Summary ───────────────────────────────────────────────────────
    print("\n[Training History Summary]")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} {'Acc':>7} {'MacroF1':>9}")
    print("-" * 45)
    for row in history:
        print(
            f"{row['epoch']:>5} {row['train_loss']:>10.4f} {row['val_loss']:>9.4f} "
            f"{row['val_acc']:>7.4f} {row['val_f1']:>9.4f}"
        )

    # ── Run Inference Test ────────────────────────────────────────────────────
    run_inference_test(SAVE_DIR)


if __name__ == "__main__":
    main()
