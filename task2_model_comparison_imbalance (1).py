# MARBERT Fine-Tuning for Multiclass Classification
import re, unicodedata, random
import numpy as np, pandas as pd, torch, torch.nn as nn, nltk
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, AdamW
from arabert.preprocess import ArabertPreprocessor

nltk.download("stopwords", quiet=True)

# 1. Reproducibility + device
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Load data + preprocessing
df = pd.read_csv("train.csv")
print("Dataset shape:", df.shape); print(df.head())

ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02-twitter"
_arabert_preprocessor = ArabertPreprocessor(model_name=ARABERT_MODEL_NAME)

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)                   # Remove URLs
    text = re.sub(r'@\S+', ' ', text)                                     # Remove mentions
    text = re.sub(r'(.)\1{2,}', r'\1', text)                              # Reduce repeated chars
    text = "".join(c for c in text if unicodedata.category(c) != "So")    # Remove symbols/emojis
    return _arabert_preprocessor.preprocess(text)                         # AraBERT preprocessing

df["clean_text"] = df["text"].apply(preprocess_text)
print(df[["text", "clean_text"]].head())

# 3. MARBERT tokenizer
MARBERT_MODEL_NAME = "UBC-NLP/MARBERT"
tokenizer = AutoTokenizer.from_pretrained(MARBERT_MODEL_NAME)

# 4. Dataset
class ArabicTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 5. MARBERT classification model
class MARBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.marbert = AutoModel.from_pretrained(MARBERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.marbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.marbert(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_embedding)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# 6. Train one MARBERT classifier
def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_f1, best_state = 0.0, None

    for epoch in range(epochs):
        # ---- Training ----
        model.train(); total_train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            token_type_ids=batch.get("token_type_ids"), labels=batch["labels"])
            loss = outputs["loss"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validation ----
        model.eval(); all_preds, all_labels, total_val_loss = [], [], 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch.get("token_type_ids"), labels=batch["labels"])
                total_val_loss += outputs["loss"].item()
                preds = torch.argmax(outputs["logits"], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = accuracy_score(all_labels, all_preds)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None: model.load_state_dict(best_state)
    return model.to(device)

# 7. Evaluate model
def evaluate_model(model, test_loader, label_name, class_names):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            token_type_ids=batch.get("token_type_ids"))
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("\n" + "=" * 60)
    print(f"MARBERT Fine-Tuning: {label_name}")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    return accuracy, macro_f1

# 8. Train separate model for each target
LABEL_COLS = ["Emotion", "Offensive", "Hate"]
results = []
BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_LENGTH = 16, 3, 2e-5, 64

for label in LABEL_COLS:
    print("\n" + "#" * 70)
    print(f"Training MARBERT for: {label}")
    print("#" * 70)

    subset = df[df[label].notna()].copy()
    texts, labels = subset["clean_text"].values, subset[label].values

    unique_labels = sorted(pd.unique(labels))
    label2id = {name: idx for idx, name in enumerate(unique_labels)}
    id2label = {idx: name for name, idx in label2id.items()}
    encoded_labels = np.array([label2id[name] for name in labels])
    class_names = [id2label[i] for i in range(len(id2label))]
    print("Classes:", class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.20, random_state=SEED, stratify=encoded_labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=SEED, stratify=y_train)
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    train_dataset = ArabicTextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = ArabicTextDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = ArabicTextDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classifier = MARBERTClassifier(num_classes=len(class_names), dropout=0.3).to(device)
    classifier = train_model(classifier, train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    accuracy, macro_f1 = evaluate_model(classifier, test_loader, label, class_names)
    results.append({"model": "MARBERT Fine-Tuned", "label": label,
                    "accuracy": accuracy, "macro_f1": macro_f1})

# 9. Final results
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df)