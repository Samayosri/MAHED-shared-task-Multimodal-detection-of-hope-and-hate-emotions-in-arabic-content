# MARBERT Fine-Tuned Multi-Head Multi-Task Classification
# One shared MARBERT backbone + Emotion/Offensive/Hate heads
# MARBERT is fine-tuned end-to-end.
import re, unicodedata, random
import numpy as np, pandas as pd, torch, torch.nn as nn, nltk
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
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

# 4. Labels
LABEL_COLS = ["Emotion", "Offensive", "Hate"]
label_mappings, num_classes = {}, {}

for label in LABEL_COLS:
    unique_labels = sorted(df[label].dropna().unique())
    label2id = {v: i for i, v in enumerate(unique_labels)}
    id2label = {i: v for v, i in label2id.items()}
    label_mappings[label] = {"label2id": label2id, "id2label": id2label}
    num_classes[label] = len(unique_labels)
    print(f"{label}: {unique_labels}")

# 5. Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_mappings, max_length=64):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer, self.label_mappings, self.max_length = tokenizer, label_mappings, max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(row["clean_text"], padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        # Missing labels = -1 (ignored during loss)
        for label in LABEL_COLS:
            value = row[label]
            item[f"{label}_label"] = torch.tensor(
                -1 if pd.isna(value) else self.label_mappings[label]["label2id"][value],
                dtype=torch.long)
        return item

# 6. Multi-head MARBERT
class MARBERTMultiHead(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.marbert = AutoModel.from_pretrained(MARBERT_MODEL_NAME)
        hidden_size = self.marbert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Three task-specific heads
        self.emotion_head = nn.Linear(hidden_size, num_classes["Emotion"])
        self.offensive_head = nn.Linear(hidden_size, num_classes["Offensive"])
        self.hate_head = nn.Linear(hidden_size, num_classes["Hate"])

    def mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # ONE shared MARBERT forward pass
        outputs = self.marbert(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        pooled = self.dropout(self.mean_pool(outputs.last_hidden_state, attention_mask))
        return {"Emotion": self.emotion_head(pooled),
                "Offensive": self.offensive_head(pooled),
                "Hate": self.hate_head(pooled)}

# 7. Calculate class weights
def get_class_weights(dataframe, label):
    subset = dataframe[dataframe[label].notna()]
    labels = subset[label].values
    classes = np.array(sorted(pd.unique(labels)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    label2id = label_mappings[label]["label2id"]
    weight_tensor = torch.ones(num_classes[label], dtype=torch.float)
    for cls, weight in zip(classes, weights):
        weight_tensor[label2id[cls]] = weight
    return weight_tensor.to(device)

# 8. Multi-task weighted loss
def calculate_loss(outputs, batch, class_weights):
    total_loss, active_tasks = 0.0, 0
    for label in LABEL_COLS:
        labels = batch[f"{label}_label"]
        valid_mask = labels != -1                # Only samples with valid labels
        if valid_mask.sum() == 0: continue
        loss_fn = nn.CrossEntropyLoss(weight=class_weights[label])
        total_loss += loss_fn(outputs[label][valid_mask], labels[valid_mask])
        active_tasks += 1
    return None if active_tasks == 0 else total_loss / active_tasks

# 9. Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = {l: [] for l in LABEL_COLS}
    all_labels = {l: [] for l in LABEL_COLS}
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            token_type_ids=batch.get("token_type_ids"))
            for label in LABEL_COLS:
                labels = batch[f"{label}_label"]
                valid_mask = labels != -1
                if valid_mask.sum() == 0: continue
                preds = torch.argmax(outputs[label][valid_mask], dim=1)
                all_preds[label].extend(preds.cpu().numpy())
                all_labels[label].extend(labels[valid_mask].cpu().numpy())

    results, macro_f1_scores = {}, []
    for label in LABEL_COLS:
        if len(all_labels[label]) == 0: continue
        results[label] = {
            "accuracy": accuracy_score(all_labels[label], all_preds[label]),
            "macro_f1": f1_score(all_labels[label], all_preds[label], average="macro")}
        macro_f1_scores.append(results[label]["macro_f1"])
    results["average_macro_f1"] = np.mean(macro_f1_scores)
    return results

# 10. Fine-tune MARBERT
def train_model(model, train_loader, val_loader, class_weights,
                epochs=3, learning_rate=2e-5, warmup_ratio=0.1):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_state = -np.inf, None

    for epoch in range(epochs):
        # ---- Training ----
        model.train(); total_train_loss, train_steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch.get("token_type_ids"))
                loss = calculate_loss(outputs, batch, class_weights)
            if loss is None: continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_train_loss += loss.item(); train_steps += 1
        avg_train_loss = total_train_loss / max(train_steps, 1)

        # ---- Validation ----
        val_results = validate_model(model, val_loader)
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        for label in LABEL_COLS:
            if label in val_results:
                print(f"{label:<10} | Acc: {val_results[label]['accuracy']:.4f} | "
                      f"Macro-F1: {val_results[label]['macro_f1']:.4f}")
        avg_f1 = val_results["average_macro_f1"]
        print(f"Average Macro-F1: {avg_f1:.4f}")

        # ---- Save best ----
        if avg_f1 > best_score:
            best_score = avg_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("✓ Best model updated")

    if best_state is not None: model.load_state_dict(best_state)
    return model.to(device)

# 11. Final test evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = {l: [] for l in LABEL_COLS}
    all_labels = {l: [] for l in LABEL_COLS}
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            token_type_ids=batch.get("token_type_ids"))
            for label in LABEL_COLS:
                labels = batch[f"{label}_label"]
                valid_mask = labels != -1
                if valid_mask.sum() == 0: continue
                preds = torch.argmax(outputs[label][valid_mask], dim=1)
                all_preds[label].extend(preds.cpu().numpy())
                all_labels[label].extend(labels[valid_mask].cpu().numpy())

    results = []
    for label in LABEL_COLS:
        accuracy = accuracy_score(all_labels[label], all_preds[label])
        macro_f1 = f1_score(all_labels[label], all_preds[label], average="macro")
        class_names = [label_mappings[label]["id2label"][i] for i in range(num_classes[label])]
        print("\n" + "=" * 65)
        print(f"MARBERT Fine-Tuned Multi-Head: {label}")
        print("=" * 65)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro-F1 : {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels[label], all_preds[label],
                                    target_names=class_names, zero_division=0))
        results.append({"model": "MARBERT Fine-Tuned Multi-Head", "label": label,
                        "accuracy": accuracy, "macro_f1": macro_f1})
    return results

# 12. Train / validation / test split
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=SEED)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)
print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# 13. Class weights
class_weights = {label: get_class_weights(train_df, label) for label in LABEL_COLS}
for label in LABEL_COLS:
    print(f"\n{label} class weights:")
    print(class_weights[label].detach().cpu().numpy())

# 14. Datasets
BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_LENGTH = 16, 3, 2e-5, 64

train_dataset = MultiTaskDataset(train_df, tokenizer, label_mappings, MAX_LENGTH)
val_dataset = MultiTaskDataset(val_df, tokenizer, label_mappings, MAX_LENGTH)
test_dataset = MultiTaskDataset(test_df, tokenizer, label_mappings, MAX_LENGTH)

# 15. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 16. Create ONE MARBERT multi-head model
model = MARBERTMultiHead(num_classes=num_classes, dropout=0.3).to(device)
print("\nModel:"); print(model)

# 17. Fine-tune MARBERT end-to-end
model = train_model(model, train_loader, val_loader, class_weights=class_weights,
                    epochs=EPOCHS, learning_rate=LEARNING_RATE, warmup_ratio=0.1)

# 18. Evaluate
results = evaluate_model(model, test_loader)

# 19. Final results
results_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(results_df)