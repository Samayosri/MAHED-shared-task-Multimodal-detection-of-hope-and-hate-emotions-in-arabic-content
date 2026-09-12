# MARBERT + 2x BiLSTM + Attention + Multi-Head Multi-Task
import re, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from arabert.preprocess import ArabertPreprocessor

# 1. SEED
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. CONFIGURATION
MARBERT_MODEL_NAME = "UBC-NLP/MARBERT"
MAX_LENGTH, BATCH_SIZE, EPOCHS = 128, 16, 8
MARBERT_LR, LSTM_LR = 1e-5, 2e-4
WEIGHT_DECAY, DROPOUT, LSTM_HIDDEN_SIZE, GRADIENT_CLIP = 0.01, 0.30, 256, 1.0
LOSS_WEIGHTS = {"Emotion": 1.5, "Offensive": 1.0, "Hate": 1.0}   # Emotion gets more weight

# 3. LOAD DATA
df = pd.read_csv("train.csv")
print(df.head()); print(df.columns)

# 4. PREPROCESSING
arabert_preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02-twitter")

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)              # Remove URLs
    text = re.sub(r'@\S+', '', text)                                # Remove mentions
    text = re.sub(r'(.)\1{2,}', r'\1', text)                        # Reduce repeated chars
    text = ''.join(ch for ch in text                               # Remove emoji/symbols
                   if not (0x1F300 <= ord(ch) <= 0x1FAFF))
    return arabert_preprocessor.preprocess(text)                    # AraBERT preprocessing

df["text"] = df["text"].fillna("").apply(preprocess_text)
print(df[["text", "Emotion", "Offensive", "Hate"]].head())

# 5. LABEL ENCODING
LABELS = ["Emotion", "Offensive", "Hate"]
label_maps, num_classes = {}, {}

for label in LABELS:
    df[label] = df[label].astype(str)                               # Avoid mixed-type issues
    unique_labels = sorted(df[label].unique())
    label_maps[label] = {name: idx for idx, name in enumerate(unique_labels)}
    df[label + "_label"] = df[label].map(label_maps[label])
    num_classes[label] = len(unique_labels)
    print(f"{label}: {num_classes[label]} classes"); print(label_maps[label])

# 6. TRAIN / VALIDATION / TEST SPLIT
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, shuffle=True)
print("\nDataset sizes:")
print("Train:", len(train_df)); print("Validation:", len(val_df)); print("Test:", len(test_df))

# 7. TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MARBERT_MODEL_NAME)

# 8. DATASET
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(row["text"], truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        item = {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)}
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        for label in LABELS:
            value = row[label + "_label"]
            item[label] = torch.tensor(-1 if pd.isna(value) else int(value), dtype=torch.long)
        return item

# 9. MODEL
class MARBERT_BiLSTM_MultiHead(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, dropout=0.3):
        super().__init__()
        self.marbert = AutoModel.from_pretrained(MARBERT_MODEL_NAME)
        marbert_hidden_size = self.marbert.config.hidden_size

        # Two stacked BiLSTMs
        self.lstm1 = nn.LSTM(marbert_hidden_size, lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

        # Attention
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

        # Task heads
        rep_size = lstm_hidden_size * 2
        def make_head(n): return nn.Sequential(nn.Linear(rep_size, rep_size), nn.ReLU(),
                                               nn.Dropout(dropout), nn.Linear(rep_size, n))
        self.emotion_head = make_head(num_classes["Emotion"])
        self.offensive_head = make_head(num_classes["Offensive"])
        self.hate_head = make_head(num_classes["Hate"])

    def attention_pooling(self, sequence_output, attention_mask):
        scores = self.attention(sequence_output).squeeze(-1)

        # Masking + softmax in FP32 to avoid FP16 overflow/underflow
        scores = scores.float().masked_fill(
            attention_mask == 0, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=1)

        # Back to original dtype before multiplying with LSTM output
        weights = weights.to(sequence_output.dtype).unsqueeze(-1)
        return torch.sum(sequence_output * weights, dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        x = self.marbert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state
        x, _ = self.lstm1(x); x = self.dropout1(x)
        x, _ = self.lstm2(x); x = self.dropout2(x)
        pooled = self.attention_pooling(x, attention_mask)
        return {"Emotion": self.emotion_head(pooled),
                "Offensive": self.offensive_head(pooled),
                "Hate": self.hate_head(pooled)}

# 10. CLASS WEIGHTS
def get_class_weights(dataframe, label, num_classes):
    y = dataframe[label + "_label"].dropna().astype(int)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(weights, dtype=torch.float).to(device)

class_weights = {label: get_class_weights(train_df, label, num_classes[label]) for label in LABELS}
for label in LABELS:
    print(f"\n{label} class weights:"); print(class_weights[label])

# 11. LOSS FUNCTION
criterions = {label: nn.CrossEntropyLoss(weight=class_weights[label]) for label in LABELS}

def calculate_loss(logits, batch):
    losses = []
    for label in LABELS:
        labels = batch[label]
        valid_mask = labels != -1                                    # Ignore missing labels
        if valid_mask.sum() == 0: continue
        loss = criterions[label](logits[label][valid_mask], labels[valid_mask])
        losses.append(LOSS_WEIGHTS[label] * loss)                    # Weight Emotion more
    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()

# 12. VALIDATION
@torch.no_grad()
def validate_model(model, dataloader):
    model.eval()
    predictions = {l: [] for l in LABELS}
    targets = {l: [] for l in LABELS}
    total_loss, batches = 0.0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
        labels = {label: batch[label].to(device) for label in LABELS}

        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        loss = calculate_loss(logits, labels)
        total_loss += loss.item(); batches += 1

        for label in LABELS:
            valid_mask = labels[label] != -1
            if valid_mask.sum() == 0: continue
            pred = torch.argmax(logits[label], dim=1)
            predictions[label].extend(pred[valid_mask].detach().cpu().numpy())
            targets[label].extend(labels[label][valid_mask].detach().cpu().numpy())

    results = {}
    for label in LABELS:
        if len(targets[label]) == 0: continue
        results[label] = {
            "accuracy": accuracy_score(targets[label], predictions[label]),
            "macro_f1": f1_score(targets[label], predictions[label], average="macro", zero_division=0)}
    results["loss"] = total_loss / max(batches, 1)
    results["average_macro_f1"] = np.mean(
        [results[label]["macro_f1"] for label in LABELS if label in results])
    return results

# 13. TRAINING
def train_model(model, train_loader, val_loader, epochs=8):
    # Different LRs: small for MARBERT, larger for LSTM + heads
    marbert_params = list(model.marbert.parameters())
    other_params = (list(model.lstm1.parameters()) + list(model.lstm2.parameters()) +
                    list(model.attention.parameters()) + list(model.emotion_head.parameters()) +
                    list(model.offensive_head.parameters()) + list(model.hate_head.parameters()))

    optimizer = torch.optim.AdamW(
        [{"params": marbert_params, "lr": MARBERT_LR},
         {"params": other_params, "lr": LSTM_LR}], weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1, best_state = -1, None

    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
            labels = {label: batch[label].to(device) for label in LABELS}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
                loss = calculate_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        val_results = validate_model(model, val_loader)
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        for label in LABELS:
            if label in val_results:
                print(f"{label}: Acc={val_results[label]['accuracy']:.4f} "
                      f"Macro-F1={val_results[label]['macro_f1']:.4f}")
        print(f"Average Macro-F1: {val_results['average_macro_f1']:.4f}")

        if val_results["average_macro_f1"] > best_f1:
            best_f1 = val_results["average_macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("✓ Best model saved")

    model.load_state_dict(best_state)
    return model.to(device)

# 14. EVALUATION
@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()
    predictions = {l: [] for l in LABELS}
    targets = {l: [] for l in LABELS}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None: token_type_ids = token_type_ids.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        for label in LABELS:
            labels = batch[label].to(device)
            valid_mask = labels != -1
            if valid_mask.sum() == 0: continue
            pred = torch.argmax(logits[label], dim=1)
            predictions[label].extend(pred[valid_mask].cpu().numpy())
            targets[label].extend(labels[valid_mask].cpu().numpy())

    for label in LABELS:
        print("\n" + "=" * 70)
        print(f"{label} CLASSIFICATION")
        print("=" * 70)
        print(classification_report(targets[label], predictions[label], digits=4, zero_division=0))
        print(f"{label} Accuracy: {accuracy_score(targets[label], predictions[label]):.4f}")
        print(f"{label} Macro-F1: {f1_score(targets[label], predictions[label], average='macro', zero_division=0):.4f}")

# 15. DATASETS
train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LENGTH)
test_dataset = MultiTaskDataset(test_df, tokenizer, MAX_LENGTH)

# 16. DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=(device.type == "cuda"), num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=(device.type == "cuda"), num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         pin_memory=(device.type == "cuda"), num_workers=2)

# 17. CREATE MODEL
model = MARBERT_BiLSTM_MultiHead(num_classes=num_classes,
                                 lstm_hidden_size=LSTM_HIDDEN_SIZE,
                                 dropout=DROPOUT).to(device)
print(model)

# 18. TRAIN
model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

# 19. FINAL TEST EVALUATION
evaluate_model(model, test_loader)