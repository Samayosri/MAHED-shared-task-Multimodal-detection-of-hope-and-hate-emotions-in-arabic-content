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
MARBERT_MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LENGTH, BATCH_SIZE, EPOCHS = 128, 16, 8
MARBERT_LR, HEAD_LR = 1e-5, 2e-4
WEIGHT_DECAY, DROPOUT, GRADIENT_CLIP = 0.01, 0.30, 1.0
LOSS_WEIGHTS = {"Emotion": 1.5, "Offensive": 1.0, "Hate": 1.0}   # Emotion gets more weight

# 3. LOAD DATA
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print("Train Data Head:"); print(train_df.head()); print(train_df.columns)
print("Validation Data Head:"); print(val_df.head()); print(val_df.columns)
print("Test Data Head:"); print(test_df.head()); print(test_df.columns)

# 4. PREPROCESSING
arabert_preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02-twitter")

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)              # Remove URLs
    text = re.sub(r'@\S+', '', text)                                # Remove mentions
    text = re.sub(r'(.)\1{2,}', r'\1', text)                        # Reduce repeated chars
    # text = ''.join(ch for ch in text                               # Remove emoji/symbols
    #                if not (0x1F300 <= ord(ch) <= 0x1FAFF))
    return arabert_preprocessor.preprocess(text)                    # AraBERT preprocessing

train_df["text"] = train_df["text"].fillna("").apply(preprocess_text)
val_df["text"] = val_df["text"].fillna("").apply(preprocess_text)
test_df["text"] = test_df["text"].fillna("").apply(preprocess_text)

print("\nPreprocessed Train Data Head:"); print(train_df[["text", "Emotion", "Offensive", "Hate"]].head())
print("\nPreprocessed Validation Data Head:"); print(val_df[["text", "Emotion", "Offensive", "Hate"]].head())
print("\nPreprocessed Test Data Head:"); print(test_df[["text", "Emotion", "Offensive", "Hate"]].head())

# 5. LABEL ENCODING
LABELS = ["Emotion", "Offensive", "Hate"]
label_maps, num_classes = {}, {}

# Process train_df to establish label_maps and num_classes
for label in LABELS:
    train_df[label] = train_df[label].astype(str)
    unique_labels = sorted(train_df[label].unique())
    label_maps[label] = {name: idx for idx, name in enumerate(unique_labels)}
    train_df[label + "_label"] = train_df[label].map(label_maps[label])
    num_classes[label] = len(unique_labels)
    print(f"{label}: {num_classes[label]} classes"); print(label_maps[label])

# Apply label mapping to val_df and test_df using the maps learned from train_df
for label in LABELS:
    val_df[label] = val_df[label].astype(str)
    test_df[label] = test_df[label].astype(str)
    val_df[label + "_label"] = val_df[label].map(label_maps[label])
    test_df[label + "_label"] = test_df[label].map(label_maps[label])

print("\nLabel Encoded Train Data Head:"); print(train_df[["text", "Emotion", "Emotion_label", "Offensive", "Hate"]].head())
print("\nLabel Encoded Validation Data Head:"); print(val_df[["text", "Emotion", "Emotion_label", "Offensive", "Hate"]].head())
print("\nLabel Encoded Test Data Head:"); print(test_df[["text", "Emotion", "Emotion_label", "Offensive", "Hate"]].head())

# 6. TRAIN / VALIDATION / TEST SPLIT (Removed 'LOAD DATA' as it's handled above)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

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
class TaskAttentionPool(nn.Module):
    """Per-task attention pooling over MARBERT token states (FP16-safe).

    Each task (Emotion / Offensive / Hate) gets its OWN instance of this module,
    so each learns its own notion of which tokens matter.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, attention_mask):
        # sequence_output: (B, T, H)
        scores = self.query(sequence_output).squeeze(-1)

        # Masking + softmax in FP32 to avoid FP16 overflow/underflow
        scores = scores.float().masked_fill(
            attention_mask == 0, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=1)

        # Back to original dtype before multiplying with MARBERT output
        weights = weights.to(sequence_output.dtype).unsqueeze(-1)
        return torch.sum(sequence_output * weights, dim=1)


class MARBERT_MultiHead(nn.Module):
    """
    Architecture:

        MARBERT (shared) → token representations
                                |
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
          Emotion           Offensive            Hate
         Attention          Attention          Attention
              │                 │                 │
         ┌────┼────┐       ┌────┼────┐       ┌────┼────┐
         │    │    │       │    │    │       │    │    │
        Mean Max Attn     Mean Max Attn     Mean Max Attn
         │    │    │       │    │    │       │    │    │
         └────┼────┘       └────┼────┘       └────┼────┘
              ▼                 ▼                 ▼
           Fusion            Fusion            Fusion
              ▼                 ▼                 ▼
          Emotion           Offensive            Hate
           Head               Head               Head
              ▼                 ▼                 ▼
         12 classes          Yes/No         Hate classes

    Mean & Max pooling are shared (computed once on the token reps).
    Attention, Fusion, and Head are per-task.
    """
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.marbert = AutoModel.from_pretrained(MARBERT_MODEL_NAME)
        h = self.marbert.config.hidden_size   # 768 for MARBERT

        # ---- Per-task attention pools (one per task) ----
        self.emotion_attention   = TaskAttentionPool(h)
        self.offensive_attention = TaskAttentionPool(h)
        self.hate_attention      = TaskAttentionPool(h)

        # ---- Per-task fusion layers: concat(mean, max, attn) → h ----
        # 768 + 768 + 768 = 2304 → Linear(2304 → 768) → GELU → Dropout
        def make_fusion():
            return nn.Sequential(nn.Linear(h * 3, h),
                                 nn.GELU(),
                                 nn.Dropout(dropout))
        self.emotion_fusion   = make_fusion()
        self.offensive_fusion = make_fusion()
        self.hate_fusion      = make_fusion()

        # ---- Per-task classifier heads ----
        def make_head(n):
            return nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(h, n))
        self.emotion_head   = make_head(num_classes["Emotion"])
        self.offensive_head = make_head(num_classes["Offensive"])
        self.hate_head      = make_head(num_classes["Hate"])

    # ---- Shared pooling helpers (computed once, reused by all tasks) ----
    @staticmethod
    def mean_pool(hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    @staticmethod
    def max_pool(hidden, mask):
        m = mask.unsqueeze(-1).float()
        neg_inf = torch.finfo(hidden.dtype).min
        hidden = hidden.masked_fill(m == 0, neg_inf)
        return hidden.max(dim=1).values

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. Shared MARBERT backbone
        x = self.marbert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state

        # 2. Shared mean & max pools (computed once)
        mean_p = self.mean_pool(x, attention_mask)
        max_p  = self.max_pool(x, attention_mask)

        # 3. Per-task branch: attention → concat → fusion → head
        # --- Emotion ---
        attn_e = self.emotion_attention(x, attention_mask)
        fused_e = self.emotion_fusion(torch.cat([mean_p, max_p, attn_e], dim=-1))

        # --- Offensive ---
        attn_o = self.offensive_attention(x, attention_mask)
        fused_o = self.offensive_fusion(torch.cat([mean_p, max_p, attn_o], dim=-1))

        # --- Hate ---
        attn_h = self.hate_attention(x, attention_mask)
        fused_h = self.hate_fusion(torch.cat([mean_p, max_p, attn_h], dim=-1))

        return {"Emotion":   self.emotion_head(fused_e),
                "Offensive": self.offensive_head(fused_o),
                "Hate":      self.hate_head(fused_h)}

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
    # Different LRs: small for MARBERT, larger for task modules
    marbert_params = list(model.marbert.parameters())
    other_params = (list(model.emotion_attention.parameters()) +
                    list(model.offensive_attention.parameters()) +
                    list(model.hate_attention.parameters()) +
                    list(model.emotion_fusion.parameters()) +
                    list(model.offensive_fusion.parameters()) +
                    list(model.hate_fusion.parameters()) +
                    list(model.emotion_head.parameters()) +
                    list(model.offensive_head.parameters()) +
                    list(model.hate_head.parameters()))

    optimizer = torch.optim.AdamW(
        [{"params": marbert_params, "lr": MARBERT_LR},
         {"params": other_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)

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

# 14. EVALUATION (per-task)
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

    # Return per-task metrics so the final average block can use them
    results = {}
    for label in LABELS:
        if len(targets[label]) == 0: continue
        results[label] = {
            "accuracy": accuracy_score(targets[label], predictions[label]),
            "macro_f1": f1_score(targets[label], predictions[label],
                                 average="macro", zero_division=0),
            "weighted_f1": f1_score(targets[label], predictions[label],
                                    average="weighted", zero_division=0)}
    return results

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
model = MARBERT_MultiHead(num_classes=num_classes, dropout=DROPOUT).to(device)
print(model)

# 18. TRAIN
model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

# 19. FINAL TEST EVALUATION
results = evaluate_model(model, test_loader)

# 20. AVERAGE EVALUATION (across all 3 tasks)
results_df = pd.DataFrame(results).T.reset_index()
results_df = results_df.rename(columns={"index": "Task"})

print("\n" + "=" * 65)
print("FINAL TEST RESULTS (MARBERT Multi-Head — Per-Task Attention)")
print("=" * 65)
print(results_df.to_string(index=False))
print(f"\nMean Accuracy    : {results_df['accuracy'].mean():.4f}")
print(f"Mean Macro-F1    : {results_df['macro_f1'].mean():.4f}")
print(f"Mean Weighted-F1 : {results_df['weighted_f1'].mean():.4f}")
