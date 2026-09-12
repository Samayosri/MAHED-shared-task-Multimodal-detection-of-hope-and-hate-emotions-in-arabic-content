import re, unicodedata, random
import numpy as np, pandas as pd, torch, nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

nltk.download('stopwords', quiet=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Load data + preprocessing
df = pd.read_csv("train.csv")
print(df.shape); df.head()

ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02-twitter"
_arabert_preprocessor = ArabertPreprocessor(model_name=ARABERT_MODEL_NAME)

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)          # Remove URLs
    text = re.sub(r'@\S+', ' ', text)                            # Remove mentions
    text = re.sub(r'(.)\1{2,}', r'\1', text)                     # Reduce repeated chars
    text = "".join(c for c in text if unicodedata.category(c) != "So")  # Remove symbols/emojis
    return _arabert_preprocessor.preprocess(text)                # AraBERT preprocessing

df["clean_text"] = df["text"].apply(preprocess_text)
df[["text", "clean_text"]].head()

# 2. MARBERT embedding
MARBERT_MODEL_NAME = "UBC-NLP/MARBERT"
tokenizer = AutoTokenizer.from_pretrained(MARBERT_MODEL_NAME)
model = AutoModel.from_pretrained(MARBERT_MODEL_NAME).to(device)
model.eval()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

def embed_texts(texts, batch_size=32, max_length=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

X = embed_texts(df["clean_text"].tolist())
print("Embedding shape:", X.shape)

# 3. Classification
def evaluate_label(X_train, X_test, y_train, y_test, label_name, plot=False):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"[MARBERT] {label_name} -> acc={acc:.4f}  macro-F1={macro_f1:.4f}")
    if plot:
        print(classification_report(y_test, preds))
    return clf, acc, macro_f1

# 4. Evaluate all labels
LABEL_COLS = ["Emotion", "Offensive", "Hate"]
results = []

for label in LABEL_COLS:
    y = df[label]
    mask = y.notna()
    X_use, y_use = X[mask.values], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y_use, test_size=0.2, random_state=SEED, stratify=y_use)
    _, acc, macro_f1 = evaluate_label(X_train, X_test, y_train, y_test, label, plot=True)
    results.append({"model": "MARBERT", "label": label, "accuracy": acc, "macro_f1": macro_f1})

# 5. Results
results_df = pd.DataFrame(results)
print("\nResults:"); print(results_df)