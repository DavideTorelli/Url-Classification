import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data_prep

# === CONFIGURAZIONE ===
model_ckpt = "bert-base-uncased"
batch_size = 128
epochs = 10
learning_rate = 1e-4
data_path = "data/final_data.csv"

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODELLO E TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

# === FUNZIONI ===

def predict(url, tokenizer, model):
    model.eval()
    inputs = data_prep.preprocess([url], tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(torch.softmax(logits, dim=1)).item()

def train_model(train_dataset, model):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # Importante: escludere mlm_labels
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "mlm_labels"]}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")

        # Salva modello e tokenizer ad ogni epoca
        save_path = f"models/BERT-URL-CLS-{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Modello salvato in {save_path}")


def eval_model(eval_dataset, model):
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "mlm_labels"]}
            labels = batch["label"].to(device)

            logits = model(**inputs).logits
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Evaluation Results - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


# === MAIN ===
if __name__ == "__main__":
    print("Caricamento dataset...")
    dataset = data_prep.URLTranDataset(data_path, tokenizer)
    print("Inizio training...")
    train_model(dataset, model)
