import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import data_prep

# === CONFIG ===
model_path = "models/BERT-URL-CLS-1"  # <-- Modifica se hai salvato in un altro path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CARICA MODELLO E TOKENIZER ===
print(f"Caricamento modello da {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# === FUNZIONE DI PREDIZIONE ===
def predict_url(url):
    # Preprocess l'URL
    inputs = data_prep.preprocess([url], tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence

# === MAIN LOOP ===
if __name__ == "__main__":
    print("=== URL Classifier ===")
    while True:
        url = input("Inserisci un URL da classificare ('exit' per uscire): ").strip()
        if url.lower() == "exit":
            break
        pred_class, confidence = predict_url(url)
        label_name = "Malicious" if pred_class == 1 else "Benign"
        print(f"Predizione: {label_name} (Confidenza: {confidence:.2f})\n")
