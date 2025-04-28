import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer
import data_prep

# === CONFIGURAZIONE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ckpt = "bert-base-uncased"
batch_size = 32
epochs = 2
learning_rate = 2e-5
data_path = "data/final_data.csv"

# === CARICA MODELLO E TOKENIZER ===
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertForMaskedLM.from_pretrained(model_ckpt)

# === TRAINING FUNCTION ===
def train(dataset, model):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Masked input
            masked_inputs = data_prep.masking_step(batch["input_ids"]).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["mlm_labels"].to(device)

            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        # Salva modello ad ogni epoca
        save_path = f"models/BERT-URL-MLM-{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Modello salvato in {save_path}")

# === PREDIZIONE MASCHERATA ===
def predict_mask(url, tokenizer, model):
    model.eval()
    inputs = data_prep.preprocess([url], tokenizer)
    masked_inputs = data_prep.masking_step(inputs["input_ids"]).to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model(input_ids=masked_inputs, attention_mask=attention_mask)

    output_ids = torch.argmax(torch.nn.functional.softmax(predictions.logits[0], dim=-1), dim=1)

    return masked_inputs, output_ids

# === MAIN ===
if __name__ == "__main__":
    print("Caricamento dataset...")
    dataset = data_prep.URLTranDataset(data_path, tokenizer, task="mlm")
    print("Inizio training MLM...")
    train(dataset, model)

    # Esempio di completamento
    url = "huggingface.co/docs/transformers/task_summary"
    input_ids, output_ids = predict_mask(url, tokenizer, model)
    masked_input = tokenizer.decode(input_ids[0].tolist()).replace(" ", "")
    prediction = tokenizer.decode(output_ids.tolist()).replace(" ", "")

    print(f"Masked Input: {masked_input}")
    print(f"Predicted Output: {prediction}")
