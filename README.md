URL Classification Project

Questo progetto si ispira a URLTran:
https://github.com/bfilar/URLTran

Il paper originale di URLTran è disponibile qui:
https://arxiv.org/pdf/2106.05256


Questo progetto utilizza modelli di transformers (come BERT) per:
- Classificare URL come benigni o maligni
- (Opzionale) Pre-allenare un modello su URL tramite Masked Language Modeling (MLM)


File principali 
- classifier.py
- data_prep.py
- mlm.py
- predict.py
- data/final_data.csv


1. classifier.py
Cosa fa: Allena un modello di classificazione BERT sui dati (URL + label).
Obiettivo: Prevede se un URL è benigno (0) o maligno (1).
Funzionalità principali:
  1 Carica il dataset.
  2 Preprocessa e tokenizza gli URL.
  3 Allena un modello BertForSequenceClassification.
  4 Salva il modello dopo ogni epoca.

2. data_prep.py
Cosa fa: Gestisce la preparazione dei dati e dei dataset.
Obiettivo: Fornire input tokenizzati compatibili con i modelli Hugging Face.
Funzionalità principali:
  1 Carica il file CSV contenente URL e label.
  2 Tokenizza gli URL.
  3 Crea dataset PyTorch (Dataset).
  4 Prepara anche i dati per il masked language modeling (mlm_labels) se necessario.
  5 Contiene una funzione masking_step per mascherare casualmente token durante il training MLM.


3. mlm.py
Cosa fa: Allena un modello BERT su Masked Language Modeling (MLM) usando URL.
Obiettivo: Migliorare il modello facendogli "completare" parti mancanti degli URL.
Funzionalità principali:
  1 Maschera casualmente parte degli URL ([MASK] token).
  2 Allena il modello BertForMaskedLM su questo compito.
  3 Salva i modelli pre-addestrati.

4. predict.py
Cosa fa: Usa il modello allenato per classificare URL nuovi.
Obiettivo: Predire se un URL fornito dall'utente è benigno o maligno.
Funzionalità principali:
  1 Carica il modello salvato dopo il training.
  2 Chiede all'utente un URL.
  3 Fornisce la predizione (Benign o Malicious) con un livello di confidenza.


5. data/final_data.csv
Cosa contiene: Il dataset usato per allenare i modelli.
Formato:
url	label
https://esempio.com	0 (Maligno)
http://phishing.com	1 (Benigno)


Flusso Tipico del Progetto
1- Avvia classifier.py per addestrare il modello di classificazione.
2- (Opzionale) Avvia mlm.py per pre-addestrare il modello su URL usando Masked Language Modeling.
3- Usa predict.py per testare nuovi URL usando il modello allenato.

Requisiti Tecnici
Python 3.8+
Librerie:
- torch
- transformers
- scikit-learn
- pandas

Installa tutto con:
pip install torch transformers scikit-learn pandas
oppure
pip install -r requirements.txt


NOTE IMPORTANTI: 
- Per creare modelli su dataset molto grossi (> 500.000 URL), il training può richiedere diverse ore
- Il modello pre-addestrato con mlm.py non è obbligatorio:
bert-base-uncased può essere utilizzato direttamente per la classificazione.
- Il modello mlm.py migliora leggermente la comprensione degli URL, ma non è stato ancora validato in modo sistematico.


Istruzioni per usare il modello pre-addestrato (MLM)
Se vuoi utilizzare il modello creato con mlm.py in classifier.py:

Sostituisci:
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

Con:
model_ckpt = "models/BERT-URL-MLM-2"  # <-- il tuo modello MLM salvato!
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

