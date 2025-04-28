import csv

# Path ai file di input
benign_file = 'benign_urls_with_labels.csv'  # Percorso del file benign_urls_with_labels.csv
phishing_file = 'phishing_data.csv'  # Percorso del file phishing_data.csv

# Path per il file di output unito
final_file = 'final_data.csv'  # Percorso del file final_data.csv

# Leggi i dati dal primo file (benign_urls_with_labels.csv)
with open(benign_file, mode='r', encoding='utf-8') as infile1:
    reader1 = csv.reader(infile1)
    benign_data = list(reader1)  # Converto le righe del primo file in una lista

# Leggi i dati dal secondo file (phishing_data.csv)
with open(phishing_file, mode='r', encoding='utf-8') as infile2:
    reader2 = csv.reader(infile2)
    phishing_data = list(reader2)  # Converto le righe del secondo file in una lista

# Unisci i dati dei due file
final_data = benign_data + phishing_data  # Concateno le righe dei due file

# Scrivi i dati uniti nel nuovo file (final_data.csv)
with open(final_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    
    # Scrivi l'intestazione (opzionale)
    writer.writerow(['url', 'label'])  # Se hai delle intestazioni specifiche, puoi modificarle
    
    # Scrivi tutte le righe nel file finale
    writer.writerows(final_data)

print(f"File unito creato con successo: {final_file}")
