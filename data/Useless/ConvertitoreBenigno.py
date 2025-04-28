import csv

# Path al tuo file CSV di input (top-1m.csv)
input_csv = 'top-1m.csv'  # Cambia questo con il percorso corretto del tuo file

# Path per il nuovo file CSV con url,label
output_csv = 'benign_urls_with_labels.csv'  # Cambia questo con il percorso del file di output

# Leggere gli URL dal file esistente e creare un nuovo file con l'etichetta "0" (benigni)
with open(input_csv, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    # Apri il file di output per scrivere i nuovi dati
    with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Scrivere l'intestazione (opzionale)
        writer.writerow(['url', 'label'])
        
        # Itera su ogni riga del file di input
        for row in reader:
            if row:  # Se la riga non è vuota
                url = row[1]  # Assumendo che l'URL sia nella seconda colonna (indice 1)
                writer.writerow([url, 0])  # Aggiungi l'URL con l'etichetta "0" (benigno)
                
print("File CSV con URL e label '0' creato con successo!")
