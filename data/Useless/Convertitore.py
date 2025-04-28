import pandas as pd

def convert_lst_to_csv(lst_file, csv_file):
    # Leggi il file .lst con gli URL di phishing specificando la codifica utf-8
    with open(lst_file, 'r', encoding='utf-8') as file:
        urls = file.readlines()

    # Rimuovi eventuali spazi bianchi o caratteri di nuova linea
    urls = [url.strip() for url in urls]

    # Crea un DataFrame con gli URL e l'etichetta di phishing (1)
    labels = [1] * len(urls)  # Tutti gli URL sono etichettati come phishing (1)

    # Creazione del DataFrame
    df = pd.DataFrame({'url': urls, 'label': labels})

    # Salva il DataFrame come CSV
    df.to_csv(csv_file, index=False)

    print(f"File CSV '{csv_file}' creato con successo!")

# Utilizzo della funzione
lst_file = "C:/Users/tored/Desktop/Stage/URLTran/data/ALL-phishing-links.lst"  # Percorso del tuo file .lst
csv_file = "phishing_data.csv"  # Nome del file CSV di output
convert_lst_to_csv(lst_file, csv_file)
