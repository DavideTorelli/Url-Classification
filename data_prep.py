import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class URLTranDataset(Dataset):
    def __init__(self, filepath, tokenizer, task="classification"):
        """
        Args:
            filepath (str): path al CSV file contenente 'url' e 'label'
            tokenizer (transformers tokenizer): tokenizer Huggingface
            task (str): 'classification' o 'mlm' (masked language modeling)
        """
        super().__init__()
        self.task = task
        self.df = pd.read_csv(filepath)
        self.df = self.df.sample(frac=1.0).reset_index(drop=True)  # shuffle

        self.url_data = self.df.url.values.tolist()
        self.labels = self.df.label.astype(int).tolist()

        self.encodings = preprocess(self.url_data, tokenizer)

    def __getitem__(self, idx):
        obs = {k: v[idx] for k, v in self.encodings.items()}

        if self.task == "classification":
            obs["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        elif self.task == "mlm":
            obs["labels"] = obs["mlm_labels"][idx]
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return obs

    def __len__(self):
        return len(self.encodings["input_ids"])


def preprocess(url_data, tokenizer):
    """
    Prepara l'input tokenizzato per il modello
    """
    inputs = tokenizer(
        url_data,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    inputs["mlm_labels"] = inputs["input_ids"].detach().clone()
    return inputs


def masking_step(input_ids):
    """
    Masking step per il masked language modeling.
    """
    rand = torch.rand(input_ids.shape)

    # Maschera: maschera solo token che non sono special token ([CLS]=101, [SEP]=102, [PAD]=0)
    mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)

    selection = [
        torch.flatten(mask_arr[i].nonzero()).tolist()
        for i in range(input_ids.shape[0])
    ]

    for i in range(input_ids.shape[0]):
        if selection[i]:  # assicurati che non sia vuoto
            input_ids[i, selection[i]] = 103  # 103 è il token [MASK]

    return input_ids


def split_data(dataset_path, test_size=0.33):
    """
    Divide il CSV in train/test e restituisce due DataFrame
    """
    df = pd.read_csv(dataset_path)
    X = df.url
    y = df.label.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    columns = ["url", "label"]
    train_df = pd.DataFrame(list(zip(X_train, y_train)), columns=columns)
    test_df = pd.DataFrame(list(zip(X_test, y_test)), columns=columns)

    return train_df, test_df
