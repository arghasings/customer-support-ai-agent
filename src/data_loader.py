import pandas as pd

def load_tickets(file_path="data/tickets.csv"):
    df = pd.read_csv(file_path)
    return df["text"].tolist()