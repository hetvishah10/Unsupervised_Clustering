import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def describe_data(df):
    return df.describe()

def show_data_head(df, n=5):
    return df.head(n)
