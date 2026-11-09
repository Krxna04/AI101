import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def explore_dataset(df):

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())


def analyse_transactions(df):

    print("\nAmount Statistics:")
    print(df["Amount"].agg(
        ["mean", "median", "std"]
    ))

    print("\nTime Statistics:")
    print(df["Time"].agg(
        ["mean", "median", "std"]
    ))

    print("\nClass Distribution:")
    print(df["Class"].value_counts())


