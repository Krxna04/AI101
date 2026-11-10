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

   def plot_class_distribution(df):

    sns.countplot(data=df, x="Class")

    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Count")

    plt.savefig(
        "results/plots/class_distribution.png"
    )

    plt.close()


def plot_correlation(df):

    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(15, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")

    plt.savefig(
        "results/plots/correlation_heatmap.png"
    )

    plt.close()


