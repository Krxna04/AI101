from pathlib import Path
import os

from src.data_analysis import (
    load_dataset,
    explore_dataset,
    analyse_transactions,
    plot_class_distribution,
    plot_correlation
)

from src.preprocessing import prepare_data
from src.model_training import train_model
from src.evaluation import evaluate_model


ROOT = Path(__file__).resolve().parent.parent


def main():

    os.makedirs(ROOT / "results" / "plots", exist_ok=True)

    df = load_dataset(ROOT / "data" / "creditcard.csv")

    explore_dataset(df)
    analyse_transactions(df)

    plot_class_distribution(df)
    plot_correlation(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()
