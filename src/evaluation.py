import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_train, y_train,
                   X_test, y_test):

    for name, X, y in [
        ("Training", X_train, y_train),
        ("Testing", X_test, y_test)
    ]:

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        print(f"\n{name} Results")

        print("Accuracy:",
              accuracy_score(y, predictions))

        print("Precision:",
              precision_score(
                  y, predictions, zero_division=0
              ))

        print("Recall:",
              recall_score(
                  y, predictions, zero_division=0
              ))

        print("F1 Score:",
              f1_score(
                  y, predictions, zero_division=0
              ))
        
        print("\nClassification Report:")
        print(classification_report(
            y, predictions,
            zero_division=0
        ))

        if name == "Testing":

            cm = confusion_matrix(y, predictions)

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )

            display.plot()

            plt.title("Test Confusion Matrix")

            os.makedirs("results/plots", exist_ok=True)
            plt.savefig(
                "results/plots/confusion_matrix.png"
            )

            plt.close()
