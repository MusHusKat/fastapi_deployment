# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import computer_metrics_slices, computer_metrics_test_set, train_model, save_model_artifacts

from dotenv import load_dotenv

load_dotenv()


def run():

    # Add the necessary imports for the starter code.

    # Add code to load in the data.
    data = pd.read_csv("data/cleaned_census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    cat_features = os.environ["CATEGORICAL_FEATURES"].split(",")

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train and save a model.
    model = train_model(X_train, y_train)

    model_group = {"model": model, "encoder": encoder, "lb": lb}

    save_model_artifacts(artifact_path=os.environ["ARTIFACT_PATH"], **model_group)

    computer_metrics_slices(
        test,
        categorical_features=cat_features,
        label="salary",
        model_artifact=model_group,
    )

    computer_metrics_test_set(X_test, y_test, model)


if __name__ == "__main__":
    run()
