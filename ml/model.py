import os
from .data import process_data
import pandas as pd
import pickle
import json
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

load_dotenv()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if not model:
        model_artifact = pickle.load(open(os.environ["ARTIFACT_PATH"], "rb"))
        model = model_artifact["model"]

        X_dict = {
            "age": int(X.age),
            "workclass": X.workclass,
            "fnlgt": int(X.fnlgt),
            "education": X.education,
            "education-num": int(X.education_num),
            "marital-status": X.marital_status,
            "occupation": X.occupation,
            "relationship": X.relationship,
            "race": X.race,
            "sex": X.sex,
            "capital-gain": int(X.capital_gain),
            "capital-loss": int(X.capital_loss),
            "hours-per-week": int(X.hours_per_week),
            "native-country": X.native_country,
        }
        X_df = pd.DataFrame(X_dict, index=[0])
        cat_features = os.environ["CATEGORICAL_FEATURES"].split(",")
        X_test, _, _, _ = process_data(
            X_df,
            cat_features,
            None,
            False,
            model_artifact["encoder"],
            model_artifact["lb"],
        )
    else:
        X_test = X
    preds = model.predict(X_test)

    return preds


def save_model_artifacts(artifact_path, **kwargs):

    pickle.dump(kwargs, open(artifact_path, "wb"))


def computer_metrics_test_set(X, ytrue, model):

    ypreds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(ytrue, ypreds)
    test_metrics = {
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta
    }
    print("---"*20)
    print("Test set overall metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    print("---"*20)
    with open("data/test_metrics.txt", "w") as f:
        json.dump(test_metrics, f)

        
def computer_metrics_slices(data, categorical_features, label, model_artifact):

    model_object = dict(**model_artifact)

    model = model_object.pop("model")

    slice_metrics = dict()
    print("Computer metrics for each slice:")
    for column in categorical_features:
        print("---"*20)
        print(f"--------COLUMN NAME: {column}")
        this_slice_metric = dict()
        for unique_value in data[column].unique():
            slice_data = data[data[column] == unique_value]

            X_slice, y_slice, _, _ = process_data(
                slice_data, categorical_features, label, training=False, **model_object
            )
            preds_slice = inference(model, X_slice)
            metrics_current = compute_model_metrics(
                y_slice, preds_slice
            )
            this_slice_metric[unique_value] = metrics_current
            print(f"Unique Value: {unique_value}")
            print(metrics_current)

        slice_metrics[column] = this_slice_metric
    print("---"*20)

    with open("data/slice_metrics.txt", "w") as f:
        json.dump(slice_metrics, f)
