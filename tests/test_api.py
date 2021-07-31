from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_welcome_path():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to this FastAPI deployment of a Salary Predictor"


def test_prediction_below_50k(mocker):
    response = client.post(
        "/predict_salary/",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    # assert response.status_code == 200
    print(response.json())
    assert response.json() == "Model predicts Salary should be <50k"


def test_prediction_above50k():
    response = client.post(
        "/predict_salary/",
        json={
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    # assert response.status_code == 200
    print(response.json())
    assert response.json() == "Model predicts Salary should be >50k"
