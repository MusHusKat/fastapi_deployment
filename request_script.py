import requests

response = requests.post(
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
        }
    )

print(response.status_code)
print(response.json())