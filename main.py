from fastapi import FastAPI
from pydantic import BaseModel, Field
from .ml import model


class PersonInfo(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(None, alias="education-num")
    marital_status: str = Field(None, alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(None, alias="capital-gain")
    capital_loss: int = Field(None, alias="capital-loss")
    hours_per_week: int = Field(None, alias="hours-per-week")
    native_country: str = Field(None, alias="native-country")


app = FastAPI()


@app.get("/")
async def welcome():
    return "Welcome to this FastAPI deployment of a Salary Predictor"


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/predict_salary/")
async def run_inference(person_info: PersonInfo):
    prediction = model.inference(None, person_info)
    salary = ">50k" if prediction[0] == 1 else "<50k"
    return f"Model predicts Salary should be {salary}"
