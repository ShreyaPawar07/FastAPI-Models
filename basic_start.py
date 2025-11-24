from fastapi import FastAPI , Path
import json

app= FastAPI()

@app.get("/")
def hello():
    return "Hello Shreya"

def load_json(file):
    with open(file,'rb') as f:
        data = json.load(f)
    return data

@app.get('/view')
def view():
    data = load_json("patients.json")
    return data

@app.get('/patient/{patient_id}')  # pyright: ignore[reportUndefinedVariable]
def patient(patient_id:str=Path(...,description='Id of a patient',example='P001')):
    data =load_json('patients.json')
    
    if patient_id in data:
        return data[patient_id]
    else:
        return "No data found"
