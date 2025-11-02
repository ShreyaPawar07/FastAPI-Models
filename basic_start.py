from fastapi import FastAPI
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