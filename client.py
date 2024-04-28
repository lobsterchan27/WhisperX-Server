import requests
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def text2speech():
    url = "http://localhost:8127/api/text2speech"

    payload = {
        "prompt": "machi is so cool",
        "voice": "reference"
    }

    response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    with open('network.wav', 'wb') as f:
        f.write(response.content)

def kobold():
    url = "http://localhost:8080/api/extra/generate/stream"

    payload = {
    "prompt": 
    '''
    Ok what next?
    So I have been a bit quiet on the blog \'whatever bro\'''',
    "temperature": 0.5,
    "top_p": 0.9,
    "max_length": 2
    }
    print(payload["prompt"])
    response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    print (response.text)

if __name__ == "__main__":
    text2speech()
    # kobold()