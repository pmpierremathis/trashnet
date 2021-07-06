from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from trashnet.predict import read_image, load_model, CLASSES
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello")

model = load_model()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    image = read_image(await file.read())
    prediction_ohe = model.predict(image)
    prediction = np.argmax(prediction_ohe, axis=1)
    return {"prediction" : CLASSES[prediction[0]],
            "probability" : float(max(prediction_ohe[0]))}