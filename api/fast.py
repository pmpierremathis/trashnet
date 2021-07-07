from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from trashnet.predict import read_image, load_model, CLASSES
from trashnet.gcp import storage_upload
import numpy as np
import shutil
from datetime import datetime


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

    
    # Make a prediction
    image = read_image(await file.read())
    prediction_ohe = model.predict(image)
    probability = float(max(prediction_ohe[0]))
    prediction = np.argmax(prediction_ohe, axis=1)
    api_answer = CLASSES[prediction[0]]
    
    #  # Download locally the file
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # filename = f"{api_answer} {timestamp}.jpg"
    # with open(filename, "wb") as buffer:
    #         shutil.copyfileobj(file.file, buffer)
            
    # # Upload the file to GCS      
    # storage_upload(filename, api_answer)
    
    return {"prediction" : api_answer,
            "probability" : probability}