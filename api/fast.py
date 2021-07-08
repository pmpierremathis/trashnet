from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from trashnet.predict import read_image, load_model, CLASSES
from trashnet.gcp import storage_upload
import numpy as np
import io
from PIL import Image
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
    image_bytes = await file.read()
    image = read_image(image_bytes)
    
    prediction_ohe = model.predict(image)
    probability = float(max(prediction_ohe[0]))
    prediction = np.argmax(prediction_ohe, axis=1)
    api_answer = CLASSES[prediction[0]]
    
    return {"prediction" : api_answer,
            "probability" : probability}
    
@app.post("/labelling")
async def upload_label(checked_label:str = Form(...), file: UploadFile = File(...)):
    # Download locally the file
    print(type(checked_label))
    print(type(file))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"{checked_label} {timestamp}.jpg"
    
    image_bytes = await file.read()
    imageStream = io.BytesIO(image_bytes)
    imageFile = Image.open(imageStream)
    imageFile.save(filename)

    # # Upload the file to GCS      
    storage_upload(filename, checked_label)
    pass