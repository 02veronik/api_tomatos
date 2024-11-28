


import datetime 
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json
import firebase_admin
from firebase_admin import credentials, firestore
import requests
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import HTTPException

app = FastAPI()

# Cargar modelos y clases al inicio
base_model = load_model("C:/Users/ALEXANDER/Documents/X_Ciclo/TESIS/PROYECTO/api/base_model.h5")
svm_model = joblib.load("C:/Users/ALEXANDER/Documents/X_Ciclo/TESIS/PROYECTO/api/svm_model.pkl")

with open("C:/Users/ALEXANDER/Documents/X_Ciclo/TESIS/PROYECTO/api/ID_names.json", "r") as f:
    id_names = json.load(f)

class_name = list(id_names.keys())
print (class_name)
print ("id", id_names)

# Inicializar Firebase
cred = credentials.Certificate("C:/Users/ALEXANDER/Documents/X_Ciclo/TESIS/PROYECTO/api/tomatoshield-e6392-firebase-adminsdk-x8ziu-a2ee73175e.json")
firebase_admin.initialize_app(cred) 
db = firestore.client()

def download_image_from_url(url, img_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción si el estado de la respuesta no es 200
        with open(img_path, "wb") as img_file:
            img_file.write(response.content)
    except requests.exceptions.RequestException as e:
        # Captura cualquier error relacionado con la solicitud HTTP
        raise HTTPException(400, detail=f"Error al descargar la imagen: {str(e)}")



# Función de predicción modificada
def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    
        # Extraer características con MobileNetV2
        features = base_model.predict(img_array)
        features_flat = features.reshape((features.shape[0], -1))

        # Clasificar usando SVM
        prediction = svm_model.predict(features_flat)
        predicted_class = class_name[prediction[0]]
        
        # Obtener la ID desde ID_names
        if predicted_class in id_names:
            return id_names[predicted_class]  # Devuelve directamente la ID asociada
        else:
            raise ValueError(f"No se encontró información para la plaga: {predicted_class}")
     
    except Exception as e:
        raise HTTPException(500, detail=f"Error en la predicción de la imagen: {str(e)}")


class ImageRequest(BaseModel):
    image_url: str
    
 

# Endpoint para recibir imagen y devolver predicción
@app.post("/predict/")
async def predict(image_request: ImageRequest):
    try:
        image_url = image_request.image_url
        # Definir el nombre de archivo temporal para la imagen
        img_path = "temp_image.jpg"
        
        # Descargar la imagen desde Firebase Storage
        download_image_from_url(image_url, img_path)

        # Realizar predicción
        plaga_detectada = predict_image(img_path)

        # Retornar solo el nombre de la plaga detectada
        return {"plaga_detectada": plaga_detectada}

    except HTTPException as e:
        # Maneja excepciones HTTP relacionadas con la descarga o la predicción
        raise e
    except Exception as e:
        # Captura cualquier otro error inesperado
        raise HTTPException(500, detail=f"Error inesperado: {str(e)}")