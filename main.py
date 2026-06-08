from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # <-- Added missing import
import tensorflow as tf
from keras import layers, models
import numpy as np
from PIL import Image
import io

app = FastAPI(title="CropCare AI")
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- BULLETPROOF BLUEPRINT METHOD ---
def build_and_load_model():
    print("Building empty MobileNetV3 blueprint...")
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None # Load structurally empty
    )
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(51)(x) 
    
    clean_model = models.Model(inputs, outputs)
    
    # Extract and inject ONLY the numerical weights, ignoring broken JSON configs
    MODEL_PATH = "model/CropCare_MobileNetV3_FineTuned.keras"
    print("Injecting mathematical weights...")
    clean_model.load_weights(MODEL_PATH)
    
    return clean_model

model = build_and_load_model()
print("AI Model loaded successfully and ready for predictions!")

CLASS_NAMES = [
    "American_Bollworm_on_Cotton", "Anthracnose_on_Cotton", 
    "Corn_maize___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_maize___Common_rust", 
    "Corn_maize___Northern_Leaf_Blight", "Corn_maize___healthy", "Cotton_Aphid", 
    "Cotton_Bacterial_Blight", "Cotton_Healthy", "Grape___Black_rot", 
    "Grape___Esca_Black_Measles", "Grape___Leaf_blight_Isariopsis_Leaf_Spot", 
    "Grape___healthy", "Mosaic_sugarcane", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "RedRot_sugarcane", "RedRust_sugarcane", 
    "Rice_Bacterial_Blight", "Rice_Blast", "Sugarcane_Healthy", "Tomato___Bacterial_spot", 
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy", "Wheat_Brown_Leaf_Rust", 
    "Wheat_Healthy", "Wheat_Stem_fly", "Wheat_Yellow_Rust", "Wheat_aphid", 
    "Wheat_black_rust", "Wheat_leaf_blight", "Wheat_mite", "Wheat_powdery_mildew", 
    "Wheat_scab", "Yellow_Rust_Sugarcane", "bollworm_on_Cotton", "cotton_mealy_bug", 
    "cotton_whitefly", "pink_bollworm_in_cotton", "red_cotton_bug", "thirps_on__cotton"
]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(tf.nn.softmax(predictions[0])[predicted_class_index])
        
        raw_name = CLASS_NAMES[predicted_class_index]
        clean_name = raw_name.replace("___", ": ").replace("_", " ")

        return {
            "success": True,
            "disease": clean_name,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}