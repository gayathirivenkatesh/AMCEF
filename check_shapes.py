import os
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = r"c:\Users\M.DHAARANI\Downloads\AMCEF\assets\models"

try:
    m1 = load_model(os.path.join(MODELS_DIR, "emodb_model.keras"))
    print(f"EMODB Output Shape: {m1.output_shape}")
    
    m2 = load_model(os.path.join(MODELS_DIR, "emovo_model.keras"))
    print(f"EMOVO Output Shape: {m2.output_shape}")
    
    m3 = load_model(os.path.join(MODELS_DIR, "shemo_model.keras"))
    print(f"SHEMO Output Shape: {m3.output_shape}")
    
except Exception as e:
    print(f"Error: {e}")
