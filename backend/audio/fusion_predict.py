import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# Monkey-patch Dense to handle 'quantization_config' error in Keras 3.x
class RobustDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

custom_objects = {'Dense': RobustDense}

# Ensure local dir is in path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from feature_extraction import extract_features
except ImportError:
    from .feature_extraction import extract_features

# Base directory for models
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "assets", "models"))

# Standard alphabetical labels (most common for model training)
COMMON_EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

emodb_labels = ['angry','disgust','fear','happy','neutral','sad']
emovo_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
shemo_labels = ['angry','fear','happy','neutral','sad']

# Global model cache to avoid reloading models on every request
models_cache = {}

def get_models():
    if not models_cache:
        try:
            models_cache['emodb'] = load_model(os.path.join(MODELS_DIR, "emodb_model.keras"), custom_objects=custom_objects)
            models_cache['emovo'] = load_model(os.path.join(MODELS_DIR, "emovo_model.keras"), custom_objects=custom_objects)
            models_cache['shemo'] = load_model(os.path.join(MODELS_DIR, "shemo_model.keras"), custom_objects=custom_objects)
            print("Models loaded successfully from assets/models/")
        except Exception as e:
            print(f"CRITICAL: Failed to load models: {e}")
            raise
    return models_cache

def map_to_common(pred, labels):
    out = np.zeros(len(COMMON_EMOTIONS))
    for i, l in enumerate(labels):
        if l in COMMON_EMOTIONS:
            idx = COMMON_EMOTIONS.index(l)
            out[idx] = pred[0][i]
    return out

def fusion_prediction(file):
    vec, mel = extract_features(file)
    m = get_models()
    
    p1 = m['emodb'].predict(vec, verbose=0)
    p2 = m['emovo'].predict(vec, verbose=0)
    p3 = m['shemo'].predict(mel, verbose=0)

    p1 = map_to_common(p1, emodb_labels)
    p2 = map_to_common(p2, emovo_labels)
    p3 = map_to_common(p3, shemo_labels)

    conf = np.array([np.max(p1), np.max(p2), np.max(p3)])
    weights = np.exp(conf) / np.sum(np.exp(conf))

    final_prob = weights[0]*p1 + weights[1]*p2 + weights[2]*p3
    emotion = COMMON_EMOTIONS[np.argmax(final_prob)]
    confidence = float(np.max(final_prob))

    return emotion, confidence, final_prob

def predict_audio_emotion(audio_path):
    """
    Main interface for audio emotion prediction used by app.py
    """
    try:
        emotion, confidence, probs = fusion_prediction(audio_path)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probs.tolist()
        }
    except Exception as e:
        import traceback
        log_path = os.path.join(BASE_DIR, "..", "..", "audio_error.log")
        with open(log_path, "a") as f:
            f.write(f"\n--- Error at {os.path.basename(audio_path)} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"Error in audio prediction pipeline: {e}")
        # Return a safe default with low confidence
        return {
            "emotion": "neutral",
            "confidence": 0.3,
            "probabilities": [0.0] * len(COMMON_EMOTIONS)
        }
