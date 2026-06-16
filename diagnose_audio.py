import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from backend.audio.fusion_predict import predict_audio_emotion
    print("Import successful")
    
    # Try to load models
    from backend.audio.fusion_predict import get_models
    print("Loading models...")
    models = get_models()
    print("Models loaded successfully")
    
    # Test with a dummy file (should fail gracefully or show file error)
    print("Testing prediction interface...")
    res = predict_audio_emotion("non_existent.wav")
    print(f"Result: {res}")

except Exception as e:
    import traceback
    traceback.print_exc()
