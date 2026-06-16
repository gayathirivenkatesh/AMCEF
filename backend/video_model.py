import os
import csv

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
CSV_PATH = os.path.join(ASSET_DIR, "video_results_100.csv")

VIDEO_DB = {}

def load_video_predictions():
    if not os.path.exists(CSV_PATH):
        print("⚠ video_results_100.csv not found")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row

        for row in reader:
            if len(row) < 3:
                continue

            filename = row[0].strip()
            emotion = row[1].strip().lower()
            conf = float(row[2])

            VIDEO_DB[filename] = (emotion, conf)

    print(f"✅ Loaded {len(VIDEO_DB)} video predictions")


load_video_predictions()


def predict_video(filename: str):
    """
    Lookup video prediction from CSV using filename (Simulation)
    """
    base = os.path.basename(filename)
    
    result = ("neutral", 0.45)
    
    if base in VIDEO_DB:
        result = VIDEO_DB[base]
    else:
        # fallback if exact name not found
        for key in VIDEO_DB:
            if key in base:
                result = VIDEO_DB[key]
                break

    return {
        "emotion": result[0],
        "confidence": result[1]
    }
