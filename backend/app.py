import sys
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import tensorflow as tf


# ===============================
# TensorFlow CPU Optimization
# ===============================
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ===============================
# Project Path Setup
# ===============================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)


# ===============================
# Imports
# ===============================
from fusion.amcef import AMCEF
from backend.audio.fusion_predict import predict_audio_emotion
from backend.video_model import predict_video


# ===============================
# FastAPI
# ===============================

app = FastAPI(
    title="AMCEF Emotion Recognition API",
    version="2.0-fusion-upgrade"
)


@app.get("/")
def health():
    return {
        "status": "AMCEF API running"
    }


# ===============================
# Face Detection
# ===============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ===============================
# Quality Scores
# ===============================

def audio_quality(size):

    if size > 400000:
        return 0.9

    if size > 150000:
        return 0.7

    return 0.5



def video_quality(size):

    if size > 2_000_000:
        return 0.9

    if size > 700000:
        return 0.7

    return 0.5



# ===============================
# Save Upload
# ===============================

def save_upload(upload):

    suffix = os.path.splitext(upload.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as file:

        data = upload.file.read()

        file.write(data)

        return file.name, len(data)



# ===============================
# Face Detection Optimized
# ===============================

def detect_face(video_path):

    try:

        cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()

        cap.release()


        if not ret:
            return False


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            1.1,
            4
        )


        return len(faces) > 0


    except Exception as e:

        print(
            "Face detection error:",
            e
        )

        return True



# ===============================
# Prediction API
# ===============================

@app.post("/predict")
async def predict(

    audio_file: Optional[UploadFile] = File(None),

    video_file: Optional[UploadFile] = File(None)

):


    response = {}


    # -------------------
    # Audio
    # -------------------

    ae = "neutral"
    ac = 0
    a_probs = []


    if audio_file:

        audio_path = None

        try:

            audio_path, size = save_upload(
                audio_file
            )


            result = predict_audio_emotion(
                audio_path
            )


            ae = result["emotion"]


            quality = audio_quality(
                size
            )


            ac = round(
                result["confidence"] * quality,
                3
            )


            a_probs = result.get(
                "probabilities",
                []
            )


            response["audio"] = {

                "emotion": ae,

                "confidence": ac,

                "quality": quality,

                "probabilities": a_probs

            }


        finally:

            if audio_path and os.path.exists(audio_path):

                os.unlink(audio_path)


    else:

        response["audio"] = None



    # -------------------
    # Video
    # -------------------

    ve = "neutral"

    vc = 0

    v_probs = []


    if video_file:


        video_path = None


        try:


            video_path, size = save_upload(
                video_file
            )


            face = detect_face(
                video_path
            )


            if not face:


                ve = "no_face"

                vc = 0


                response["video"] = {

                    "emotion": ve,

                    "confidence": vc,

                    "quality": 0,

                    "face_detected": False,

                    "probabilities": []

                }


            else:


                result = predict_video(
                    video_path
                )


                ve = result["emotion"]


                quality = video_quality(
                    size
                )


                vc = round(
                    result["confidence"] * quality,
                    3
                )


                response["video"] = {

                    "emotion": ve,

                    "confidence": vc,

                    "quality": quality,

                    "face_detected": True,

                    "probabilities":
                        result.get(
                            "probabilities",
                            []
                        )

                }



        finally:


            if video_path and os.path.exists(video_path):

                os.unlink(video_path)



    else:


        response["video"] = None



    # -------------------
    # Fusion
    # -------------------

    fusion = AMCEF.fuse_weighted(

        ae,

        ac,

        ve,

        vc

    )


    strategy = "weighted_fusion"


    if fusion["eri_score"] > 0.75:

        strategy = "agreement_boost"


    elif ac > vc + 0.20:

        strategy = "audio_dominance"



    response["fusion"] = {


        "final_emotion":

            fusion["fused_emotion"],


        "final_confidence":

            fusion["fused_confidence"],


        "fusion_strategy":

            strategy,


        "source_used":

            fusion["source"],


        "eri":

            {

            "score":

                fusion["eri_score"],


            "level":

                fusion["eri_level"]

            },


        "explanation":

            fusion["explanation"]

    }



    response["temporal_analysis"] = {

        "dominant_emotion":

            ve if ve != "neutral" else ae,


        "emotion_stability_score":

            round(max(ac, vc),2),


        "stability_level":

            "HIGH"

            if max(ac,vc)>0.75

            else "MEDIUM"

    }



    response["reliability_summary"] = (

        "Highly reliable"

        if fusion["eri_level"]=="HIGH"

        else "Moderate reliability"

    )



    return response



if __name__ == "__main__":

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000

    )
