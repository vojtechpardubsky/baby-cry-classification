from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import librosa
import numpy as np
import tempfile
import os
import subprocess

from app.preprocessing import extract_features_from_audio
from app.inference import predict_class

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
TARGET_DURATION = 7

@app.get("/healthz")
def health():
    return {"status": "ok"}

def convert_to_wav(input_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        output_path = temp_wav.name

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "8000",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        error_text = e.stderr.decode(errors="ignore")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(
            status_code=400,
            detail=f"Nepodařilo se převést audio soubor pomocí FFmpeg. Detail: {error_text}"
        )

    return output_path

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"

    temp_audio_path = None
    converted_audio_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(contents)
        temp_audio_path = temp_audio.name

    try:
        converted_audio_path = convert_to_wav(temp_audio_path)

        audio, sr = librosa.load(converted_audio_path, sr=8000)

        target_length = sr * TARGET_DURATION

        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")

        features = extract_features_from_audio(audio, sr)

        prediction = predict_class(features)

        return {
            "predicted_class": prediction
        }

    finally:
        for path in [temp_audio_path, converted_audio_path]:
            if path and os.path.exists(path):
                os.remove(path)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
