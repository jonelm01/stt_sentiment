import os
import time
import hashlib
import io
import librosa
import tensorflow as tf
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tempfile
from pathlib import Path 

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

FT_DIR = os.path.join(SRC_DIR, "sentiment-ecommerce-balanced")
if not os.path.isdir(FT_DIR):
    FT_DIR = os.path.join(PROJECT_ROOT, "sentiment-ecommerce-balanced")

FT_TOKENIZER_DIR = os.path.join(FT_DIR, "tokenizer")
FT_MODEL_DIR = os.path.join(FT_DIR, "model")

HITL_DIR = os.path.join(PROJECT_ROOT, "hitl_data")
HITL_FILE = os.path.join(HITL_DIR, "labels.csv")

# Config
FW_MODEL_ID = "small.en"
FW_DEVICE = "cpu"
FW_COMPUTE = "int8"

LABELS_UI = ["Negative", "Neutral", "Positive"]

# Utility functions
def stable_text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def stable_run_id(audio_bytes):
    h = hashlib.sha256()
    h.update(audio_bytes)
    h.update(FW_MODEL_ID.encode())
    return h.hexdigest()[:16]

def append_hitl_row(row):
    # CHANGED: use Path APIs so it works regardless of working directory
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])

    if HITL_FILE.exists():
        df_old = pd.read_csv(HITL_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["id"], keep="last")
    else:
        df_all = df_new

    df_all.to_csv(HITL_FILE, index=False)

# Model loaders
def load_whisper():
    return WhisperModel(FW_MODEL_ID, device=FW_DEVICE, compute_type=FW_COMPUTE)

def load_sentiment():
    # CHANGED: HF loaders need string paths
    tokenizer = AutoTokenizer.from_pretrained(str(FT_TOKENIZER_DIR))
    model = TFAutoModelForSequenceClassification.from_pretrained(str(FT_MODEL_DIR))
    return tokenizer, model

# Audio / Text processing
def load_audio(file_bytes):
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    return audio

def transcribe_bytes(file_bytes: bytes, filename: str, whisper_model):
    """
    Transcribe audio bytes by writing to a temporary file and letting faster-whisper decode it.
    This avoids librosa/soundfile limitations with .m4a (AAC).
    """
    suffix = ""
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        segments, _ = whisper_model.transcribe(tmp_path, language="en")
        return " ".join(s.text.strip() for s in segments)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def transcribe(audio, whisper_model):
    segments, _ = whisper_model.transcribe(audio, language="en")
    return " ".join(s.text.strip() for s in segments)

def analyze_sentiment(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="tf", truncation=True, max_length=512)
    out = model(**enc)
    probs = tf.nn.softmax(out.logits[0], axis=-1).numpy()
    return LABELS_UI[int(np.argmax(probs))], probs

# CLI runner - multiple files
def run_from_cli(audio_paths):
    """
    Run transcription + sentiment analysis on one or more audio files.
    audio_paths: list of file paths
    """
    whisper_model = load_whisper()
    sent_tokenizer, sent_model = load_sentiment()

    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"Error: File '{audio_path}' does not exist. Skipping.")
            continue

        print(f"\n=== Processing: {audio_path} ===")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio = load_audio(audio_bytes)

        print("Transcribing...")
        text = transcribe(audio, whisper_model)
        print("\nTranscript:\n", text)

        pred_label, probs = analyze_sentiment(text, sent_tokenizer, sent_model)
        conf = float(np.max(probs))
        print("\nPredicted Sentiment:")
        print(f"{pred_label} (confidence: {conf:.3f})")
        print(f"Probabilities - Neg: {probs[0]:.3f}, Neu: {probs[1]:.3f}, Pos: {probs[2]:.3f}")

        tid = stable_text_id(text)
        append_hitl_row({
            "id": tid,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "pred_label": pred_label.lower(),
            "pred_conf": conf,
            "p_neg": float(probs[0]),
            "p_neu": float(probs[1]),
            "p_pos": float(probs[2]),
            "human_label": "",
            "whisper_model": FW_MODEL_ID,
        })
        print("Saved results to labels.csv")
