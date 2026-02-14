import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import List, Tuple, Optional
from datetime import datetime

import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from pydantic import BaseModel
from pydub import AudioSegment

# Import existing inference logic
from w2v2.inference import PhonemeRecognizer, InferenceConfig

# --- Configuration ---
# Allow configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/phoneme-model")
# Default to None to let InferenceConfig decide (prevents early CUDA init in main process)
DEVICE = os.getenv("DEVICE") or None

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
# Number of worker processes (default: 1). Set >1 for parallelism.
# Each worker loads its own copy of the model (High RAM usage).
WORKERS = int(os.getenv("WORKERS", "2"))

# Global State
recognizer: Optional[PhonemeRecognizer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Life-span event handler to load the model on startup.
    This ensures we only load the heavy model once.
    """
    global recognizer
    print(f"INFO: Initializing phoneme recognition server...")
    print(f"INFO: Model Path: {MODEL_PATH}")
    print(f"INFO: Device request: {DEVICE if DEVICE else 'auto-detect'}")
    
    try:
        # Initialize the config and recognizer
        # InferenceConfig handles device auto-detection if passed None
        config = InferenceConfig(model_path=MODEL_PATH, device=DEVICE)
        
        print("INFO: Loading model weights (this may take a moment)...")
        recognizer = PhonemeRecognizer(config)
        
        print(f"SUCCESS: Model loaded on device: {recognizer.device}")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        traceback.print_exc()
        # We allow the server to start even if the model fails, 
        # so /health can report the error, but /infer will fail.
    
    yield
    
    print("INFO: Shutting down...")
    recognizer = None

# Create FastAPI app
app = FastAPI(
    title="Hindi Phoneme Recognition API",
    description="Inference server for converting Hindi speech to timestamped phonemes.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for Response ---

class Segment(BaseModel):
    phoneme: str
    start: float
    end: float

class TranscriptionResponse(BaseModel):
    phonemes: List[str]
    timestamps: List[Tuple[float, float]]
    segments: List[Segment]
    duration_seconds: float
    filePath: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str

# --- Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Hindi Phoneme Recognition API is running",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Check system status and model availability."""
    return {
        "status": "online",
        "model_loaded": recognizer is not None,
        "device": str(recognizer.device) if recognizer else "unknown",
        "model_path": MODEL_PATH
    }

@app.post("/infer", response_model=TranscriptionResponse, tags=["Inference"])
def transcribe_audio(
    file: UploadFile = File(...),
    extend_timestamps: bool = Query(True, description="Extend each phoneme duration to close gaps between them."),
    raw_output: bool = Query(False, description="Return raw frame predictions without CTC collapse (debugging only).")
):
    """
    Upload an audio file (wav, mp3, etc.) and get timestamped phonemes.
    Also saves the audio file with a random filename.
    """
    if recognizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded on the server. Please check server logs."
        )

    # 1. Create a temporary file for the uploaded audio
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".wav"  # Default fallback
        
    tmp_path = None
    saved_abs_path = ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Efficiently copy uploaded file to disk
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # 1.5 Save the audio file to repository root "data" folder
        try:
            # Define destination directory (repository root)
            save_dir = os.path.abspath("data")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"INFO: Created directory {save_dir}")

            # Construct filename
            # Generate a random hex filename
            filename = f"{uuid.uuid4().hex}.mp3"
            target_path = os.path.join(save_dir, filename)
            
            # Convert to mp3 if needed and save
            audio = AudioSegment.from_file(tmp_path)
            audio.export(target_path, format="mp3")
            
            saved_abs_path = os.path.abspath(target_path)
            print(f"INFO: Saved audio to {saved_abs_path}")

        except Exception as save_error:
            print(f"WARNING: Failed to save audio file: {save_error}")
            traceback.print_exc()
            # We continue with inference even if saving fails, but filepath will be empty or handle appropriately
            saved_abs_path = ""

        # 2. Run inference
        # PhonemeRecognizer reads the file from disk using torchaudio
        try:
            if raw_output:
                phonemes, timestamps = recognizer.recognize_raw(tmp_path)
            else:
                phonemes, timestamps = recognizer.recognize(
                    tmp_path,
                    extend_timestamps=extend_timestamps,
                    use_ctc_segmentation=False # We don't support text alignment in this endpoint yet
                )
        except Exception as inf_error:
            # Wrap inference errors (e.g. corrupt audio, CUDA OOM)
            raise HTTPException(status_code=422, detail=f"Inference error: {str(inf_error)}")

        # 3. Format response
        segments = []
        if timestamps:
            segments = [
                Segment(phoneme=p, start=t[0], end=t[1])
                for p, t in zip(phonemes, timestamps)
            ]
            duration = timestamps[-1][1]
        else:
            duration = 0.0

        return {
            "phonemes": phonemes,
            "timestamps": timestamps,
            "segments": segments,
            "duration_seconds": duration,
            "filePath": saved_abs_path
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
        
    finally:
        # 4. Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass # Best effort cleanup

if __name__ == "__main__":
    # Run standalone
    print(f"Starting server on {HOST}:{PORT} with {WORKERS} workers")
    uvicorn.run("server:app", host=HOST, port=PORT, workers=WORKERS, reload=False)
