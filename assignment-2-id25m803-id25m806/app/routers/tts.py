from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.synthesizer import generate_speech
import socket
import base64

router = APIRouter()

class SpeechRequest(BaseModel):
    text: str
    language: str = "en"

@router.post("/speak")
async def text_to_speech(request: SpeechRequest):
    """
    Converts text to audio.
    Returns base64 audio + container_id (required for Swarm).
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_bytes = generate_speech(request.text, request.language)

        # Convert audio to base64 to send JSON
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "container_id": socket.gethostname()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
