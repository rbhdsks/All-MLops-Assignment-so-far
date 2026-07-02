# Load the required files
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.translator import perform_translation
import socket

# Define the router
router = APIRouter()

# Define the expected input using Pydantic
class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "es"  # Default language Spanish


@router.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translates text to target language.
    Returns translation result + container_id (required for Swarm).
    """

    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = perform_translation(request.text, request.target_lang)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        **result,  # keep existing keys (original_text, translated_text, etc.)
        "container_id": socket.gethostname()
    }
