from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.analyzer import perform_ner
import socket

router = APIRouter()

class NERRequest(BaseModel):
    text: str


@router.post("/ner")
async def extract_entities(request: NERRequest):
    """
    Endpoint to extract Named Entities (Person, Org, Date, etc.)
    """

    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = perform_ner(request.text)

        return {
            "entities": result,
            "container_id": socket.gethostname()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
