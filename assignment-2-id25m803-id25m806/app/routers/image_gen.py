# Load the libraries
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.imager import generate_image
import socket
import base64

# Define the router
router = APIRouter()

# Image request class
class ImageRequest(BaseModel):
    prompt: str


@router.post("/generate-image")
async def create_image(request: ImageRequest):
    """
    Endpoint to generate an image from text.
    Returns base64 image + container_id (required for Swarm proof).
    """

    if not request.prompt or request.prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        image_bytes = generate_image(request.prompt)

        # Convert image bytes to base64 so we can send JSON
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "image_base64": image_base64,
            "container_id": socket.gethostname()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
