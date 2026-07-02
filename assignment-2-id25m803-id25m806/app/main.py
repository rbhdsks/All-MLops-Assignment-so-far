from fastapi import FastAPI
from app.routers import translation, ner, image_gen, tts
import socket

# Define the app
app = FastAPI(title="Multi-Modal AI Service")

# Function to get container ID / hostname
def get_container_id():
    return socket.gethostname()

# Register Routers
app.include_router(translation.router, prefix="/api/v1", tags=["Translation"])
app.include_router(ner.router, prefix="/api/v1", tags=["NER"])
app.include_router(image_gen.router, prefix="/api/v1", tags=["Image Generation"])
app.include_router(tts.router, prefix="/api/v1", tags=["Speech Synthesis"])


@app.get("/")
def root():
    """
    Root endpoint
    Returns:
        dict: Service status with container metadata
    """
    return {
        "message": "AI Multi-Modal Service is Online",
        "container_id": get_container_id()
    }
