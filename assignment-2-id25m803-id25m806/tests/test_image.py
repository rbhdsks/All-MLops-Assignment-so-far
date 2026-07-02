from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

#Start the test client
client = TestClient(app)

@patch("app.services.imager.requests.get")
def test_image_generation_success(mock_get):
    """
    Test successful image generation
    
    Args:
        mock_get (MagicMock): The mock object for requests.get
        
    Returns:
        None
    """
    # Setup a fake response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake_image_bytes_data"
    mock_response.headers = {"content-type": "image/jpeg"}
    # Return mock response
    mock_get.return_value = mock_response
    # Run the tests.
    payload = {"prompt": "A futuristic city"}
    response = client.post("/api/v1/generate-image", json=payload)
    #ASSERT
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"fake_image_bytes_data"

def test_empty_prompt():
    """
    TValidation test for empty string prompt
    """
    payload = {"prompt": ""}
    response = client.post("/api/v1/generate-image", json=payload)
    # 400 Bad Requests
    assert response.status_code == 400