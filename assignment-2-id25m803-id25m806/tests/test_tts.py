from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
import io

client = TestClient(app)

# We mock 'app.services.synthesizer.gTTS' so we don't hit Google's servers

@patch("app.services.synthesizer.gTTS")
def test_speech_generation_success(mock_gtts):
    """
    Test successful speech generation by MOCKING the gTTS library.
    """
    # 1. Setup the Mock
    # We need to mock the 'write_to_fp' method of the gTTS object
    mock_tts_instance = MagicMock()
    
    # When write_to_fp(fp) is called, we write fake bytes into that file pointer
    def side_effect(fp):
        fp.write(b"fake_mp3_audio_content")
        
    mock_tts_instance.write_to_fp.side_effect = side_effect
    
    # Make the gTTS constructor return our mock instance
    mock_gtts.return_value = mock_tts_instance

    # 2. Run the Test
    payload = {"text": "Hello World", "language": "en"}
    response = client.post("/api/v1/speak", json=payload)
    
    # 3. Assertions

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"fake_mp3_audio_content"

def test_empty_text():
    """
    Test validation for empty text.
    """
    payload = {"text": ""}
    response = client.post("/api/v1/speak", json=payload)
    
    # Expect 400 Bad Request

    assert response.status_code == 400