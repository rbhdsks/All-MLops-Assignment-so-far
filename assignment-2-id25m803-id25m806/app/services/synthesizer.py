from gtts import gTTS
import io

def generate_speech(text: str, lang: str = 'en'):
    """
    Converts text to speech using Google TTS.
    Returns the audio binary data (MP3 format).
    """
    try:

        # Create the TTS object
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save to an in-memory file instead of disk
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        # Reset the file pointer to the beginning so we can read it

        mp3_fp.seek(0)
        
        return mp3_fp.getvalue()
        
    except Exception as e:
        raise Exception(f"TTS Generation Failed: {str(e)}")