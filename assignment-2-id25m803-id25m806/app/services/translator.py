from deep_translator import GoogleTranslator

def perform_translation(text: str, target_lang: str = 'en') -> dict:
    """
    Translates text to the target language using Google Translate (free tier).
    """
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = translator.translate(text)
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "target_language": target_lang
        }
    except Exception as e:
        return {"error": str(e)}