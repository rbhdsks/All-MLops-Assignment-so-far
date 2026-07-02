import spacy

# Load the small English model
# In a real app, you might load this globally to avoid reloading on every request
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model isn't found (useful for local dev before download)
    print("Downloading model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def perform_ner(text: str) -> dict:
    """
    Extracts named entities from the text using spaCy.
    """
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
    return {
        "original_text": text,
        "entities": entities
    }