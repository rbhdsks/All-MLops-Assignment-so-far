import requests
import urllib.parse

# We use DiceBear because it is a rock-solid, public API.
# It generates unique robot/avatar images based on the text seed.
# No keys, no logins, no 403 errors.
BASE_URL = "https://api.dicebear.com/9.x/bottts/png"

def generate_image(prompt: str):
    """
    Generates a unique AI avatar image based on the prompt.
    Returns binary image data (PNG).
    """
    try:
        # 1. URL Encode the prompt so spaces/symbols are safe
        seed = urllib.parse.quote(prompt)
        
        # 2. Construct the URL (e.g., .../png?seed=A%20Cyberpunk%20Cat)
        url = f"{BASE_URL}?seed={seed}"
        
        print(f"DEBUG: Fetching image from {url}")
        
        # 3. Fetch the image
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
            
        return response.content
        
    except Exception as e:
        # If even this fails (offline?), fallback to a simple placeholder
        print(f"DEBUG: DiceBear failed ({e}). Using text placeholder.")
        fallback_url = f"https://dummyimage.com/600x400/000/fff.png?text=Error"
        return requests.get(fallback_url).content