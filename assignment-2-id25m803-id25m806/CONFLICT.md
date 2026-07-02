# Merge Conflict Documentation

## First Conflict faced 
- It was faced by the developer B when he was trying to pull from main but mistakely when he pulled it into the branch.. the conflict happened and then the developer B had to do reset hard  from his branch to come to the previous version of the code... and then after " git stash".. git pull and git pop helped..


## Second one
## Branches Involved
- main
- feature/tts-service

## File with Conflict
app/main.py

## Cause of Conflict
Both branches modified the FastAPI app title in the same line.

- main branch was deliberately changed it to: "Multi-Modal API v1" to perform the conflict
- feature branch changed it to: "Multi-Modal AI REST Service" as the "feature/tts-service"

Since the same line was edited differently, Git could not automatically merge and hence the conflict happened

## Conflict Markers Observed

```text
<<<<<<< HEAD
app = FastAPI(title="Multi-Modal AI REST Service")
=======
app = FastAPI(title="Multi-Modal API v1")
>>>>>>> main
```

## Resolution
We resolved the conflict by standardizing the application title to:

app = FastAPI(title="Multi-Modal AI REST Service") .... was was basically replacing it to the newer same version

## Learning
Frequent synchronization with the main branch reduces the probability of merge conflicts.


## Third conflict faced
It was faced by developer A when trying to make a correction to the image-generation API, after all features of developer B and A have been pushed. This was solved by manually recording all the conflicts and then deleting the conflicting region

### Branches involved
- main
- image-gen

### Files with conflict
- app/main.py
- services/synthesizer.py
- tests/test_tts.py

### Cause of Conflict
Developer A branch was commits behind the already merged production branch of the code, so it must be pulled first for developer A to make further changes which caused the conflict

- **app/main.py**
```text
<<<<<<< HEAD
#Define the app
app = FastAPI(title="Multi-Modal AI Service")
# Define the router
# Register Translation Router
app.include_router(translation.router, prefix="/api/v1", tags=["Translation"])
# Register NER router
app.include_router(ner.router, prefix="/api/v1", tags=["NER"])
# Register Image Generation Router
app.include_router(image_gen.router, prefix="/api/v1", tags=["Image Generation"])
# Register TTS router
=======

app = FastAPI(title="Multi-Modal AI REST Service")

app.include_router(translation.router, prefix="/api/v1", tags=["Translation"])

app.include_router(ner.router, prefix="/api/v1", tags=["NER"])
app.include_router(image_gen.router, prefix="/api/v1", tags=["Image Generation"])
```

- **services/synthesizer.py**
```text
<<<<<<< HEAD
        # Create the TTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to an in-memory file instead of disk
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        
        # Reset the file pointer to the beginning so we can read it
=======
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
>>>>>>> 5999845b76d1469f518336ea82eba609bbf8d59e
        mp3_fp.seek(0)
```

- **tests/test_tts.py**
```text
<<<<<<< HEAD

# We mock 'app.services.synthesizer.gTTS' so we don't hit Google's servers
=======
>>>>>>> 5999845b76d1469f518336ea82eba609bbf8d59e
@patch("app.services.synthesizer.gTTS")
def test_speech_generation_success(mock_gtts):
    """
    Test successful speech generation by MOCKING the gTTS library.
    """
<<<<<<< HEAD
    # 1. Setup the Mock
    # We need to mock the 'write_to_fp' method of the gTTS object
    mock_tts_instance = MagicMock()
    
    # When write_to_fp(fp) is called, we write fake bytes into that file pointer
=======
    mock_tts_instance = MagicMock()
>>>>>>> 5999845b76d1469f518336ea82eba609bbf8d59e
```