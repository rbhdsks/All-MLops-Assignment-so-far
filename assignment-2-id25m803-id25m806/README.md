# Assignment 2 – AI Microservices API
### Kowshik Arko Dey (ID25M803) | Nitesh Kumar Shah (ID25M806)

A modular FastAPI-based AI service platform providing:

- Text Translation  
- Named Entity Recognition (NER)  
- Image Generation  
- Text-to-Speech (TTS)  

This project follows clean architecture principles with proper separation of routers, services, and tests.

---

# Project Structure

```
assignment-2/
├── app/
│   ├── __init__.py
│   ├── config.py                  # Application configuration
│   ├── main.py                    # FastAPI application entry point
│   │
│   ├── routers/                   # API route definitions
│   │   ├── translation.py
│   │   ├── ner.py
│   │   ├── image_gen.py
│   │   └── tts.py
│   │
│   └── services/                  # Business logic & external integrations
│       ├── translator.py
│       ├── analyzer.py
│       ├── imager.py
│       └── synthesizer.py
│
├── tests/                         # Automated unit tests
│   ├── test_image.py
│   ├── test_ner.py
│   └── test_tts.py
│
├── docker-stack.yml               # Docker Swarm deployment configuration
├── Dockerfile                     # Container build configuration
│
├── CONFLICT.md                    # Merge conflict documentation (Assignment requirement)
├── tester.py                      # Manual testing script
│
├── output_audio.mp3               # Generated audio output (example)
├── output_image.png               # Generated image output (example)
├── test_audio.mp3                 # Test audio file
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
└── Developer_A_1.mp4              # Demo video 1
    Developer_A_2.mp4              # Demo video 2
    implementation part 2 demo.mp4 # Implementation demo
```

---

# Setup Instructions

## 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file in the root directory and add required API keys:

```
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

---

# Run the Server

```bash
uvicorn app.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Demonstration

## Translation API

**Endpoint**
```
POST /api/v1/translate
```
### Windows
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/translate" `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "Hello world", "source_lang": "en", "target_lang": "fr"}'
```

### curl
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "source_lang": "en", "target_lang": "fr"}'
```

---

## Named Entity Recognition (NER)

**Endpoint**
```
POST /api/v1/ner
```

### Windows
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/ner" `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "Apple is looking at buying U.K. startup for $1 billion"}'
```

### curl
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ner" \
     -H "Content-Type: application/json" \
     -d '{"text": "Apple is looking at buying U.K. startup for $1 billion"}'
```

---

## Image Generation

**Endpoint**
```
POST /api/v1/generate-image
```
### Windows
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/generate-image" `
  -Method Post -ContentType "application/json" `
  -Body '{"prompt": "A cyberpunk robot"}' `
  -OutFile "output_image.png"
```
### curl
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate-image" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A cyberpunk robot"}' \
     --output output_image.png
```

---

## Text-to-Speech (TTS)

**Endpoint**
```
POST /api/v1/speak
```
### Windows
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/speak" `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "Welcome to our AI service demonstration.", "language": "en"}' `
  -OutFile "output_audio.mp3"
```
### curl
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/speak" \
     -H "Content-Type: application/json" \
     -d '{"text": "Welcome to our AI service demonstration.", "language": "en"}' \
     --output output_audio.mp3
```

---

# Running Tests

```bash
pytest
```

Tests validate:

- Response correctness  
- API status codes  
- Output formats  
- Error handling  

---

# Architecture Design

This project follows a layered architecture pattern:

- Routers Layer → Defines API endpoints  
- Service Layer → Handles business logic and third-party integrations  
- Tests Layer → Validates functionality  

This separation ensures:

- Maintainability  
- Scalability  
- Clean code organization  
- Better testability

# Dockerization & Deployment

## 1. Build Docker Image
To build the docker image locally, ensure the docker is running and execute
```bash
docker build -t multimodal-api .
```

## 2. Deploy with Docker Swarm
Docker swarm is used to spin up multiple instances of the API. The *docker-stack.yml* file is configured to run 4 replicas with an automatic restart policy on failure.
**Initialize Docker Swarm**
```bash
docker swarm init
```
**Deploy the stack**
```bash
docker stack deploy -c docker-stack.yml ai_microservices
```
**Verification**
```bash
docker service ls
```

## 3. Load Balancing Test
To verify that Docker Swarm is correctly distributing incoming traffic across all 4 replicas, it has been included a concurrency test script (tester.py).
```bash
python tester.py
```


# AI Usage Disclosure

## AI Assistance Statement

This project was developed as part of the DA5402 MLOps assignment.

During the development process, AI-based tools (including large language models such as ChatGPT) were used as supportive aids for the following purposes:

- Clarifying conceptual doubts related to FastAPI, Docker, Git, and deployment workflows  
- Debugging error messages and resolving merge conflicts  
- Improving code organization and formatting  
- Assisting with documentation writing and README structuring  
- Understanding testing and containerization strategies  

AI tools were **not used to generate a complete solution without understanding**. All implementation decisions, integrations, and final code were reviewed, modified, and validated manually.

## Independent Verification

All code submitted in this repository:

- Was written and/or adapted with full understanding  
- Was tested locally before submission  
- Meets the functional and structural requirements of the assignment  
- Was integrated and debugged independently  

## Responsibility Statement

We (Kowshik ID25M803 and Nitesh ID25M806) take full responsibility for the correctness, integrity, and originality of the submitted work. AI tools were used strictly as learning and productivity aids, and not as a replacement for individual effort or understanding.
