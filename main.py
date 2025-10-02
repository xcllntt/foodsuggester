import os
import json
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-flash-001"

# FastAPI app
app = FastAPI(title="Naija Mood Meals")
# Allow cross-origin requests (important for multiple frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://moodtofood.lovable.app", "https://foodsuggester.netlify.app"],  # you can restrict to ["http://localhost:3000", "https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request/Response models
class SuggestionRequest(BaseModel):
    mood: Optional[str] = None
    prompt: Optional[str] = None

class MealSuggestion(BaseModel):
    name: str
    why_it_fits: str

class SuggestionResponse(BaseModel):
    meals: list[MealSuggestion]

# Prompt template
SYSTEM_INSTRUCTION = (
    "You are a Nigerian food expert. Based on the user's mood or custom prompt, "
    "suggest exactly 2 authentic Nigerian meals. Each meal should have a short reason why it fits. "
    "Return JSON matching the schema."
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/suggestions", response_model=SuggestionResponse)
def get_suggestions(req: SuggestionRequest):
    try:
        # Ensure at least one input is provided
        if not req.mood and not req.prompt:
            raise HTTPException(
                status_code=400,
                detail="You must provide at least 'mood' or 'prompt'."
            )

        # Build the final prompt
        if req.mood and req.prompt:
            user_prompt = f"Mood: {req.mood}\nExtra Prompt: {req.prompt}\nSuggest meals."
        elif req.mood:
            user_prompt = f"Mood: {req.mood}\nSuggest meals."
        else:
            user_prompt = f"Prompt: {req.prompt}\nSuggest meals."

        # Gemini config
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SuggestionResponse,
            temperature=0.8,
        )

        result = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=config,
        )

        if hasattr(result, "parsed") and result.parsed:
            return result.parsed

        text = getattr(result, "text", None)
        if text:
            return SuggestionResponse(**json.loads(text))

        raise HTTPException(status_code=500, detail="No response from model")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))