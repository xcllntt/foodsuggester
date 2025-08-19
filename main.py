import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-flash"

# FastAPI app
app = FastAPI(title="Naija Mood Meals")

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
    "You are a Nigerian food expert. Based on the user's mood and/or request, "
    "suggest exactly 2 authentic Nigerian meals. "
    "Each meal should have a short reason why it fits. "
    "Return JSON matching the schema."
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/suggestions", response_model=SuggestionResponse)
def get_suggestions(req: SuggestionRequest):
    try:
        if not req.mood and not req.prompt:
            raise HTTPException(status_code=400, detail="At least one of 'mood' or 'prompt' must be provided")

        # Build dynamic prompt
        prompt_parts = []
        if req.mood:
            prompt_parts.append(f"Mood: {req.mood}")
        if req.prompt:
            prompt_parts.append(f"Request: {req.prompt}")

        prompt = "\n".join(prompt_parts)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SuggestionResponse,
            temperature=0.8,
        )

        result = client.models.generate_content(
            model=MODEL,
            contents=prompt,
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