import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    mood: str

class MealSuggestion(BaseModel):
    name: str
    why_it_fits: str

class SuggestionResponse(BaseModel):
    meals: list[MealSuggestion]

# Prompt template
SYSTEM_INSTRUCTION = (
    "You are a Nigerian food expert. Based on the user's mood, suggest exactly 2 authentic Nigerian meals. "
    "Each meal should have a short reason why it fits the mood. Return JSON matching the schema."
)

PROMPT_TEMPLATE = "Mood: {mood}\nSuggest 3 meals."

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/suggestions", response_model=SuggestionResponse)
def get_suggestions(req: SuggestionRequest):
    try:
        prompt = PROMPT_TEMPLATE.format(mood=req.mood)

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
