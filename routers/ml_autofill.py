import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter(prefix="/api/ml", tags=["ml-autofill"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class AutofillRequest(BaseModel):
    form_fields: list[dict]
    form_html: str
    user_data: dict

@router.post("/fill-form")
def fill_form_endpoint(req: AutofillRequest):
    """
    Uses an LLM to map a user's data to the scraped form fields on a job application.
    Now uses the raw form HTML as context so it doesn't have to guess what each field means!
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing in backend")

    # Limit to a reasonable size so we don't blow up context window
    fields = req.form_fields[:100]

    system_prompt = (
        "You are an AI assistant that auto-fills job application forms.\n"
        "You will be given a JSON object containing the user's data, a list of form fields, and a dense text representation of the form.\n"
        "The dense text representation contains all the visible text on the page, with input fields represented as tags like [FIELD:autoApplyId name=\"...\"] directly where they appear.\n"
        "Your task is to analyze this text to understand the exact visual context of each field (e.g., the text immediately preceding the [FIELD:...] tag is usually its label or question).\n"
        "Pay attention to field 'options' if they exist, and select the closest matching option exactly as written.\n"
        "If a field asks for boolean (checkboxes), return a boolean.\n"
        "If a field cannot be answered using the provided user data, omit it or guess a reasonable default if it's a generic question.\n"
        "Output ONLY a valid JSON object where the keys are the 'autoApplyId' of the form field, and the values are the filled data.\n"
        "DO NOT output markdown formatting like ```json. Output raw JSON.\n"
    )
    
    user_prompt = f"User Data:\n{json.dumps(req.user_data)}\n\nDense Form Text:\n{req.form_html}\n\nForm Fields to Fill:\n{json.dumps(fields)}"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        mapping_str = response.choices[0].message.content
        mapping = json.loads(mapping_str)

        return {"success": True, "mapping": mapping}

    except Exception as e:
        print(f"[ML Autofill] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
