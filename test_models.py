import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

for model in ["gemini-embedding-001", "gemini-embedding-2"]:
    try:
        response = client.models.embed_content(
            model=model,
            contents="Hello world",
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        print(f"SUCCESS {model}: dims={len(response.embeddings[0].values)}")
    except Exception as e:
        print(f"ERROR {model}:", e)

