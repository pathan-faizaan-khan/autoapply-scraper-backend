import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

client = Groq(api_key=api_key)

try:
    models = client.models.list()
    print("Available Groq Models:")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"Error: {e}")
