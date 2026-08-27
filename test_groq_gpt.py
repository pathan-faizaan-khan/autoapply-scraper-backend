import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

client = Groq(api_key=api_key)

print("Calling Groq API for qwen/qwen3.8-27b...")

try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, are you GPT-OSS 120B?",
            }
        ],
        model="qwen/qwen3.8-27b",
    )
    print("\nSUCCESS! The model is active and responded:")
    print('-----------------------------------------')
    print(chat_completion.choices[0].message.content)
    print('-----------------------------------------')
except Exception as e:
    print(f"Error: {e}")
