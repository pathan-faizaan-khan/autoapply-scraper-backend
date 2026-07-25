import asyncio
from dotenv import load_dotenv
load_dotenv()
from services.llm_service import LLMService

async def main():
    llm = LLMService()
    try:
        response = await llm.generate_text(system_prompt='You are a helpful assistant.', user_prompt='Say hello')
        print(response.content)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
