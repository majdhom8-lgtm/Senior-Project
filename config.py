import os
from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


MODEL = OPENAI_MODEL


MAX_STEPS = 12       
MAX_RETRIES = 3      
TEMPERATURE = 0.1  
MAX_TOKENS = 2000    

if not OPENAI_API_KEY:
    print("═══════════════════════════════════════════════════════")
    print("  خطأ: لم يتم العثور على OPENAI_API_KEY في ملف .env")
    print("═══════════════════════════════════════════════════════")
    exit(1)