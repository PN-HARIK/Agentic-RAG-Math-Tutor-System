import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-2.5-flash") 
    response = model.generate_content("Say hello world!")
    print("✅ Gemini API key working (Free model)")
    print("Response:", response.text)
except Exception as e:
    print("❌ Gemini API key not working!")
    print("Error:", e)
