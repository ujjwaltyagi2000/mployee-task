import os
from dotenv import load_dotenv
import spacy
from transformers import pipeline
from openai import OpenAI

print("🔍 Verifying setup...")

# 1. Load .env and check GROQ_API_KEY
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("❌ GROQ_API_KEY not found in .env file.")
else:
    print("✅ GROQ_API_KEY loaded successfully.")

# 2. Check SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model 'en_core_web_sm' is installed.")
except OSError:
    print("❌ SpaCy model 'en_core_web_sm' is NOT installed.")
    print("👉 Run: python -m spacy download en_core_web_sm")

# 3. Check Hugging Face transformer loading
try:
    _ = pipeline("question-answering", model="deepset/roberta-base-squad2")
    print("✅ Transformers model 'deepset/roberta-base-squad2' is accessible.")
except Exception as e:
    print(f"❌ Transformers model load failed: {e}")

# 4. Test Groq API call with dummy input
try:
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "How many years of experience are required?"}
        ],
        temperature=0.1,
        max_tokens=10,
    )
    print("✅ Groq API test call succeeded.")
except Exception as e:
    print(f"❌ Groq API call failed: {e}")

print("✅ Verification complete.")
