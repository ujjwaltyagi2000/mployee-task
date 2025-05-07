from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # This will load variables from a .env file into the environment

# Load API key (set this in your .env or hardcode for now)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)


def extract_experience_llama3(jd_text: str) -> str:
    prompt = (
        "Only analyze job descriptions written in English. "
        "Extract only the number of years of experience required. "
        "Only return a number followed by the word 'years', like '3 years'. "
        "If it's not mentioned or the text is not in English, return 'Not mentioned'. "
        "Do not include any explanation.\n\n"
        f"{jd_text}\n\n"
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error"
