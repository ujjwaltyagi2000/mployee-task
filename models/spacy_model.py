import spacy

WORD_NUM_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("[SpaCy] Model not found. Downloading 'en_core_web_sm'...")
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_experience_years(text, nlp):
    text = text.lower()
    doc = nlp(text)

    # First: Spacy-based entity detection
    for ent in doc.ents:
        if ent.label_ in ["CARDINAL", "QUANTITY", "ORDINAL"]:
            window = text[ent.start_char : ent.end_char + 20]
            if "year" in window or "yrs" in window:
                return ent.text.strip() + " years"

    # Second: Manual word-number scanning
    for word, digit in WORD_NUM_MAP.items():
        if f"{word} year" in text or f"{word} years" in text:
            return digit + " years"

    return "Not mentioned"
