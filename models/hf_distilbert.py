from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"[Model Load Error] Make sure the model is downloaded locally.\n{e}")
    qa_pipeline = None


def extract_experience_hf_distilbert(jd_text: str) -> str:
    if qa_pipeline is None:
        return "Error: Model not loaded"

    question = "How many years of experience is required for this job?"

    try:
        result = qa_pipeline(question=question, context=jd_text)
        answer = result["answer"].strip()

        if any(char.isdigit() for char in answer):
            return answer
        else:
            return "Not mentioned"

    except Exception as e:
        print(f"[HF DistilBERT Error] {e}")
        return "Error"
