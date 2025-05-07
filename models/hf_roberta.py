from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "deepset/roberta-base-squad2"

# Load tokenizer and model locally (won’t download from the internet)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"[Model Load Error] Make sure the model is downloaded locally.\n{e}")
    qa_pipeline = None


def extract_experience_hf_roberta(jd_text: str) -> str:
    if qa_pipeline is None:
        return "Error: Model not loaded"

    question = "How many years of experience is required for this job?"

    try:
        result = qa_pipeline({"context": jd_text, "question": question})
        answer = result["answer"].strip()

        if any(char.isdigit() for char in answer):
            return answer
        else:
            return "Not mentioned"

    except Exception as e:
        print(f"[HF RoBERTa Error] {e}")
        return "Error"
