from transformers import pipeline

# Load once globally (alternatively wrap this in a load_model function if needed)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def extract_experience_hf_roberta(jd_text: str) -> str:
    question = "How many years of experience is required for this job?"

    try:
        result = qa_pipeline({"context": jd_text, "question": question})

        answer = result["answer"].strip()

        # Post-processing: Clean up answer
        if any(char.isdigit() for char in answer):
            return answer
        else:
            return "Not mentioned"

    except Exception as e:
        print(f"[HF RoBERTa Error] {e}")
        return "Error"
