from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODELS = {
    "hf_roberta": "deepset/roberta-base-squad2",
    "hf_tiny_roberta": "deepset/tinyroberta-squad2",
    "hf_distilbert": "distilbert/distilbert-base-cased-distilled-squad",
}


def check_model_loading():
    print("[Model Loading Check]")
    for name, path in MODELS.items():
        try:
            print(f"Loading: {name} ({path})")
            AutoTokenizer.from_pretrained(path, local_files_only=True)
            AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
            print(f"✅ {name} loaded successfully.\n")
        except Exception as e:
            print(f"❌ Failed to load {name}.\n   Reason: {e}\n")


if __name__ == "__main__":
    check_model_loading()
