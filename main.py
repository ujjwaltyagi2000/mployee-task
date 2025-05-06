from models import spacy_model, hf_roberta, groq_llama3
import pandas as pd

# Load JD dataset and take a sample of 20
file_path = "data/jd_data.xlsx"
df = pd.read_excel(file_path)
sample_df = df.sample(20, random_state=42).copy()

# Load spaCy model
nlp = spacy_model.load_model()

# Apply extraction functions
sample_df["spacy_experience"] = sample_df["JD_Text"].apply(
    lambda x: spacy_model.extract_experience_years(str(x), nlp)
)

sample_df["hf_roberta_experience"] = sample_df["JD_Text"].apply(
    lambda x: hf_roberta.extract_experience_hf_roberta(str(x))
)

sample_df["groq_experience"] = sample_df["JD_Text"].apply(
    lambda x: groq_llama3.extract_experience_llama3(str(x))
)

# Save output
output_path = "sample_outputs/sample_model_comparison.xlsx"
sample_df.to_excel(output_path, index=False)

print(f"Extraction complete. Results saved to {output_path}")
