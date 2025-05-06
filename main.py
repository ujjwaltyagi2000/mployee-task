from models import spacy_model

# other models will be imported here
import pandas as pd

# Load JD dataset
file_path = "data/jd_data.xlsx"
df = pd.read_excel(file_path)

# Load spaCy model
nlp = spacy_model.load_model()

# Apply extraction
df["spacy_experience"] = df["JD_Text"].apply(
    lambda x: spacy_model.extract_experience_years(str(x), nlp)
)

# Save output
df.to_excel("outputs/spacy_model_results.xlsx", index=False)

print("Extraction complete. Results saved to outputs/spacy_model_results.xlsx")
