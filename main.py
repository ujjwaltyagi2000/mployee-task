import pandas as pd
from multiprocessing import Process, Manager
from models import spacy_model, hf_roberta, groq_llama3, hf_tiny_roberta


def run_spacy_extraction(data, output_dict):
    nlp = spacy_model.load_model()
    output_dict["spacy_experience"] = data["JD_Text"].apply(
        lambda x: spacy_model.extract_experience_years(str(x), nlp)
    )


def run_hf_roberta_extraction(data, output_dict):
    output_dict["hf_roberta_experience"] = data["JD_Text"].apply(
        lambda x: hf_roberta.extract_experience_hf_roberta(str(x))
    )


def run_groq_llama_extraction(data, output_dict):
    output_dict["groq_experience"] = data["JD_Text"].apply(
        lambda x: groq_llama3.extract_experience_llama3(str(x))
    )


def run_hf_tiny_roberta_extraction(data, output_dict):
    output_dict["hf_tiny_roberta_experience"] = data["JD_Text"].apply(
        lambda x: hf_tiny_roberta.extract_experience_tiny_roberta(str(x))
    )


if __name__ == "__main__":
    # Load dataset
    file_path = "data/jd_data.xlsx"
    df = pd.read_excel(file_path)
    sample_df = df.sample(20, random_state=42).copy()

    # Shared dictionary for storing results
    manager = Manager()
    output_dict = manager.dict()

    # Define all processes
    processes = [
        Process(target=run_spacy_extraction, args=(sample_df, output_dict)),
        Process(target=run_hf_roberta_extraction, args=(sample_df, output_dict)),
        Process(target=run_groq_llama_extraction, args=(sample_df, output_dict)),
        Process(target=run_hf_tiny_roberta_extraction, args=(sample_df, output_dict)),
    ]

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all to finish
    for p in processes:
        p.join()

    # Add results back to sample_df
    for key in output_dict.keys():
        sample_df[key] = output_dict[key]

    # Save results
    output_path = "sample_outputs/sample_model_comparison_parallel.xlsx"
    sample_df.to_excel(output_path, index=False)
    print(f"Parallel extraction complete. Results saved to {output_path}")
