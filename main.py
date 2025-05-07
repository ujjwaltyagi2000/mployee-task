# script to run models in parallel

import os
import pandas as pd
from multiprocessing import Process, Manager
from models import spacy_model, hf_roberta, groq_llama3, hf_tiny_roberta, hf_distilbert
from models.model_loader import check_model_loading


def chunk_dataframe(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size].copy()


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


def run_hf_distilbert_extraction(data, output_dict):
    output_dict["hf_distilbert_experience"] = data["JD_Text"].apply(
        lambda x: hf_distilbert.extract_experience_hf_distilbert(str(x))
    )


if __name__ == "__main__":
    # Initial model check (optional)
    check_model_loading()

    # Load entire dataset
    file_path = "data/jd_data.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel(file_path).iloc[1:].reset_index(drop=True)

    # Set chunk size
    chunk_size = 50

    # Output directory
    output_dir = "batch_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Process in chunks
    for i, chunk_df in enumerate(chunk_dataframe(df, chunk_size)):
        print(f"Processing chunk {i + 1}/{(len(df) // chunk_size) + 1}")

        manager = Manager()
        output_dict = manager.dict()

        processes = [
            Process(target=run_spacy_extraction, args=(chunk_df, output_dict)),
            Process(target=run_hf_roberta_extraction, args=(chunk_df, output_dict)),
            Process(target=run_groq_llama_extraction, args=(chunk_df, output_dict)),
            Process(
                target=run_hf_tiny_roberta_extraction, args=(chunk_df, output_dict)
            ),
            Process(target=run_hf_distilbert_extraction, args=(chunk_df, output_dict)),
        ]

        # Start processes
        for p in processes:
            p.start()

        # Wait for all to finish
        for p in processes:
            p.join()

        # Add extracted columns
        for key in output_dict.keys():
            chunk_df[key] = output_dict[key]

        # Save chunk output
        chunk_output_path = os.path.join(output_dir, f"chunk_{i+1}.xlsx")
        chunk_df.to_excel(chunk_output_path, index=False)

        print(f"Saved chunk {i + 1} to {chunk_output_path}")

    print(f"\n✅ All chunks processed. Outputs saved in '{output_dir}/'")
