# Experience Extraction from Job Descriptions

## Objective

This project focuses on extracting the **Total Years of Experience** required from each Job Description (JD) in a dataset of 3000 records. The aim is to:

1. Research and implement 5–6 models/techniques for experience extraction.
2. Execute all models within a unified script for side-by-side comparison.
3. Optimize performance using parallel processing.
4. Save the outputs to an Excel sheet.
5. Recommend the best model based on consistency, accuracy, and efficiency.

## Models Implemented

- **SpaCy Rule-based Model**
- **HuggingFace Roberta (transformers)**
- **HuggingFace Tiny Roberta**
- **HuggingFace DistilBERT**
- **Groq API - LLaMA3-8B (OpenAI-compatible API)**

## Project Structure

```
├── data/
│   └── jd_data.xlsx                # Input JD file (3000 rows)
├── outputs/
│   └── result.xlsx                # Final combined results (if completed)
├── batch_outputs/                 # Intermediate chunk results for safety
├── notebook/                      # Notebook for analysis
├── models/
│   ├── __init__.py
│   ├── spacy_model.py
│   ├── hf_roberta.py
│   ├── hf_tiny_roberta.py
│   ├── hf_distilbert.py
│   └── groq_llama3.py
├── main.py                        # Final execution script with multiprocessing
├── sample_test.py                 # Test run on a batch of 20 random rows
├── generate_output_excel.py       # Run this to merge chunk Excel files
├── requirements.txt               # Python packages
├── environment.yml                # Optional: conda environment file
└── README.md                     # You are here
```

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/ujjwaltyagi2000/mployee-task.git
cd mployee-task
```

### Option 1: Using `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Option 2: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate jd-extraction
```

## Set Your Groq API Key

To use the **Groq API** for LLaMA3-based extraction, you need an API key:

1. Go to [https://groq.com](https://groq.com) and sign up or log in.
2. Generate your API key from the dashboard.
3. Save the key as an environment variable or in a `.env` file in the root directory.

**Option A (recommended): Create a `.env` file**

```
GROQ_API_KEY=your_api_key_here
```

**Option B: Export it manually (for temporary sessions)**

- On **Linux/macOS**:

  ```bash
  export GROQ_API_KEY=your_api_key_here
  ```

- On **Windows (CMD)**:

  ```cmd
  set GROQ_API_KEY=your_api_key_here
  ```

> ⚠️ Make sure your `.env` file is included in `.gitignore` to avoid committing your API key.

## Running the Scripts

### Run the Full Pipeline

```bash
python main.py
```

This will:

- Read `data/jd_data.xlsx`
- Split it into chunks
- Run 5 models in parallel for each chunk
- Save outputs to `batch_outputs/`
- You can later combine all chunk files using:

```bash
python generate_output_excel.py
```

Final result will be saved to `outputs/result.xlsx`.

### Run a Test Batch (20 Random Rows)

```bash
python sample_test.py
```

This script runs all models on a random batch of 20 rows from `jd_data.xlsx` for quick testing/debugging.

## Model Recommendation

Based on our evaluation:

- **LLaMA3 via Groq API** was the most balanced model, offering strong accuracy and speed
- **SpaCy** was the fastest but less semantically deep
- **Transformers (RoBERTa, DistilBERT)** were accurate but very slow

> For practical purposes, **LLaMA3** is recommended if API limits are not an issue. Else, SpaCy offers a fast, no-cost fallback.

## Notes

- If your API quota is exceeded, the Groq-based model will return fallback values.
- All intermediate chunk files are saved, so partial failures won't lose progress.

## Contact

For any queries or issues, please reach out to: **Ujjwal Tyagi**
