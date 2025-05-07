# script to combine chunked outputs into a single file

import pandas as pd
import os

batch_files_path = "batch_outputs"

files = os.listdir(batch_files_path)

combined_df = pd.DataFrame()

for index in range(1, len(files) + 1):
    file_name = f"chunk_{index}.xlsx"
    file_path = os.path.join(batch_files_path, file_name)
    print("Reading file:", file_name)
    df = pd.read_excel(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_excel("outputs/result.xlsx", index=False)
