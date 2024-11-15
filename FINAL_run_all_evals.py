"""
run_all_evals.py
Goal: Run all evaluations in sequence.
Last Updated: 2024-11-15
"""

# --------------------------------
# Import Statements:
# Run evals_precision_recall_NEW.py, using python:
import subprocess
import pandas as pd


# Generate timestamp:
from datetime import datetime
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --------------------------------

# Run the script as a separate process

# # Run this first tocreate the generate answers json file
# subprocess.run(["python", "evals_LLMOutput_V3.py"])

# print("--------------------------------")
# print("--------------------------------")

# # Run LLM_Output_NEW.py:
# subprocess.run(["python", "evals_precision_recall_V3.py"])

# --------------------------------

# Read first csv file:
rag_results = pd.read_csv('rag_results.csv')

# Read second csv file:
precision_recall_results = pd.read_csv('precision_recall_results.csv')

# --------------------------------

# Modify precision_recall_results for concat:
updated_precision_recall_results = precision_recall_results[["Context Precision", "Context Recall"]]

# update rag_results for concat:
updated_rag_results = rag_results.drop(columns=["reference"], inplace = True)

# Concat both files based on row, since it'll always be the same lenjgth and order:
new_df = pd.concat([rag_results, updated_precision_recall_results], axis=1)

# Export to CSV, with time_stamp
new_df.to_csv(f'evaluation_results_{time_stamp}.csv', index=False)
# new_df.to_csv('evaluation_results.csv', index=False)
