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

# --------------------------------


# # Run the script as a separate process
# subprocess.run(["python", "evals_precision_recall_NEW.py"])

# # Run LLM_Output_NEW.py:
# subprocess.run(["python", "evals_LLMOutput_NEW.py"])


# --------------------------------

# Open both csv output files:

# Read first csv file:
rag_results = pd.read_csv('rag_results.csv')

# Read second csv file:
precision_recall_results = pd.read_csv('precision_recall_results.csv')

# --------------------------------

# Modify precision_recall_results for concat:
updated_precision_recall_results = precision_recall_results[["Context Precision", "Context Recall"]]

# Concat both files based on row, since it'll always be the same lenjgth and order:
new_df = pd.concat([rag_results, updated_precision_recall_results], axis=1)

# Export to CSV:
new_df.to_csv('all_evals_results.csv', index=False)
