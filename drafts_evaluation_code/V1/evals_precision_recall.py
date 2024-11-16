"""
evals_precision_recall.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics

This script will focus on calculating context precision and context recall using RAGAS.
PrecisionSource: https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_precision.html
RecallSource: https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_recall.html
Uses Ragas Version 0.1.21, we will need to handle dependencies carefully.

Last Updated: 2024-11-15
"""

# Force installation of Ragas Version 0.1.21
# pip install ragas==0.1.21

import sys
import subprocess

# -------------------------------

def install_old_ragas():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ragas==0.1.21"])

install_old_ragas()

# -------------------------------

# RAGAS imports
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset
import json
from pathlib import Path

# Import datasets for RAGAS evaluation
from datasets import Dataset
import pandas as pd

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Other imports
from MMAI5040_TP_Model.main import generate_safety_plan
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv(".env", override=True)

def load_test_set(filename: str = None):
    """Load test questions from JSON file"""
    test_sets_dir = Path("test_sets")
    if not filename:
        # Get the most recent test set with v2 naming pattern
        test_sets = list(test_sets_dir.glob("test_set_v2_*.json"))
        if not test_sets:
            raise FileNotFoundError("No test sets found")
        latest_test_set = max(test_sets, key=lambda x: x.stat().st_mtime)
        filename = latest_test_set.name
    
    with open(test_sets_dir / filename) as f:
        test_set = json.load(f)
    return test_set["questions"]

def evaluate_context_precision():
    """Evaluate context precision using RAGAS context_precision metric"""
    print("Starting context precision evaluation...")

    # Load test cases
    test_cases = load_test_set()
    print(f"Testing {len(test_cases)} questions...")

    # Prepare data for evaluation
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    for test_case in test_cases:
        data_samples['question'].append(test_case["question"])
        data_samples['answer'].append(" ".join(test_case["ground_truth"]))
        data_samples['contexts'].append(test_case["ground_truth_context"])
        data_samples['ground_truth'].append(" ".join(test_case["ground_truth"]))

    dataset = Dataset.from_dict(data_samples)

    print("\nRunning context precision evaluation...")
    try:
        score = evaluate(dataset, metrics=[context_precision])
        print(score.to_pandas())
    except Exception as e:
        print(f"Error during context precision evaluation: {str(e)}")
        raise

def evaluate_context_recall():
    """Evaluate context recall using RAGAS context_recall metric"""
    print("Starting context recall evaluation...")

    # Load test cases
    test_cases = load_test_set()
    print(f"Testing {len(test_cases)} questions...")

    # Prepare data for evaluation
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    for test_case in test_cases:
        data_samples['question'].append(test_case["question"])
        data_samples['answer'].append(" ".join(test_case["ground_truth"]))
        data_samples['contexts'].append(test_case["ground_truth_context"])
        data_samples['ground_truth'].append(" ".join(test_case["ground_truth"]))

    dataset = Dataset.from_dict(data_samples)

    print("\nRunning context recall evaluation...")
    try:
        score = evaluate(dataset, metrics=[context_recall])
        print(score.to_pandas())
    except Exception as e:
        print(f"Error during context recall evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Context Precision Evaluation")
    print("=======================================")
    
    # Evaluate context precision
    evaluate_context_precision()
    
    print("\nStarting Context Recall Evaluation")
    print("=======================================")
    
    # Evaluate context recall
    evaluate_context_recall()
    
    print("\nEvaluation complete!")