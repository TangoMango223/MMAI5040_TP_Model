"""
evals_precision_recall_V3.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics

This script will focus on calculating context precision and context recall using RAGAS.
PrecisionSource: https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_precision.html
RecallSource: https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_recall.html
Uses Ragas Version 0.1.21, we will need to handle dependencies carefully.

Last Updated: 2024-11-15
"""

import sys
import subprocess
import pandas as pd
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load Environment Variables:
load_dotenv(".env", override=True)

#--------------------------------

def install_old_ragas():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ragas==0.1.21"],
        stdout=subprocess.DEVNULL,  # Suppresses standard output
        stderr=subprocess.DEVNULL   # Suppresses error messages
    )

install_old_ragas()

#--------------------------------

def load_test_set(filename: str = None):
    test_sets_dir = Path("test_sets")
    if not filename:
        test_sets = list(test_sets_dir.glob("test_set_v2_*.json"))
        if not test_sets:
            raise FileNotFoundError("No test sets found")
        latest_test_set = max(test_sets, key=lambda x: x.stat().st_mtime)
        filename = latest_test_set.name
    
    with open(test_sets_dir / filename) as f:
        test_set = json.load(f)
    return test_set["questions"]

def evaluate_context_metrics():
    test_cases = load_test_set()
    
    # Load previously generated answers from "generated_answers.json":
    try:
        with open('generated_answers.json', 'r') as f:
            generated_answers = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Please run evals_LLMOutput_V3.py first to generate answers")

    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    for test_case in test_cases:
        question = test_case["question"]
        # Use the stored LLM-generated answer instead of ground truth
        if question in generated_answers:
            data_samples['question'].append(question)
            data_samples['answer'].append(generated_answers[question])
            data_samples['contexts'].append(test_case["ground_truth_context"])
            data_samples['ground_truth'].append(" ".join(test_case["ground_truth"]))
        else:
            print(f"Warning: No generated answer found for question: {question[:100]}...")

    dataset = Dataset.from_dict(data_samples)

    precision_scores = evaluate(dataset, metrics=[context_precision])
    recall_scores = evaluate(dataset, metrics=[context_recall])

    results = pd.DataFrame({
        'Question': data_samples['question'],
        'Context': data_samples['contexts'],
        'Context Precision': precision_scores['context_precision'],
        'Context Recall': recall_scores['context_recall']
    })

    results.to_csv('precision_recall_results.csv', index=False)
    print("Precision and Recall results saved to precision_recall_results.csv")

if __name__ == "__main__":
    evaluate_context_metrics()