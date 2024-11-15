"""
evals_LLM_Output_NEW.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics.
Make sure to use RAGAS Version 0.2.5 to run this code.

Exports results to CSV file.

Last Updated: 2024-11-15
"""

# -------------------------------

# Force installation of Ragas Version 0.2.5
# pip install ragas==0.2.5
# Otherwise, code will not run.

import sys
import subprocess

def install_new_ragas():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ragas==0.2.5"],
        stdout=subprocess.DEVNULL,  # Suppresses standard output
        stderr=subprocess.DEVNULL   # Suppresses error messages
    )

install_new_ragas()

# -------------------------------

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    # RAG evaluation metrics
    faithfulness,    # Does answer stick to retrieved context? - LLM output evaluation
    answer_relevancy,  # Is answer addressing the question? - LLM output evaluation
)

# Import datasets for RAGAS evaluation
from datasets import Dataset
import pandas as pd

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Other imports
from main import generate_safety_plan
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

def run_rag_evaluation():
    """Run RAGAS evaluation on the RAG pipeline and save generated answers"""
    print("Starting LLM Output evaluation...")
    
    # Load test cases
    test_cases = load_test_set()
    
    # Prepare results in RAGAS format
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": [],
        "reference": []
    }
    
    # Store generated answers for reuse
    generated_answers = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing question {i}/{len(test_cases)}")
        try:
            structured_input = test_case["metadata"]["structured_input"]
            
            # Generate answer using the safety plan generator
            result = generate_safety_plan(
                neighbourhood=structured_input["neighbourhood"],
                crime_type=structured_input["crime_type"],
                user_context=structured_input["user_context"]
            )
            
            # Store the generated answer with question as key
            generated_answers[test_case["question"]] = result
            
            # Format for RAGAS evaluation
            results["question"].append(test_case["question"])
            results["answer"].append(result)
            results["contexts"].append(test_case["ground_truth_context"])
            results["ground_truths"].append(test_case["ground_truth"])
            results["reference"].append(" ".join(test_case["ground_truth_context"]))
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    # Save generated answers to file
    with open('generated_answers.json', 'w') as f:
        json.dump(generated_answers, f, indent=2)
    
    print("\nCreating dataset...")
    eval_data = Dataset.from_dict(results)
    
    print("\nRunning RAGAS evaluation...")
    try:
        scores = evaluate(
            eval_data,
            metrics=[
                faithfulness,
                answer_relevancy
            ]
        )
        return scores.to_pandas()
    
    except Exception as e:
        print(f"Error during RAGAS evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Comprehensive Evaluation Pipeline")
    print("=======================================")
    
    # Evaluate RAG pipeline
    print("\nEvaluating RAG Pipeline...")
    rag_results = run_rag_evaluation()
    
    # Export results to CSV:
    rag_results.to_csv('rag_results.csv', index=False)
    
    print("\nEvaluation complete!")