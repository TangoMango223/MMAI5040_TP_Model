"""
evals_LLM_Output.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics.
Make sure to use RAGAS Version 0.2.5 to run this code.

Last Updated: 2024-11-15
"""

# -------------------------------

# Force installation of Ragas Version 0.2.5
# pip install ragas==0.2.5
# Otherwise, code will not run.

import sys
import subprocess

def install_new_ragas():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ragas==0.2.5"])

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
from MMAI5040_TP_Model.prompt_engineering_main.old_main import generate_safety_plan
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
    """Run RAGAS evaluation on the RAG pipeline"""
    print("Starting RAG evaluation...")
    
    # Load test cases
    test_cases = load_test_set()
    print(f"Testing {len(test_cases)} questions...")
    
    # Prepare results in RAGAS format
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": [],
        "reference": []
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing question {i}/{len(test_cases)}")
        try:
            # Extract structured input from metadata
            structured_input = test_case["metadata"]["structured_input"]
            
            # Generate answer using the safety plan generator
            result = generate_safety_plan(
                neighbourhood=structured_input["neighbourhood"],
                crime_type=structured_input["crime_type"],
                user_context=structured_input["user_context"]
            )
            
            # Format for RAGAS evaluation
            results["question"].append(test_case["question"])
            results["answer"].append(result)
            results["contexts"].append(test_case["ground_truth_context"])
            results["ground_truths"].append(test_case["ground_truth"])
            results["reference"].append(" ".join(test_case["ground_truth_context"]))
            
            # Debug print
            print("\nSample data for question", i)
            print("Question:", test_case["question"][:200])
            print("\nAnswer preview:", result[:200])
            print("\nContext sample:", test_case["ground_truth_context"][0][:200] if test_case["ground_truth_context"] else "No context")
            print("\nGround truth sample:", test_case["ground_truth"][0][:200] if test_case["ground_truth"] else "No ground truth")
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
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
        print("Dataset structure:", eval_data)
        raise

def analyze_rag_quality(results_df):
    """Analyze RAG quality and provide recommendations"""
    print("\nRAG Quality Analysis:")
    print("===================")
    
    try:
        if isinstance(results_df, pd.Series):
            results_df = results_df.to_frame().T
            
        metric_columns = [
            'faithfulness',
            'answer_relevancy'
        ]
        
        print("Detailed Metrics Analysis:")
        print("------------------------")
        for metric in metric_columns:
            if metric in results_df.columns:
                score = float(results_df[metric].iloc[0])
                print(f"\n{metric}:")
                print(f"Score: {score:.3f}")
                
                if score > 0.9:
                    print("⚠️ Warning: Score might be suspiciously high")
                    print("Consider:")
                    print("- Generating more diverse test cases")
                    print("- Adding edge cases to the test set")
                    print("- Including more complex scenarios")
                
                if score < 0.7:
                    if metric == 'faithfulness':
                        print("Suggestion: Consider adjusting prompt to stay closer to context")
                    elif metric == 'answer_relevancy':
                        print("Suggestion: Refine prompt to focus more on question")
            else:
                print(f"{metric}: Not available")
        
        print("\nOverall Assessment:")
        print("------------------")
        avg_score = results_df[metric_columns].mean().mean()
        print(f"Average Score: {avg_score:.3f}")
        if avg_score > 0.9:
            print("\nNote: High scores across all metrics suggest we should:")
            print("1. Increase test set size and variety")
            print("2. Include more challenging scenarios")
            print("3. Add edge cases to better stress-test the system")
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print("Results structure:", results_df)
        print("Results type:", type(results_df))

if __name__ == "__main__":
    print("Starting Comprehensive Evaluation Pipeline")
    print("=======================================")
    
    # Evaluate RAG pipeline
    print("\nEvaluating RAG Pipeline...")
    rag_results = run_rag_evaluation()
    analyze_rag_quality(rag_results)
    
    # Log basic results
    print("\nEvaluation Results:")
    print(rag_results)
    
    print("\nEvaluation complete!")