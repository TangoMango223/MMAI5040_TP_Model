"""
main_evals.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics
"""

from typing import List, Dict
import pandas as pd
from datasets import Dataset

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Other imports
from main import generate_safety_plan
from rag_tracker import RAGTracker
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv(".env", override=True)

def load_test_set(filename: str = "latest_test_set.json"):
    """Load test questions from JSON file"""
    test_sets_dir = Path("test_sets")
    if not filename:
        # Get the most recent test set
        test_sets = list(test_sets_dir.glob("test_set_*.json"))
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
        "ground_truths": []
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing question {i}/{len(test_cases)}")
        try:
            # Get the question text - extract just the actual question part
            full_question = test_case["question"]
            
            # Parse location and concerns for generate_safety_plan
            location_start = full_question.find("LOCATION:") + 9
            concerns_start = full_question.find("SAFETY CONCERNS:") + 16
            context_start = full_question.find("ADDITIONAL USER CONTEXT:")
            
            location = full_question[location_start:concerns_start].strip()
            concerns = full_question[concerns_start:context_start].strip().split('\n')
            concerns = [c.strip().strip('123.') for c in concerns if c.strip()]
            
            # Create a simpler question format
            actual_question = f"Please provide a safety plan for {location} addressing these concerns: {', '.join(concerns)}"
            
            # Generate answer
            result = generate_safety_plan(
                neighbourhood=location,
                crime_concerns=concerns,
                user_context={"full_context": full_question}
            )
            
            # Format the answer
            if isinstance(result, dict):
                answer = str(result.get('answer', result))
            else:
                answer = str(result)
            
            # Get contexts and ground truth
            contexts = test_case.get("ground_truth_context", [])
            if isinstance(contexts, str):
                contexts = [contexts]
            if not contexts:
                contexts = ["No context available"]
                
            # Clean up contexts - remove any HTML or special formatting
            contexts = [c.replace('\n', ' ').strip() for c in contexts]
            contexts = [c for c in contexts if c and not c.startswith('[![') and not c.startswith('###')]
            
            ground_truths = test_case.get("ground_truth", [""])
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]
            
            # Debug print
            print("\nSample data for question", i)
            print("Question:", actual_question)
            print("\nAnswer preview:", answer[:200])
            print("\nContext sample:", contexts[0][:200] if contexts else "No context")
            print("\nGround truth sample:", ground_truths[0][:200] if ground_truths else "No ground truth")
            print("\nNumber of contexts:", len(contexts))
            print("Number of ground truths:", len(ground_truths))
            
            # Append to results
            results["question"].append(actual_question)
            results["answer"].append(answer)
            results["contexts"].append(contexts)
            results["ground_truths"].append(ground_truths)
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    print("\nCreating dataset...")
    # Create dataset directly from dictionary
    eval_data = Dataset.from_dict(results)
    
    print("\nSample evaluation data:")
    print("First row:")
    for key, value in results.items():
        if value:
            print(f"\n{key}:")
            if isinstance(value[0], list):
                print(f"First item (list of {len(value[0])} items):", value[0][0][:200])
            else:
                print("First item:", value[0][:200])
    
    print("\nRunning RAGAS evaluation...")
    try:
        # Run RAGAS evaluation
        scores = evaluate(
            eval_data,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
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
            'answer_relevancy',
            'context_precision',
            'context_recall'
        ]
        
        print("Detailed Metrics Analysis:")
        print("------------------------")
        for metric in metric_columns:
            if metric in results_df.columns:
                score = float(results_df[metric].iloc[0])
                print(f"\n{metric}:")
                print(f"Score: {score:.3f}")
                
                # Add interpretation
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
                    elif metric == 'context_precision':
                        print("Suggestion: Adjust retrieval strategy or expand knowledge base")
                    elif metric == 'context_recall':
                        print("Suggestion: Modify prompt to better utilize context")
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
    
    # Track evaluation with RAGTracker
    tracker = RAGTracker("toronto_safety_rag")
    
    # Evaluate RAG pipeline
    print("\nEvaluating RAG Pipeline...")
    rag_results = run_rag_evaluation()
    analyze_rag_quality(rag_results)
    
    # Log results with tracker
    config = {
        'embedding_model': "text-embedding-3-large",
        'retriever_k': 5,
        'model_name': "gpt-4",
        'changes_made': "Comprehensive evaluation",
    }
    
    tracker.log_experiment(
        rag_results,
        config,
        notes="Updated evaluation pipeline"
    )
    
    # Plot metrics trends
    tracker.plot_metrics_over_time()
    
    print("\nEvaluation complete!")
    