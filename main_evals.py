"""
main_evals.py
Goal: Evaluate both RAG pipeline and LLM output using RAGAS and custom metrics
"""

from typing import List, Dict
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from main import generate_safety_plan
from rag_tracker import RAGTracker
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_correctness

# Load environment variables
load_dotenv(".env", override=True)

# Specify

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Default evaluation questions
EVAL_QUESTIONS = [
    "What safety precautions should I take when walking alone at night in downtown Toronto?",
    # "How can I protect my home from break-ins while I'm away on vacation?",
    # "What should I do if I witness a crime in progress in Toronto?",
    # "How can I stay safe while using the TTC late at night?",
    # "What are the best practices for protecting myself from phone scams in Toronto?",
]

def run_rag_evaluation():
    """Run RAGAS evaluation on the RAG pipeline"""
    print("Starting RAG evaluation...")
    
    # Load test cases
    test_cases = load_test_set()
    print(f"Testing {len(test_cases)} questions...")
    
    # Initialize OpenAI components with wrappers
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # Use ada-002 as shown in example
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    openai_llm = OpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Wrap them with RAGAS wrappers
    wrapped_embeddings = LangchainEmbeddingsWrapper(openai_embeddings)
    wrapped_llm = LangchainLLMWrapper(openai_llm)
    
    # Initialize vector store for retrieval
    vectorstore = Pinecone.from_existing_index(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=openai_embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Prepare results in RAGAS format
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"Processing question {i}/{len(test_cases)}")
        try:
            # Get the question text
            question = test_case["question"]
            
            # Get the ground truth and contexts
            ground_truths = test_case.get("ground_truths", [""])
            contexts = test_case.get("contexts", [])
            
            # Generate new answer using the safety plan function
            result = generate_safety_plan(
                neighbourhood="Toronto",  # Default value
                crime_concerns=["General Safety"],  # Default value
                user_context={"Question": question}  # Pass the question as context
            )
            
            results.append({
                "question": question,
                "answer": result,
                "contexts": contexts,
                "ground_truths": ground_truths
            })
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    # Convert to RAGAS dataset format
    eval_data = Dataset.from_pandas(pd.DataFrame(results))
    
    print("\nRunning RAGAS evaluation...")
    # Run RAGAS evaluation with specified LLM and embeddings
    result = evaluate(
        eval_data,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )
    
    return result.to_pandas()

def analyze_rag_quality(results_df):
    """Analyze RAG quality and provide recommendations"""
    print("\nRAG Quality Analysis:")
    print("===================")
    
    for metric in results_df.columns:
        score = results_df[metric].mean()
        print(f"{metric}: {score:.3f}")
        
        if score < 0.7:
            if metric == 'faithfulness':
                print("Suggestion: Consider adjusting prompt to stay closer to context")
            elif metric == 'answer_relevancy':
                print("Suggestion: Refine prompt to focus more on question")
            elif metric == 'context_precision':
                print("Suggestion: Adjust retrieval strategy or expand knowledge base")
            elif metric == 'context_recall':
                print("Suggestion: Modify prompt to better utilize context")

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

if __name__ == "__main__":
    print("Starting Comprehensive Evaluation Pipeline")
    print("=======================================")
    
    # Track evaluation with RAGTracker
    tracker = RAGTracker("toronto_safety_rag")
    
    # Try to load test questions, fall back to defaults if not found
    try:
        test_questions = load_test_set()
        EVAL_QUESTIONS = [q["question"] for q in test_questions]
    except FileNotFoundError:
        print("No test set found, using default questions")
    
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
    