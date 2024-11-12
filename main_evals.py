"""
main_evals.py
Goal: Evaluate both RAG pipeline and embeddings using RAGAS and custom metrics
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main import generate_safety_plan
from rag_tracker import RAGTracker
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv(".env", override=True)

# Configure OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def evaluate_embeddings(questions: List[str]):
    """Evaluate embedding quality using cosine similarity"""
    print("\nEvaluating Embeddings...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Get embeddings for all questions
    embedded_questions = embeddings.embed_documents(questions)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embedded_questions)
    
    # Calculate metrics
    avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    max_similarity = np.max(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    min_similarity = np.min(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    
    return {
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity,
        'similarity_matrix': similarity_matrix,
        'questions': questions
    }

def run_rag_evaluation():
    """Run RAGAS evaluation on the RAG pipeline"""
    print("Starting RAG evaluation...")
    print(f"Testing {len(EVAL_QUESTIONS)} questions...")
    
    # Generate responses for all questions
    results = []
    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"Processing question {i}/{len(EVAL_QUESTIONS)}")
        try:
            result = generate_safety_plan(question, return_all=True)
            results.append({
                'question': question,
                'answer': result['answer'],
                'contexts': [doc.page_content for doc in result['contexts']],
                'ground_truths': [""]  # Required by RAGAS but can be empty
            })
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
    
    # Convert to RAGAS dataset format
    eval_data = Dataset.from_pandas(pd.DataFrame(results))
    
    print("\nRunning RAGAS evaluation...")
    # Run RAGAS evaluation with correct metric names
    result = evaluate(
        eval_data,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,  # Updated metric name
            context_recall,     # Updated metric name
        ]
    )
    
    return result.to_pandas()

def analyze_embedding_quality(embedding_results):
    """Analyze embedding quality and provide recommendations"""
    print("\nEmbedding Analysis:")
    print("==================")
    print(f"Average Similarity: {embedding_results['avg_similarity']:.3f}")
    print(f"Max Similarity: {embedding_results['max_similarity']:.3f}")
    print(f"Min Similarity: {embedding_results['min_similarity']:.3f}")
    
    # Find most and least similar pairs
    similarity_matrix = embedding_results['similarity_matrix']
    questions = embedding_results['questions']
    n = len(questions)
    most_similar_pair = None
    least_similar_pair = None
    max_sim = -1
    min_sim = 2
    
    for i in range(n):
        for j in range(i+1, n):
            sim = similarity_matrix[i][j]
            if sim > max_sim:
                max_sim = sim
                most_similar_pair = (questions[i], questions[j])
            if sim < min_sim:
                min_sim = sim
                least_similar_pair = (questions[i], questions[j])
    
    print("\nMost Similar Questions:")
    print(f"1. {most_similar_pair[0]}")
    print(f"2. {most_similar_pair[1]}")
    print(f"Similarity: {max_sim:.3f}")
    
    print("\nLeast Similar Questions:")
    print(f"1. {least_similar_pair[0]}")
    print(f"2. {least_similar_pair[1]}")
    print(f"Similarity: {min_sim:.3f}")
    
    return {
        'most_similar_pair': most_similar_pair,
        'least_similar_pair': least_similar_pair,
        'max_sim': max_sim,
        'min_sim': min_sim
    }

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
            elif metric == 'context_precision':  # Updated metric name
                print("Suggestion: Adjust retrieval strategy or expand knowledge base")
            elif metric == 'context_recall':     # Updated metric name
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
    
    # 1. Evaluate embeddings
    print("\nEvaluating Embeddings...")
    try:
        EVAL_QUESTIONS = [q["question"] for q in load_test_set()]
    except FileNotFoundError:
        print("No test set found, using default questions")
        EVAL_QUESTIONS = [
            "What safety precautions should I take when walking alone at night in downtown Toronto?",
            "How can I protect my home from break-ins while I'm away on vacation?",
            "What should I do if I witness a crime in progress in Toronto?",
            "How can I stay safe while using the TTC late at night?",
            "What are the best practices for protecting myself from phone scams in Toronto?",
        ]
    embedding_results = evaluate_embeddings(EVAL_QUESTIONS)
    embedding_analysis = analyze_embedding_quality(embedding_results)
    
    # 2. Evaluate RAG pipeline
    print("\nEvaluating RAG Pipeline...")
    rag_results = run_rag_evaluation()
    analyze_rag_quality(rag_results)
    
    # 3. Log results with tracker
    config = {
        'embedding_model': "text-embedding-3-large",
        'retriever_k': 10,
        'model_name': "gpt-4",
        'changes_made': "Comprehensive evaluation",
    }
    
    tracker.log_experiment(
        rag_results,
        config,
        notes=f"Embedding avg similarity: {embedding_results['avg_similarity']:.3f}"
    )
    
    # Plot metrics trends
    tracker.plot_metrics_over_time()
    
    print("\nEvaluation complete!")
    