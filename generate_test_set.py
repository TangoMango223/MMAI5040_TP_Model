"""
generate_test_set.py
Goal: Generate and store test questions using OpenAI API to align with RAGAS evaluation format.
The test set will be used as part of the evaluation of our LLM + RAG model for Toronto Police Project.

Note: The test set is generated using GPT-4o, which is a powerful model that can generate a wide variety of questions and answers. However, our team checked for accuracy (human evaluation) for this test set to reduce hallucinations.
"""

import json
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Dict

# Load environment variables
load_dotenv(".env", override=True)

# Types of questions to generate
QUESTION_TYPES = [
    "factual",          # Direct questions about safety procedures
    "procedural",       # Step-by-step safety instructions
    "conditional",      # "What if" or scenario-based questions
    "location_based",   # Questions about specific Toronto locations
    "emergency"         # Emergency response questions
]

# System Instructions
SYSTEM_PROMPT = """You are a Toronto public safety expert. 
Generate safety-related questions and their corresponding ground truth answers and contexts.

Each question should include:
1. The question itself
2. The ideal ground truth answer
3. The necessary context information that would be needed to answer this question accurately
4. The type of question (factual, procedural, conditional, location_based, or emergency)

Make sure all information is specific to Toronto and based on official safety guidelines and best practices.
Remember, the safety plan's target audience is the general public for the City of Toronto, so ensure all information is relevant and applicable to this audience.

Remember, stay within your role as a public safety expert for advisory purposes. Do not provide any legal, medical, or other professional advice. 
"""

def generate_questions(question_type: str, num_questions: int = 5) -> List[Dict]:
    """Generate questions with ground truth answers and contexts for a specific type"""
    
    chat = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = f"""Generate {num_questions} {question_type} safety-related questions for Toronto residents.
    
    For each question, provide:
    1. The question text
    2. A list of relevant context information needed to answer the question accurately
    3. The complete ground truth answer
    
    Format each as a JSON object with these fields:
    - question: the actual question
    - ground_truth_context: list of relevant context information (each item should be a complete, informative sentence)
    - ground_truth: list containing the complete, accurate answer
    - question_type: "{question_type}"
    - episode_done: false
    
    The ground truth answer should be comprehensive and based on the context information.
    The context should contain all information needed to construct the answer.
    
    Return the response as a JSON array of these objects."""
    
    response = chat.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    try:
        questions = json.loads(response.content)
        for q in questions:
            # Ensure ground_truth is a list
            if isinstance(q["ground_truth"], str):
                q["ground_truth"] = [q["ground_truth"]]
            # Ensure ground_truth_context is a list
            if isinstance(q["ground_truth_context"], str):
                q["ground_truth_context"] = [q["ground_truth_context"]]
            # Add generation timestamp
            q["generated_at"] = datetime.now().isoformat()
            # Ensure episode_done is false
            q["episode_done"] = False
    except json.JSONDecodeError:
        print(f"Error parsing response for {question_type}")
        questions = []
    
    return questions

def save_test_set(questions: List[Dict], filename: str = None):
    """Save the generated test set to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_set_{timestamp}.json"
    
    test_sets_dir = Path("test_sets")
    test_sets_dir.mkdir(exist_ok=True)
    
    output_path = test_sets_dir / filename
    
    test_set = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "gpt-4",
            "total_questions": len(questions),
            "question_types": QUESTION_TYPES
        },
        "questions": questions
    }
    
    with open(output_path, "w") as f:
        json.dump(test_set, f, indent=2)
    
    # Also save as latest
    with open(test_sets_dir / "latest_test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)
    
    print(f"Test set saved to {output_path}")
    print(f"Also saved as latest_test_set.json")

def generate_full_test_set(questions_per_type: int = 5):
    """Generate a complete test set with all question types"""
    all_questions = []
    
    for question_type in QUESTION_TYPES:
        print(f"Generating {question_type} questions...")
        questions = generate_questions(question_type, questions_per_type)
        all_questions.extend(questions)
        print(f"Generated {len(questions)} questions for {question_type}")
    
    save_test_set(all_questions)
    return all_questions

if __name__ == "__main__":
    print("Generating comprehensive test set...")
    questions = generate_full_test_set(questions_per_type=5)
    print(f"Generated total of {len(questions)} questions")