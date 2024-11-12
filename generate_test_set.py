"""
generate_test_set.py
Goal: Generate and store test questions using OpenAI API
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
CATEGORIES = [
    "direct_questions",
    "scenario_based",
    "edge_cases",
    "multi_part",
    "location_specific"
]

# System Instructions
SYSTEM_PROMPTS = {
    "direct_questions": """You are a Toronto public safety expert. Generate simple, direct questions about Toronto safety procedures, emergency contacts, and basic safety information. Make sure your information is relevant for the general public.""",
    
    "scenario_based": """You are a Toronto public safety expert. Generate realistic scenario-based questions about safety situations that could occur in Toronto, that could affect.""",
    
    "edge_cases": """You are a Toronto public safety expert. Generate questions about complex or unusual safety situations that could occur in Toronto.""",
    
    "multi_part": """You are a Toronto public safety expert. Generate complex questions that involve multiple aspects of Toronto safety.""",
    
    "location_specific": """You are a Toronto public safety expert. Generate questions about safety in specific Toronto locations."""
}

def generate_questions(category: str, num_questions: int = 5) -> List[Dict]:
    """Generate questions for a specific category using OpenAI"""
    
    chat = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = f"""Generate {num_questions} safety-related questions for Toronto residents.
    Make sure your questions are relevant for the general public.
    
    For each question, also provide:
    1. Expected key information that should be in the answer
    
    2. Relevant context that would be needed
    3. Complexity level (simple, medium, or complex)
    
    Format each question as a JSON object with these fields:
    - question: the actual question
    - expected_elements: list of key points that should be in the answer
    - expected_contexts: list of relevant context topics
    - complexity: complexity level
    
    The questions should be {category} type questions.
    
    Return the response as a JSON array of these objects."""
    
    response = chat.invoke([
        {"role": "system", "content": SYSTEM_PROMPTS[category]},
        {"role": "user", "content": prompt}
    ])
    
    try:
        questions = json.loads(response.content)
        for q in questions:
            q["category"] = category
            q["generated_at"] = datetime.now().isoformat()
    except json.JSONDecodeError:
        print(f"Error parsing response for {category}")
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
            "model": "gpt-4o",
            "total_questions": len(questions)
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

def generate_full_test_set(questions_per_category: int = 5):
    """Generate a complete test set with all categories"""
    all_questions = []
    
    for category in CATEGORIES:
        print(f"Generating {category} questions...")
        questions = generate_questions(category, questions_per_category)
        all_questions.extend(questions)
        print(f"Generated {len(questions)} questions for {category}")
    
    save_test_set(all_questions)
    return all_questions

if __name__ == "__main__":
    print("Generating comprehensive test set...")
    questions = generate_full_test_set(questions_per_category=5)
    print(f"Generated total of {len(questions)} questions") 