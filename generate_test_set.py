"""
generate_test_set.py
Goal: Generate and store test questions using OpenAI API to align with the safety plan generator's input format.
The test set will be used to evaluate our LLM + RAG model for the Toronto Police Project.
"""

import json
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict

# Load environment variables
load_dotenv(".env", override=True)

# Update the categories to focus on safety plan scenarios
SCENARIO_TYPES = [
    "property_theft",
    "personal_safety",      # Individual safety concerns",
    "transit_safety",       # Public transportation & commuting
    "neighborhood_watch",   # Community safety and awareness
    "emergency_prep"        # Emergency preparedness
]

def generate_test_case(scenario_type: str) -> Dict:
    """Generate a single test case in RAGAS format"""
    
    # Initialize OpenAI with higher temperature for more variety
    chat = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Initialize Pinecone components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    
    # First, generate a basic query to retrieve relevant contexts
    base_query_prompt = f"""Generate a complex safety-related query for Toronto focusing on {scenario_type}.
    Include:
    - A specific Toronto neighborhood
    - 2-3 interconnected safety concerns
    - Make the scenario challenging and realistic
    - Include specific details that would require precise recommendations"""
    
    base_query = chat.invoke([{"role": "user", "content": base_query_prompt}]).content
    
    # Retrieve relevant contexts using the base query
    retrieved_docs = vectorstore.similarity_search(
        base_query,
        k=5
    )
    
    # Extract contexts
    contexts = [doc.page_content for doc in retrieved_docs]
    
    # Generate the user question/request
    question_prompt = f"""Generate a safety plan request for Toronto with this format:

    LOCATION: [specific Toronto neighborhood]
    
    SAFETY CONCERNS: [2-3 specific concerns]
    
    ADDITIONAL USER CONTEXT:
    Q: How often do you [relevant activity]?
    A: [contextual response]
    
    Q: [relevant follow-up question]?
    A: [contextual response]
    
    Q: [specific safety measure question]?
    A: [contextual response]

    Focus on {scenario_type} and use these verified Toronto safety resources:
    {chr(10).join(f'- {context}' for context in contexts)}"""
    
    question = chat.invoke([{"role": "user", "content": question_prompt}]).content
    
    # Generate ground truth safety plan using the same contexts
    ground_truth_prompt = f"""Create a comprehensive safety plan following this structure:

    1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT
    2. TARGETED SAFETY RECOMMENDATIONS
    3. PERSONAL SAFETY PROTOCOL
    4. PREVENTIVE MEASURES

    Base the plan on:
    USER REQUEST:
    {question}

    VERIFIED RESOURCES:
    {chr(10).join(f'- {context}' for context in contexts)}

    Guidelines for your response:
    - Provide specific, actionable advice that can be implemented immediately
    - Include both preventive measures and emergency response protocols
    - Reference relevant Toronto Police Service programs or initiatives when applicable
    - Maintain a supportive and empowering tone while being clear about risks
    - Prioritize recommendations based on the specific crime patterns
    - Include relevant contact numbers and resources"""
    
    ground_truth = chat.invoke([{"role": "user", "content": ground_truth_prompt}]).content
    
    # Format in RAGAS structure
    test_case = {
        "question": question.strip(),
        "ground_truth_context": contexts,
        "ground_truth": [ground_truth.strip()],  # RAGAS expects a list
        "question_type": scenario_type,
        "episode_done": False
    }
    
    return test_case

def generate_test_set(cases_per_type: int = 5) -> List[Dict]:
    """Generate a complete test set with all scenario types"""
    test_cases = []
    
    for scenario_type in SCENARIO_TYPES:
        print(f"Generating {scenario_type} scenarios...")
        for _ in range(cases_per_type):
            test_case = generate_test_case(scenario_type)
            if test_case:
                test_cases.append(test_case)
        print(f"Generated {cases_per_type} cases for {scenario_type}")
    
    return test_cases

def save_test_set(test_cases: List[Dict], filename: str = None):
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
            "total_cases": len(test_cases),
            "scenario_types": SCENARIO_TYPES
        },
        "questions": test_cases  # RAGAS expects "questions" key
    }
    
    with open(output_path, "w") as f:
        json.dump(test_set, f, indent=2)
    
    # Also save as latest
    with open(test_sets_dir / "latest_test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)
    
    print(f"Test set saved to {output_path}")
    print(f"Also saved as latest_test_set.json")

if __name__ == "__main__":
    print("Generating comprehensive test set...")
    test_cases = generate_test_set(cases_per_type=5)
    save_test_set(test_cases)
    print(f"Generated total of {len(test_cases)} test cases")