"""
generate_test_set_v2.py
Goal: Generate and store test questions using OpenAI API to align with both:
1. The safety plan generator's structured input format
2. RAGAS evaluation format requirements

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
    "personal_safety",      # Individual safety concerns
    # "transit_safety",       # Public transportation & commuting
    # "neighborhood_watch",   # Community safety and awareness
    # "emergency_prep"        # Emergency preparedness
]

def generate_test_case(scenario_type: str) -> Dict:
    """Generate a single test case that works with both RAGAS and the safety plan generator"""
    
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
    
    # Generate structured components first
    structure_prompt = f"""Generate a safety plan request with these specific components:
    1. A specific Toronto neighborhood name
    2. 2-3 crime types with severity (format: "Crime: Severity")
    3. Three Q&A pairs about user context

    Return in this exact format:
    NEIGHBOURHOOD: [name]
    CRIME_TYPES: ["Crime1: Severity", "Crime2: Severity"]
    CONTEXT: [
        "Q: [specific question]",
        "A: [specific answer]",
        "Q: [specific question]",
        "A: [specific answer]",
        "Q: [specific question]",
        "A: [specific answer]"
    ]"""
    
    structured_input = chat.invoke([{"role": "user", "content": structure_prompt}]).content
    
    # Format the structured input into the question format expected by the safety plan generator
    formatted_question = f"""
    LOCATION: {structured_input['neighbourhood']}
    
    SAFETY CONCERNS:
    - {', '.join(structured_input['crime_types'])}
    
    ADDITIONAL USER CONTEXT:
    {chr(10).join(structured_input['context'])}
    """
    
    # Generate ground truth safety plan using the same contexts
    ground_truth_prompt = f"""Create a comprehensive safety plan following this structure:

    1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT
    2. TARGETED SAFETY RECOMMENDATIONS
    3. PERSONAL SAFETY PROTOCOL
    4. PREVENTIVE MEASURES

    Base the plan on:
    USER REQUEST:
    {formatted_question}

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
    
    # Create test case in RAGAS format with structured metadata
    test_case = {
        "question": formatted_question.strip(),
        "ground_truth_context": contexts,
        "ground_truth": [ground_truth.strip()],
        "episode_done": False,
        # Store structured data in metadata for use with safety plan generator
        "metadata": {
            "structured_input": {
                "neighbourhood": structured_input['neighbourhood'],
                "crime_type": structured_input['crime_types'],
                "user_context": structured_input['context']
            },
            "question_type": scenario_type
        }
    }
    
    return test_case

def generate_test_set(cases_per_type: int = 2) -> List[Dict]:
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
        filename = f"test_set_v2_{timestamp}.json"
    
    test_sets_dir = Path("test_sets")
    test_sets_dir.mkdir(exist_ok=True)
    
    output_path = test_sets_dir / filename
    
    test_set = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "gpt-4",
            "total_cases": len(test_cases),
            "scenario_types": SCENARIO_TYPES,
            "version": "v2"
        },
        "questions": test_cases  # RAGAS expects "questions" key
    }
    
    with open(output_path, "w") as f:
        json.dump(test_set, f, indent=2)
    
    print(f"Test set saved to {output_path}")

if __name__ == "__main__":
    print("Generating test set v2...")
    # Generate fewer cases initially for testing
    test_cases = generate_test_set(cases_per_type=2)
    save_test_set(test_cases)
    print(f"Generated total of {len(test_cases)} test cases") 