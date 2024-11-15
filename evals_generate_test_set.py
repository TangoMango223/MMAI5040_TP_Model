"""
generate_test_set_v2.py
Goal: Generate and store test questions using OpenAI API to align with both:
1. The safety plan generator's structured input format
2. RAGAS evaluation format requirements

The test set will be used to evaluate our LLM + RAG model for the Toronto Police Project.

Last Updated: 2024-11-15
"""

import json
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict
import random

# Load environment variables
load_dotenv(".env", override=True)


# Toronto Neighbourhoods:
neighbourhoods = [
    "Agincourt North (129)",
    "Agincourt South-Malvern West (128)",
    "Alderwood (20)",
    "Annex (95)",
    "Avondale (153)",
    "Banbury-Don Mills (42)",
    "Bathurst Manor (34)",
    "Bay-Cloverhill (169)",
    "Bayview Village (52)",
    "Bayview Woods-Steeles (49)",
    "Bedford Park-Nortown (39)",
    "Beechborough-Greenbrook (112)",
    "Bendale South (157)",
    "Bendale-Glen Andrew (156)",
    "Birchcliffe-Cliffside (122)",
    "Black Creek (24)",
    "Blake-Jones (69)",
    "Briar Hill-Belgravia (108)",
    "Bridle Path-Sunnybrook-York Mills (41)",
    "Broadview North (57)",
    "Brookhaven-Amesbury (30)",
    "Cabbagetown-South St.James Town (71)",
    "Caledonia-Fairbank (109)",
    "Casa Loma (96)",
    "Centennial Scarborough (133)",
    "Church-Wellesley (167)",
    "Clairlea-Birchmount (120)",
    "Clanton Park (33)",
    "Cliffcrest (123)",
    "Corso Italia-Davenport (92)",
    "Danforth (66)",
    "Danforth East York (59)",
    "Don Valley Village (47)",
    "Dorset Park (126)",
    "Dovercourt Village (172)",
    "Downsview (155)",
    "Downtown Yonge East (168)",
    "Dufferin Grove (83)",
    "East End-Danforth (62)",
    "East L'Amoreaux (148)",
    "East Willowdale (152)",
    "Edenbridge-Humber Valley (9)",
    "Eglinton East (138)",
    "Elms-Old Rexdale (5)",
    "Englemount-Lawrence (32)",
    "Eringate-Centennial-West Deane (11)",
    "Etobicoke City Centre (159)",
    "Etobicoke West Mall (13)",
    "Fenside-Parkwoods (150)",
    "Flemingdon Park (44)",
    "Forest Hill North (102)",
    "Forest Hill South (101)",
    "Fort York-Liberty Village (163)",
    "Glenfield-Jane Heights (25)",
    "Golfdale-Cedarbrae-Woburn (141)",
    "Greenwood-Coxwell (65)",
    "Guildwood (140)",
    "Harbourfront-CityPlace (165)",
    "Henry Farm (53)",
    "High Park North (88)",
    "High Park-Swansea (87)",
    "Highland Creek (134)",
    "Hillcrest Village (48)",
    "Humber Bay Shores (161)",
    "Humber Heights-Westmount (8)",
    "Humber Summit (21)",
    "Humbermede (22)",
    "Humewood-Cedarvale (106)",
    "Ionview (125)",
    "Islington (158)",
    "Junction Area (90)",
    "Junction-Wallace Emerson (171)",
    "Keelesdale-Eglinton West (110)",
    "Kennedy Park (124)",
    "Kensington-Chinatown (78)",
    "Kingsview Village-The Westway (6)",
    "Kingsway South (15)",
    "Lambton Baby Point (114)",
    "L'Amoreaux West (147)",
    "Lansing-Westgate (38)",
    "Lawrence Park North (105)",
    "Lawrence Park South (103)",
    "Leaside-Bennington (56)",
    "Little Portugal (84)",
    "Long Branch (19)",
    "Malvern East (146)",
    "Malvern West (145)",
    "Maple Leaf (29)",
    "Markland Wood (12)",
    "Milliken (130)",
    "Mimico-Queensway (160)",
    "Morningside (135)",
    "Morningside Heights (144)",
    "Moss Park (73)",
    "Mount Dennis (115)",
    "Mount Olive-Silverstone-Jamestown (2)",
    "Mount Pleasant East (99)",
    "New Toronto (18)",
    "Newtonbrook East (50)",
    "Newtonbrook West (36)",
    "North Riverdale (68)",
    "North St.James Town (74)",
    "North Toronto (173)",
    "Oakdale-Beverley Heights (154)",
    "Oakridge (121)",
    "Oakwood Village (107)",
    "O'Connor-Parkview (54)",
    "Old East York (58)",
    "Palmerston-Little Italy (80)",
    "Parkwoods-O'Connor Hills (149)",
    "Pelmo Park-Humberlea (23)",
    "Playter Estates-Danforth (67)",
    "Pleasant View (46)",
    "Princess-Rosethorn (10)",
    "Regent Park (72)",
    "Rexdale-Kipling (4)",
    "Rockcliffe-Smythe (111)",
    "Roncesvalles (86)",
    "Rosedale-Moore Park (98)",
    "Runnymede-Bloor West Village (89)",
    "Rustic (28)",
    "Scarborough Village (139)",
    "South Eglinton-Davisville (174)",
    "South Parkdale (85)",
    "South Riverdale (70)",
    "St.Andrew-Windfields (40)",
    "Steeles (116)",
    "Stonegate-Queensway (16)",
    "Tam O'Shanter-Sullivan (118)",
    "Taylor-Massey (61)",
    "The Beaches (63)",
    "Thistletown-Beaumond Heights (3)",
    "Thorncliffe Park (55)",
    "Trinity-Bellwoods (81)",
    "University (79)",
    "Victoria Village (43)",
    "Wellington Place (164)",
    "West Hill (136)",
    "West Humber-Clairville (1)",
    "West Queen West (162)",
    "West Rouge (143)",
    "Westminster-Branson (35)",
    "Weston (113)",
    "Weston-Pelham Park (91)",
    "Wexford/Maryvale (119)",
    "Willowdale West (37)",
    "Willowridge-Martingrove-Richview (7)",
    "Woburn North (142)",
    "Woodbine Corridor (64)",
    "Woodbine-Lumsden (60)",
    "Wychwood (94)",
    "Yonge-Bay Corridor (170)",
    "Yonge-Doris (151)",
    "Yonge-Eglinton (100)",
    "Yonge-St.Clair (97)",
    "York University Heights (27)",
    "Yorkdale-Glen Park (31)"
]

# Update the categories to focus on safety plan scenarios
SCENARIO_TYPES = [
    "residential_safety",      # Home security, break-ins, neighborhood safety
    "vehicle_security",        # Auto theft, parking safety, carjacking prevention
    "transit_safety",        # Public transit, walking to/from stations, commuting
    "personal_public_safety",         # Individual safety in public spaces, robbery prevention
    "night_safety",           # Evening/night-specific concerns, dark hours safety
] 

# Add this constant after SCENARIO_TYPES
RISK_LEVELS = ["low", "medium", "high"]

def generate_test_case(scenario_type: str) -> Dict:
    """Generate a single test case that works with both RAGAS and the safety plan generator"""
    
    # Initialize OpenAI with higher temperature for more variety
    chat = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Initialize Pinecone components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    
    # First, generate a basic query to retrieve relevant contexts
    # We will have one LLM agent make the questions.
    base_query_prompt = f"""Generate a complex safety-related query for the city of Toronto focusing on {scenario_type}.
    Include:
    - A specific Toronto neighborhood - pick from the 158 Divisions Neighbourhoods in Toronto.
    - Pick between 1-4 crime types: Assault, Auto Theft, Break and Enter, Robbery
    - Make the scenario challenging and realistic to the general public in the city of Toronto
    - Include specific details that would require precise recommendations
    
    """
    base_query = chat.invoke([{"role": "user", "content": base_query_prompt}]).content
    
    # Retrieve relevant contexts using the base query
    retrieved_docs = vectorstore.similarity_search(
        base_query,
        k=5
    )
    
    # Extract contexts
    contexts = [doc.page_content for doc in retrieved_docs]
    
    # Randomly select 1-4 crime types and assign random risk levels
    selected_crimes = random.sample(["Assault", "Auto Theft", "Break and Enter", "Robbery"], k=random.randint(1, 4))
    crime_types_with_risk = [f"{crime}: {random.choice(RISK_LEVELS)}" for crime in selected_crimes]
    
    # Randomly select a neighborhood
    selected_neighborhood = random.choice(neighbourhoods)
    
    structure_prompt = f"""Generate a safety plan request with these specific components:
    1. Use this specific Toronto neighborhood: {selected_neighborhood}
    2. Use these specific crime types with their severity levels: {crime_types_with_risk}
    3. Three Q&A pairs about user context

    Return in this exact format (do not include any other text or indentation):
    NEIGHBOURHOOD: {selected_neighborhood}
    CRIME_TYPES: {crime_types_with_risk}
    CONTEXT: [
    "Q: [specific question]",
    "A: [specific answer]",
    "Q: [specific question]",
    "A: [specific answer]",
    "Q: [specific question]",
    "A: [specific answer]"
    ]"""
    
    structured_response = chat.invoke([{"role": "user", "content": structure_prompt}]).content
    
    # Parse the response into components
    try:
        # Split into sections
        sections = structured_response.strip().split('\n')
        structured_input = {}
        
        current_section = None
        context_lines = []
        
        for line in sections:
            line = line.strip()
            if line.startswith('NEIGHBOURHOOD:'):
                structured_input['neighbourhood'] = line.replace('NEIGHBOURHOOD:', '').strip()
            elif line.startswith('CRIME_TYPES:'):
                crime_types_str = line.replace('CRIME_TYPES:', '').strip()
                # Remove any extra whitespace and evaluate
                structured_input['crime_types'] = eval(crime_types_str.strip())
            elif line.startswith('CONTEXT:'):
                current_section = 'context'
            elif current_section == 'context' and line:
                if line not in ['[', ']']:  # Skip brackets
                    context_lines.append(line.strip('"').strip("'"))
        
        # Process context lines into pairs
        context_pairs = []
        for i in range(0, len(context_lines), 2):
            if i + 1 < len(context_lines):
                context_pairs.extend([context_lines[i], context_lines[i + 1]])
        
        structured_input['context'] = context_pairs
        
    except Exception as e:
        print(f"Error parsing structured response: {e}")
        print(f"Raw response:\n{structured_response}")
        raise
    
    # Format the structured input into the question format
    formatted_question = f"""
    LOCATION: {structured_input['neighbourhood']}
    
    SAFETY CONCERNS:
    - {', '.join(structured_input['crime_types'])}
    
    ADDITIONAL USER CONTEXT:
    {chr(10).join(structured_input['context'])}
    """
    
    # Generate ground truth safety plan using the same contexts
    ground_truth_prompt = f"""You are a City of Toronto safety advisor specializing in crime prevention and public safety in Toronto, Ontario. 
    
    Your goal is to create a comprehensive and actionable safety plan that addresses the user's concerns and enhances their safety, in the City of Toronto. Your tone should be respectful and professional.

    Create a detailed safety plan following this structure:

    1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:
    - Current safety landscape of the specified neighbourhood
    - Known risk factors and patterns
    - Specific areas or times that require extra caution

    2. TARGETED SAFETY RECOMMENDATIONS:
    For each crime concern mentioned:
    - Specific prevention strategies
    - Warning signs to watch for
    - Immediate actions to take if encountered
    - Available community resources

    3. PERSONAL SAFETY PROTOCOL:
    - Daily safety habits to develop
    - Essential safety tools or resources to have
    - Emergency contact information and procedures
    - Community support services available

    4. PREVENTIVE MEASURES:
    - Home/property security recommendations
    - Personal safety technology suggestions
    - Community engagement opportunities
    - Reporting procedures and important contacts

    Base the plan on:
    USER REQUEST:
    {formatted_question}

    VERIFIED RESOURCES:
    {chr(10).join([f"Content: {doc.page_content}\nSource: {doc.metadata.get('title', 'Untitled')} ({doc.metadata['source']})" 
                   for doc in retrieved_docs])}

    Guidelines for your response:
    - Be specific and refer only to the information provided in the input and context
    - Provide specific, actionable advice that can be implemented immediately
    - Include both preventive measures and emergency response protocols
    - Reference relevant Toronto Police Service programs or initiatives when applicable
    - Maintain a supportive and empowering tone while being clear about risks
    - Prioritize recommendations based on the specific crime patterns
    - Include relevant contact numbers and resources
    - If you use information from the context, cite the source
    
    If certain information is not available in the knowledge base, acknowledge this and provide general best practices, while encouraging the user to contact Toronto Police Service's non-emergency line for more specific guidance. 

    Refrain from providing legal, medical, financial or personal or professional advice, stay within the scope of a safety plan and a role as a safety advisor.

    Remember: Focus on prevention and awareness without causing undue alarm. Empower the user with knowledge and practical steps they can take to enhance their safety."""
    
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
            "model": "gpt-4o",
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
    test_cases = generate_test_set(cases_per_type=4)
    save_test_set(test_cases)
    print(f"Generated total of {len(test_cases)} test cases") 