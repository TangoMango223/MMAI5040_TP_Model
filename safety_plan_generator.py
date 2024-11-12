"""
safety_plan_generator.py
Goal: Generate personalized safety plans for Toronto residents based on neighborhood and crime concerns.
"""

# Import statements
import os
from typing import List, Dict

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env", override=True)

# Define the prompt template
SAFETY_PLAN_PROMPT = PromptTemplate.from_template("""
You are a Toronto Police Service safety advisor specializing in crime prevention and public safety in Toronto, Ontario. Your role is to provide practical, evidence-based safety recommendations tailored to specific neighborhoods and crime concerns.

USER PROFILE:
Neighbourhood: {neighbourhood}
Primary Crime Concerns: {crime_concerns}
Additional Context: {user_context}

RELEVANT TORONTO POLICE RESOURCES:
{context}

Based on the provided information, create a detailed safety plan that includes:

1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:
- Current safety landscape of {neighbourhood}
- Known risk factors and patterns
- Specific areas or times that require extra caution

2. TARGETED SAFETY RECOMMENDATIONS:
For each crime concern ({crime_concerns}):
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

Guidelines for your response:
- Provide specific, actionable advice that can be implemented immediately
- Include both preventive measures and emergency response protocols
- Reference relevant Toronto Police Service programs or initiatives when applicable
- Maintain a supportive and empowering tone while being clear about risks
- Prioritize recommendations based on the neighbourhood's specific crime patterns
- Include relevant contact numbers and resources

If certain information is not available in the knowledge base, acknowledge this and provide general best practices while encouraging the user to contact Toronto Police Service's non-emergency line for more specific guidance.

Remember: Focus on prevention and awareness without causing undue alarm. Empower the user with knowledge and practical steps they can take to enhance their safety.
""")

def generate_safety_plan(
    neighbourhood: str,
    crime_concerns: List[str],
    user_context: Dict[str, str],
    return_all: bool = False
):
    """
    Generate a safety plan based on specific neighbourhood and crime concerns.
    
    Args:
        neighbourhood (str): Toronto neighbourhood name
        crime_concerns (List[str]): List of top 3 crime concerns
        user_context (Dict[str, str]): Additional context from user questions
        return_all (bool): Whether to return full context and retrieved documents
    
    Returns:
        str or dict: Formatted safety plan string, or dictionary with full context if return_all=True
    """
    # Format the crime concerns for the prompt
    formatted_crime_concerns = ", ".join(crime_concerns)
    
    # Format user context into a readable string
    formatted_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in user_context.items()])
    
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    chat = ChatOpenAI(verbose=True, temperature=0.2, model="gpt-4")
    
    # Create the chains
    stuff_documents_chain = create_stuff_documents_chain(chat, SAFETY_PLAN_PROMPT)
    qa = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_documents_chain)
    
    # Prepare input data
    input_data = {
        "neighbourhood": neighbourhood,
        "crime_concerns": formatted_crime_concerns,
        "user_context": formatted_context
    }
    
    # Generate the safety plan
    result = qa.invoke(input=input_data)
    
    if return_all:
        return {
            "answer": result["answer"],
            "contexts": result["context"],
            "input_data": input_data
        }
    
    # Format the final safety plan
    plan_string = f"""
    TORONTO POLICE SERVICE SAFETY PLAN
    Neighbourhood: {neighbourhood}
    Primary Concerns: {formatted_crime_concerns}

    {result["answer"]}

    Sources Consulted:
    {chr(10).join(f"- {doc.metadata.get('title', 'Untitled')} ({doc.metadata['source']})" 
                  for doc in result["context"])}
    
    Note: This safety plan is generated based on Toronto Police Service resources and general 
    safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
    call 416-808-2222.
    """
    
    return plan_string

# Example usage
if __name__ == "__main__":
    sample_input = {
        "neighbourhood": "Downtown Core - Union Station Area",
        "crime_concerns": [
            "Theft",
            "Assault",
            "Break and Enter"
        ],
        "user_context": {
            "When are peak crime hours?": "I often work late and leave the office around 11 PM",
            "Are there safe walking routes?": "I need to walk to Union Station",
            "What security measures exist?": "Looking for information about surveillance and police presence"
        }
    }
    
    result = generate_safety_plan(
        neighbourhood=sample_input["neighbourhood"],
        crime_concerns=sample_input["crime_concerns"],
        user_context=sample_input["user_context"]
    )
    print(result) 