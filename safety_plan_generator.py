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

# -------------------------------------------------------------------------------------------------

# Few Shot Prompting Examples - Best Safety Plans generated:    
# Christine fill this out later with the group.






# -------------------------------------------------------------------------------------------------


# Define the prompt template
SAFETY_PLAN_PROMPT = PromptTemplate.from_template("""
You are a Toronto Police Service safety advisor specializing in crime prevention and public safety in Toronto, Ontario. 
Your role is to provide practical, evidence-based safety recommendations.

USER REQUEST:
{input}

RELEVANT TORONTO POLICE RESOURCES:
{context}

Based on the provided information, create a detailed safety plan that includes:

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

Guidelines for your response:
- Provide specific, actionable advice that can be implemented immediately
- Include both preventive measures and emergency response protocols
- Reference relevant Toronto Police Service programs or initiatives when applicable
- Maintain a supportive and empowering tone while being clear about risks
- Prioritize recommendations based on the specific crime patterns
- Include relevant contact numbers and resources

If certain information is not available in the knowledge base, acknowledge this and provide general best practices while encouraging the user to contact Toronto Police Service's non-emergency line for more specific guidance.

Remember: Focus on prevention and awareness without causing undue alarm. Empower the user with knowledge and practical steps they can take to enhance their safety.
""")


# -------------------------------------------------------------------------------------------------

def generate_safety_plan(
    neighbourhood: str,
    crime_concerns: List[str],
    user_context: Dict[str, str],
):
    """
    Generate a safety plan based on specific neighbourhood and crime concerns.
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
    
    # Initialize the LLM and the VectorStore retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4o")
    
    # Create the chains
    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=SAFETY_PLAN_PROMPT
    )
    
    # Create the retrieval chain
    # https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html
    # QA Chain only accepts two inputs, input and context.
    qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_documents_chain
    )
    
    # Format the input as a structured string
    # Need to combine into one string, "input" to QA Chain.
    formatted_input = f"""
    LOCATION: {neighbourhood}
    
    SAFETY CONCERNS:
    - {formatted_crime_concerns}
    
    ADDITIONAL CONTEXT:
    {formatted_context}
    """
    
    # Run the chain with simplified input structure
    result = qa.invoke({
        "input": formatted_input
    })
    
    # Get unique sources by creating a set of tuples containing title and source
    unique_sources = {
        (doc.metadata.get('title', 'Untitled'), doc.metadata['source'])
        for doc in result["context"]
    }
    
    # Format the final safety plan with unique sources
    plan_string = f"""
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: {neighbourhood}
    Primary Concerns: {formatted_crime_concerns}

    {result["answer"].split("Sources Consulted:")[0].strip()}

    Sources Consulted:
    {chr(10).join(f"- {title} ({source})" for title, source in unique_sources)}
    
    Note: This safety plan is generated based on Toronto Police Service resources and general 
    safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
    call 416-808-2222.
    """
    
    return plan_string

# Example usage
if __name__ == "__main__":
    # Test Case
    sample_input = {
        "neighbourhood": "York University - Keele Street",
        "crime_concerns": [
            "Theft",
            "Assault",
            "Break and Enter"
        ],
        "user_context": {
            "How often do you walk around this area?": "I study in the area, and I often walk back to residence late.",
            "Are you looking to find safe walking routes?": "Yes, I need to walk to Union Station to catch my GO train.",
            "Are you looking for what security measures exist?": "Yes, I would like to know what information exists about surveillance and police presence."
        }
    }
    
    result = generate_safety_plan(
        neighbourhood=sample_input["neighbourhood"],
        crime_concerns=sample_input["crime_concerns"],
        user_context=sample_input["user_context"]
    )
    print(result) 