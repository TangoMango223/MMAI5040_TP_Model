"""
main.py
PURPOSE: This script generates a safety plan for a given neighbourhood and crime concerns, using a LLM and a vector database.

Last Updated: 2024-11-14
Version: 2.0
Written by: Christine Tang
"""

# -----------------

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

# -----------------

def generate_safety_plan(
    neighbourhood: str,
    crime_type: List[str],
    user_context: List[str],
    ):
    
    """
    This code generates a safety plan based on specific neighbourhood and crime concerns.
    
    Chain of Thought is implemented, where the LLM is (1) Prompted to provide a detailed analysis considering only the information provided above, and (2) Prompted to provide a comprehensive and actionable safety plan that addresses the user's concerns and enhances their safety, in the City of Toronto.
    """
    
    # Example Safety Plan
    example_safety_plan = """
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: Agincourt North (129)
    Primary Concerns: Assault: Low, Auto Theft: Medium, Break and Enter: Low, Robbery: Medium

    1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:

    Agincourt North is generally a safe neighbourhood with low levels of assault and break and enter incidents. However, there are medium levels of auto theft and robbery. It's important to be vigilant, especially during late hours and in less crowded areas. Parking lots and streets with less foot traffic may be hotspots for auto theft. 

    2. TARGETED SAFETY RECOMMENDATIONS:

    - Assault and Break and Enter: Although these crimes are low in your area, it's important to stay vigilant. Keep your home well-lit, especially around entrances and exits. Install a peephole or doorbell camera to monitor who approaches your home. If you notice any suspicious activity, report it to the police immediately.

    - Auto Theft and Robbery: Always lock your car doors and keep windows rolled up. Check inside your car before entering, including the back seat. If you notice anyone loitering around parking areas, report it to the police. 

    3. PERSONAL SAFETY PROTOCOL:

    Develop daily safety habits such as locking all doors and windows when leaving home, and keeping your car keys ready when walking to your vehicle. Keep a list of emergency contacts in your phone and a physical copy at home. In case of an emergency, call 9-1-1. For non-emergencies, call the Toronto Police at 416-808-2222. 

    4. PREVENTIVE MEASURES:

    Enhance your home security by installing a security system and outdoor lighting. Consider using personal safety technology such as a personal alarm or a safety app on your phone. Engage with your community through neighbourhood watch programs. Report any suspicious activity to the police and keep their non-emergency number handy: 416-808-2222.

    Remember, your safety is a priority. Stay vigilant, be aware of your surroundings, and don't hesitate to report any suspicious activity. The Toronto Police Service is here to help and support you.

    Sources Consulted:
    - Crime Prevention -  Toronto Police Service  (https://www.tps.ca/crime-prevention/)
    - Break & Enter Prevention -  Toronto Police Service  (https://www.tps.ca/crime-prevention/break-and-enter-prevention/)
    - Crime Prevention Through Environmental Design -  Toronto Police Service  (https://www.tps.ca/crime-prevention/crime-prevention-through-environmental-design/)
    - Your Personal Safety Checklist – City of Toronto (https://www.toronto.ca/community-people/public-safety-alerts/safety-tips-prevention/posters-pamphlets-and-other-safety-resources/your-personal-safety-checklist/)
    - Apartment, Condo Security -  Toronto Police Service  (https://www.tps.ca/crime-prevention/apartment-condo-security-1/)
        
    ----
        
    Note: This safety plan is generated based on Toronto Police Service resources and general 
    safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
    call 416-808-2222.
            
    """
    
    
    # Define the prompt template
    FIRST_SAFETY_PROMPT = PromptTemplate.from_template("""
    You are a City of Toronto safety advisor specializing in crime prevention and public safety in Toronto, Ontario. 
    
    Your task is to provide a relevant, factual and meaningful analysis based on the user's request and the relevant resources.

    USER REQUEST:
    {input}

    RELEVANT TORONTO POLICE, CITY OF TORONTO, AND GOVERNMENT RESOURCES:
    {context}
    
    Please brainstorm, think through this information, and provide a detailed analysis considering only the information provided above. Address the following points:

    In your analysis:
    - Be specific and refer only to the information provided in the input and context.
    - If the input or context doesn't provide sufficient information for any point, clearly state this lack of information.
    - If you use information from the context, cite the source.
    
    Refrain from providing legal, medical, financial or personal or professional advice, stay within the scope of a safety plan and a role as a safety advisor.
    """)
    
    SECOND_SAFETY_PROMPT = PromptTemplate.from_template("""You are a City of Toronto safety advisor specializing in crime prevention and public safety in Toronto, Ontario. 
    
    Your goal is to synthesize the provided analysis into an actionable, tailored safety plan that supports the user's safety concerns and enhances their safety, in the City of Toronto. Your tone should be respectful and professional.
    
    You are provided the following information regarding the user:
    {input}
    
    I have conducted the following analysis regarding the user's request and the relevant resources:
    
    <analysis>
    {analysis}
    </analysis>
    
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
    
    If certain information is not available in the knowledge base, acknowledge this and provide general best practices, while encouraging the user to contact Toronto Police Service's non-emergency line for more specific guidance. 

    Refrain from providing legal, medical, financial or personal or professional advice, stay within the scope of a safety plan and a role as a safety advisor.

    Remember: Focus on prevention and awareness without causing undue alarm. Empower the user with knowledge and practical steps they can take to enhance their safety.
    """)
    
    # Format the crime concerns for the prompt - remove any trailing semicolons
    formatted_crime_concerns = ", ".join(crime_type).rstrip(';')
    
    # Convert the list of Q&A strings into a formatted string
    formatted_context = "\n".join(user_context)
    
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    
    # Initialize the LLM and the VectorStore retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4o")
    
    # Initialize the stuffing chain:
    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=FIRST_SAFETY_PROMPT
    )
    
    # Create Question-Answering Chain:
    qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_documents_chain
    )
    
    # Format the input as a structured string
    formatted_user_input = f"""
    LOCATION: {neighbourhood}
    
    SAFETY CONCERNS:
    - {formatted_crime_concerns}
    
    ADDITIONAL USER CONTEXT:
    {formatted_context}
    """
    
    # Run the first invocation = "Brainstorming" step
    first_result = qa.invoke({
        "input": formatted_user_input
    })
    
    
    # Run the second invocation = "Safety Plan" step
    final_safety_plan = chat.invoke(SECOND_SAFETY_PROMPT.format(
        input=formatted_user_input,
        analysis=first_result["answer"],
        example_safety_plan=example_safety_plan
    ))
    
    # Get unique sources by creating a set of tuples containing title and source
    unique_sources = {
        (doc.metadata.get('title', 'Untitled'), doc.metadata['source'])
        for doc in first_result["context"]
    }
    
    # Format the final safety plan with unique sources
    sources_list = [f"- {title} ({source})" for title, source in unique_sources]
    sources_text = "\n".join(sources_list)
    
    plan_string = f"""
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: {neighbourhood}
    Primary Concerns: {formatted_crime_concerns}

    {final_safety_plan.content}

    Sources Consulted:
    {sources_text}
    
    ----
    
    Note: This safety plan is generated based on Toronto Police Service resources and general 
    safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
    call 416-808-2222.
    """
    
    return plan_string


# Main Control to run function:
if __name__ == "__main__":
    # Test Case
    sample_input = {
        "neighbourhood": "Agincourt North (129)",
        "crime_type": ["Assault: Low", "Auto Theft: Medium", "Break and Enter: Low", "Robbery: Medium"],
        "user_context": [
            "Q: Preferred Parking Spot Lighting", 
            "A: Well-Lit Area", 
            "Q: Select Anti-Theft Devices for Your Car", 
            "A: True", 
            "Q: Select Home Security Measures", 
            "A: True"
        ]
    }

    result = generate_safety_plan(
        neighbourhood=sample_input["neighbourhood"],
        crime_type=sample_input["crime_type"],
        user_context=sample_input["user_context"]
    )
    print(result) 