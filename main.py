"""
main.py
PURPOSE: This script generates a safety plan for a given neighbourhood and crime concerns, using a LLM and a vector database.
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

# -----------------

def generate_safety_plan(
    neighbourhood: str,
    crime_type: List[str],
    user_context: str,
    ):
    
    """
    This code generates a safety plan based on specific neighbourhood and crime concerns.
    
    Chain of Thought is implemented, where the LLM is (1) Prompted to provide a detailed analysis considering only the information provided above, and (2) Prompted to provide a comprehensive and actionable safety plan that addresses the user's concerns and enhances their safety, in the City of Toronto.
    """
    
    # Example Safety Plan
    example_safety_plan = """
    ### Safety Plan for York University - Keele Street Area

#### 1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT

**Current Safety Landscape:**
- York University’s Keele campus is a large, bustling area with a mix of students, faculty, and visitors. While generally safe, it has experienced incidents of theft, assault, and break-ins, particularly in less populated areas and during late hours.

**Known Risk Factors and Patterns:**
- Theft and break-ins are more common in isolated areas, such as parking lots and less frequented pathways.
- Assaults have been reported during late hours, often targeting individuals walking alone.

**Specific Areas or Times Requiring Extra Caution:**
- Isolated pathways and parking lots, especially after dark.
- Areas with limited lighting or natural surveillance.

#### 2. TARGETED SAFETY RECOMMENDATIONS

**Theft Prevention:**
- **Prevention Strategies:** Keep valuables out of sight and secure. Use lockers or secure storage for personal items.
- **Warning Signs:** Be cautious of individuals loitering or acting suspiciously near personal belongings.
- **Immediate Actions:** If you suspect theft, report it immediately to campus security or the Toronto Police.
- **Community Resources:** Utilize campus security escort services when traveling alone at night.

**Assault Prevention:**
- **Prevention Strategies:** Walk in groups, especially at night. Use well-lit and populated routes.
- **Warning Signs:** Be aware of individuals following you or behaving aggressively.
- **Immediate Actions:** If threatened, attract attention by shouting or using a personal alarm. Seek help from nearby people or buildings.
- **Community Resources:** Access emergency phones located throughout the campus for immediate assistance.

**Break and Enter Prevention:**
- **Prevention Strategies:** Ensure doors and windows are locked when leaving your residence or office.
- **Warning Signs:** Unfamiliar individuals attempting to access buildings or rooms.
- **Immediate Actions:** Report suspicious activity to campus security or police.
- **Community Resources:** Participate in campus safety workshops to learn more about securing personal spaces.

#### 3. PERSONAL SAFETY PROTOCOL

**Daily Safety Habits:**
- Plan your route and inform someone of your expected arrival time.
- Stay alert and avoid distractions like using your phone while walking.

**Essential Safety Tools or Resources:**
- Carry a personal alarm or whistle.
- Use campus safety apps that provide real-time alerts and emergency contacts.

**Emergency Contact Information and Procedures:**
- Toronto Police Emergency: 9-1-1
- Toronto Police Non-Emergency: 416-808-2222
- York University Security Services: [Insert contact number]

**Community Support Services Available:**
- York University offers counseling and support services for students affected by crime.

#### 4. PREVENTIVE MEASURES

**Home/Property Security Recommendations:**
- Install additional locks or security devices on doors and windows.
- Use motion-sensor lighting around entrances.

**Personal Safety Technology Suggestions:**
- Download safety apps that provide location tracking and emergency alerts.
- Use smart home devices to monitor and secure your residence.

**Community Engagement Opportunities:**
- Join or form a neighborhood watch group with fellow students and residents.
- Participate in campus safety meetings and workshops.

**Reporting Procedures and Important Contacts:**
- Report crimes anonymously via Crime Stoppers at 1-800-222-8477 or [www.222tips.com](http://222tips.com).
- For immediate threats, contact campus security or call 9-1-1.

By following these guidelines and utilizing available resources, you can enhance your personal safety and contribute to a safer community environment. Stay informed, stay alert, and prioritize your well-being.
        
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
    
    # Format the crime concerns for the prompt
    formatted_crime_concerns = ", ".join(crime_type)
    
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
    plan_string = f"""
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: {neighbourhood}
    Primary Concerns: {formatted_crime_concerns}

    {final_safety_plan.content}

    Sources Consulted:
    {chr(10).join(f"- {title} ({source})" for title, source in unique_sources)}
    
    
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
        'crime_type': ['Assault: Low', 'Auto Theft: Medium', 'Break and Enter: Low', 'Robbery: Medium;'], 
        "user_context": {
            ['Q: Preferred Parking Spot Lighting', 'A: Well-Lit Area', 'Q: Select Anti-Theft Devices for Your Car', 'A: True', 'Q: Select Home Security Measures', 'A: True']
            }
    }

    
    result = generate_safety_plan(
        neighbourhood=sample_input["neighbourhood"],
        crime_concerns=sample_input["crime_type"],
        user_context=sample_input["user_context"]
    )
    print(result) 