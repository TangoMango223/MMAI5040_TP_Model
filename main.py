"""
main_v2.py
PURPOSE: This script generates a safety plan for a given neighbourhood and crime concerns, using a LLM and a vector database.

CHANGES: 
* Enhanced LangSmith tracing to show complete formatted safety plan in traces, will be used to make evaluation sets.
* Prompt engineering for the analysis chain, to improve context recall and faithfulness.
* Provided a good one-shot example from Trinity-Bellwoods, to help the LLM understand the style and tone expected from the safety plan.
* Added tagging so runs are easier to track in LangSmith.

Last Updated: 2024-11-18
Version: 3.00

USE CASE: Improved model versus "Base Model" evaluation of metric improvements.

Written by: Christine Tang
"""

# Import statements
import os
from typing import List, Dict

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from operator import itemgetter

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env", override=True)

# Define example safety plans
TRINITY_BELLWOODS_EXAMPLE = """
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: Trinity-Bellwoods (81)
    Primary Concerns: Break and Enter: medium, Assault: medium, Auto Theft: medium

    **NEIGHBOURHOOD-SPECIFIC ASSESSMENT:**
    Trinity-Bellwoods is a vibrant neighbourhood with a mix of residential and commercial areas. However, there have been reports of break and enters, assaults, and auto thefts. The park area, particularly during late hours, has been identified as a potential risk zone due to poor lighting.

    **TARGETED SAFETY RECOMMENDATIONS:**
    1. **Break and Enter**: 
    - Prevention Strategies: Improve home security by keeping a record of valuables, identifying property using a Trace Identified pen
    - Warning Signs: Suspicious activity around your property or neighbourhood
    - Immediate Actions: Report suspicious activity to police
    - Community Resources: Toronto Police Service's Break and Enter Prevention Guide

    [Rest of Trinity-Bellwoods example...]
"""

YORK_UNIVERSITY_EXAMPLE = """
    CITY OF TORONTO SERVICE SAFETY PLAN
    Neighbourhood: York University Heights (27)
    Primary Concerns: Assault: High, Robbery: High

    **NEIGHBOURHOOD-SPECIFIC ASSESSMENT:**
    York University Heights is a diverse area centered around a major educational institution. The neighbourhood experiences higher foot traffic during academic terms, particularly around campus and student housing areas. Safety concerns are heightened during evening hours and in less populated areas.

    **TARGETED SAFETY RECOMMENDATIONS:**
    1. **Assault**: 
    - Prevention Strategies: Use well-lit walkways, travel in groups, utilize campus safety services
    - Warning Signs: Isolated areas, suspicious individuals, lack of pedestrian traffic
    - Immediate Actions: Move to populated areas, contact campus security
    - Community Resources: York University Security Services, goSAFE program

    2. **Robbery**: 
    - Prevention Strategies: Stay alert, avoid displaying valuable items
    - Warning Signs: Being followed, suspicious gathering of individuals
    - Immediate Actions: Move to safe location, contact security
    - Community Resources: Toronto Police Service's Robbery Prevention Guide

    **PERSONAL SAFETY PROTOCOL:**
    - Daily Safety Habits: Plan routes in advance, stay in well-lit areas
    - Essential Safety Tools: Fully charged phone, personal alarm
    - Emergency Contacts: Campus Security (416-736-5333), Toronto Police
    - Community Support: goSAFE escort service, York U Safety App

    **PREVENTIVE MEASURES:**
    - Personal Safety Technology: York U Safety App, location sharing
    - Community Resources: Campus Walk Program, Security Shuttle Service
    - Reporting Procedures: Use York U Safety App or call Campus Security
    - Emergency Response: Program emergency numbers into phone
"""

# Create the list of examples
safety_plan_examples = [
    ("Trinity-Bellwoods", TRINITY_BELLWOODS_EXAMPLE),
    ("York University Heights", YORK_UNIVERSITY_EXAMPLE)
]

# Functions - Safety Plan Generation

def generate_safety_plan(
    neighbourhood: str,
    crime_type: List[str],
    user_context: List[str],
    ):
    
    """
    This code generates a safety plan based on specific neighbourhood and crime concerns.
    
    Chain of Thought is implemented, where the LLM is (1) Prompted to provide a detailed analysis considering only the information provided above, and (2) Prompted to provide a comprehensive and actionable safety plan that addresses the user's concerns and enhances their safety, in the City of Toronto.
    """
    
    # Example Safety Plans
    TRINITY_BELLWOODS_EXAMPLE = """
        CITY OF TORONTO SERVICE SAFETY PLAN
        Neighbourhood: Trinity-Bellwoods (81)
        Primary Concerns: Break and Enter: medium, Assault: medium, Auto Theft: medium

         **NEIGHBOURHOOD-SPECIFIC ASSESSMENT:**

        Trinity-Bellwoods is a vibrant neighbourhood with a mix of residential and commercial areas. However, there have been reports of break and enters, assaults, and auto thefts. The park area, particularly during late hours, has been identified as a potential risk zone due to poor lighting. It is advisable to exercise caution during late hours, especially when walking alone or with a pet.

        **TARGETED SAFETY RECOMMENDATIONS:**

        1. **Break and Enter**: 
        - Prevention Strategies: Improve home security by keeping a record of valuables, identifying property using a Trace Identified pen, and reporting burnt out lights on the property to the building superintendent or management immediately.
        - Warning Signs: Suspicious activity around your property or neighbourhood.
        - Immediate Actions: Report any suspicious activity to the police and advise the building Superintendent or Management.
        - Community Resources: Toronto Police Service's Break and Enter Prevention Guide.

        2. **Assault**: 
        - Prevention Strategies: Avoid poorly lit areas, especially during late hours. Be aware of your surroundings and move towards an area with more people if you feel uncomfortable.
        - Warning Signs: Unfamiliar individuals or groups loitering in poorly lit areas.
        - Immediate Actions: If you feel threatened, move to a well-lit area with people around and call the police.
        - Community Resources: Toronto Police Service's Personal Safety Guide.

        3. **Auto Theft**: 
        - Prevention Strategies: Always lock your car, keep windows rolled up, park in well-lit areas, and avoid leaving valuables in the car.
        - Warning Signs: Suspicious individuals loitering around parking areas.
        - Immediate Actions: If you notice suspicious activity around your vehicle, report it to the police.
        - Community Resources: Toronto Police Service's Auto Theft Prevention Guide.

        **PERSONAL SAFETY PROTOCOL:**

        - Daily Safety Habits: Be aware of your surroundings, especially during late hours. Keep your home and vehicle secure. Limit the use of electronic devices when walking alone.
        - Essential Safety Tools: Consider carrying a personal alarm or whistle. Have emergency numbers saved in your phone.
        - Emergency Contact Information: Toronto Police Service (Non-Emergency): 416-808-2222, Emergency: 911, Crime Stoppers: 416-222-TIPS (8477).
        - Community Support Services: Neighbourhood Watch Program, Toronto Crime Stoppers.

        **PREVENTIVE MEASURES:**

        - Home/Property Security Recommendations: Install good quality locks on doors and windows. Consider installing a home security system.
        - Personal Safety Technology Suggestions: Consider using personal safety apps that can share your location with trusted contacts.
        - Community Engagement Opportunities: Join or start a Neighbourhood Watch Program. Attend community safety meetings.
        - Reporting Procedures: Report any suspicious activity to the Toronto Police Service's non-emergency line or Crime Stoppers. In case of an emergency, call 911.

        Remember, your safety is paramount. By taking these preventive measures and being aware of your surroundings, you can significantly enhance your safety in Trinity-Bellwoods.

        Sources Consulted:
        - Transit Safety -  Toronto Police Service  (https://www.tps.ca/crime-prevention/transit-safety/)
        - Apartment, Condo Security -  Toronto Police Service  (https://www.tps.ca/crime-prevention/apartment-condo-security-1/)
        - Crime Prevention Through Environmental Design -  Toronto Police Service  (https://www.tps.ca/crime-prevention/crime-prevention-through-environmental-design/)
        -  (https://www.tps.ca/crime-prevention/feed)
        
        ----
        
         Note: This safety plan is generated based on Toronto Police Service resources and general 
         safety guidelines. For emergencies, always call 911. For non-emergency police matters, call 416-808-2222.
        """
    
    ANNEX_EXAMPLE = """
        CITY OF TORONTO SERVICE SAFETY PLAN
        Neighbourhood: Annex (95)
        Primary Concerns: Break and Enter: low, Assault: low

        1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:

        The Annex neighbourhood is generally safe with low rates of break-ins and assaults. However, like any urban area, it is not immune to crime. The primary concerns are break-ins and personal safety when walking alone at night. It is advisable to be extra cautious during late-night hours and in less populated areas. 

        2. TARGETED SAFETY RECOMMENDATIONS:

        Break-ins:
        - Prevention Strategies: Continue to keep doors and windows locked at all times. Consider installing a peephole to see who is outside without opening the door. Verify a person's identification before opening the door.
        - Warning Signs: Look out for suspicious activity in your neighbourhood, such as unfamiliar vehicles or individuals loitering around homes.
        - Immediate Actions: If you notice signs of a break-in, do not enter your home. Go to a neighbour's house and call the police.
        - Community Resources: Stay connected with your neighbourhood watch group and report any suspicious activity. 

        Personal Safety:
        - Prevention Strategies: Avoid walking alone late at night. Have your keys ready when approaching your home.
        - Warning Signs: Be aware of your surroundings. If you notice someone following you or behaving suspiciously, seek help immediately.
        - Immediate Actions: If you feel threatened, call 911 immediately. 
        - Community Resources: Utilize the community app to stay informed about local incidents and safety alerts.

        3. PERSONAL SAFETY PROTOCOL:

        - Daily Safety Habits: Be aware of your surroundings, especially when walking alone at night. Keep your doors and windows locked at all times.
        - Essential Safety Tools: Consider carrying a personal alarm or whistle. 
        - Emergency Contact Information: Keep the local police station's non-emergency number (416-808-2222) saved in your phone. In case of an emergency, dial 911.
        - Community Support Services: Stay active in your neighbourhood watch group and use the community app to stay informed.

        4. PREVENTIVE MEASURES:

        - Home/Property Security Recommendations: Consider installing a home security system or surveillance cameras. Report any burnt out lights on your property to the building superintendent or management immediately.
        - Personal Safety Technology Suggestions: Consider using a personal safety app that can alert authorities or trusted contacts if you're in danger.
        - Community Engagement Opportunities: Participate in community safety initiatives and events. Encourage your neighbours to join the neighbourhood watch group.
        - Reporting Procedures: Report any suspicious activity to the police and advise the building superintendent or management. 

        Remember, your safety is a priority. Stay vigilant, stay informed, and don't hesitate to reach out to the Toronto Police Service for any concerns or questions.

        Sources Consulted:
        - Break & Enter Prevention -  Toronto Police Service  (https://www.tps.ca/crime-prevention/break-and-enter-prevention/)
        - Apartment, Condo Security -  Toronto Police Service  (https://www.tps.ca/crime-prevention/apartment-condo-security-1/)
        - Crime Prevention Through Environmental Design -  Toronto Police Service  (https://www.tps.ca/crime-prevention/crime-prevention-through-environmental-design/)
        - Crime Prevention -  Toronto Police Service  (https://www.tps.ca/crime-prevention/)
        - Your Personal Safety Checklist ‚Äì City of Toronto (https://www.toronto.ca/community-people/public-safety-alerts/safety-tips-prevention/posters-pamphlets-and-other-safety-resources/your-personal-safety-checklist/)
                
        ----
                
        Note: This safety plan is generated based on Toronto Police Service resources and general 
        safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
        call 416-808-2222.
        """
    
    YORK_UNIVERSITY_HEIGHTS_EXAMPLE = """
            CITY OF TORONTO SERVICE SAFETY PLAN
            Neighbourhood: York University Heights (27)
            Primary Concerns: Assault: High, Robbery: High

            **1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:**

            York University Heights is a vibrant community located in the northwestern part of Toronto. It is home to York University, which brings a diverse population of students, faculty, and residents. The area features a mix of residential, commercial, and educational spaces. However, like many urban areas, it faces challenges related to crime, particularly assault and robbery.

            ate evenings and early mornings.

            **2. TARGETED SAFETY RECOMMENDATIONS:**

            **Assault:**
            - **Prevention Strategies:** Avoid walking alone at night; use well-lit, busy streets. Consider walking with a friend or using public transportation.
            - **Warning Signs:** Be aware of individuals loitering or following you. Trust your instincts if you feel uncomfortable.
            - **Immediate Actions:** If you feel threatened, move to a populated area and call 9-1-1. Use your phone to alert someone of your location.
            - **Community Resources:** Engage with campus security services and local police for safety escorts or patrols.

            **Robbery:**
            - **Prevention Strategies:** Keep valuables out of sight and avoid using your phone in isolated areas. Plan your routes to avoid high-risk areas.
            - **Warning Signs:** Be cautious of individuals approaching you in a suspicious manner or asking for directions in secluded areas.
            - **Immediate Actions:** If confronted, prioritize your safety over possessions. Comply with demands and call 9-1-1 when safe.
            - **Community Resources:** Utilize Crime Stoppers to report suspicious activities anonymously.

            **3. PERSONAL SAFETY PROTOCOL:**

            - **Daily Safety Habits:** Develop a habit of checking your surroundings regularly. Plan and familiarize yourself with safe routes to and from your home.
            - **Essential Safety Tools:** Carry a personal alarm or whistle. Use safety apps that allow real-time location sharing with trusted contacts.
            - **Emergency Contacts:** Save emergency numbers on speed dial, including 9-1-1, Toronto Police non-emergency line (416-808-2222), and campus security.
            - **Community Support Services:** Participate in local safety workshops and community meetings to stay informed and connected.

            **4. PREVENTIVE MEASURES:**

            - **Home/Property Security:** Ensure your home is well-lit and secure. Consider installing motion-sensor lights and security cameras.
            - **Personal Safety Technology:** Use apps like "bSafe" or "Life360" for location sharing and emergency alerts.
            - **Community Engagement:** Join or initiate a neighbourhood watch program. Attend community safety meetings to voice concerns and collaborate on solutions.
            - **Reporting Procedures:** Report any suspicious activities to the Toronto Police Service. Use Crime Stoppers (1-800-222-8477) for anonymous tips.

            By implementing these strategies, you can enhance your personal safety and contribute to a safer community environment in York University Heights. Stay informed, stay connected, and prioritize your safety in all situations.

            Sources Consulted:
            - Your Personal Safety Checklist – City of Toronto (https://www.toronto.ca/community-people/public-safety-alerts/safety-tips-prevention/posters-pamphlets-and-other-safety-resources/your-personal-safety-checklist/)
            - Apartment, Condo Security -  Toronto Police Service  (https://www.tps.ca/crime-prevention/apartment-condo-security-1/)
            - Transit Safety -  Toronto Police Service  (https://www.tps.ca/crime-prevention/transit-safety/)
            - Crime Prevention Through Environmental Design -  Toronto Police Service  (https://www.tps.ca/crime-prevention/crime-prevention-through-environmental-design/)
            - Crime Prevention -  Toronto Police Service  (https://www.tps.ca/crime-prevention/)
            
            ----
            
            Note: This safety plan is generated based on Toronto Police Service resources and general 
            safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
            call 416-808-2222.
            """
    
    # Create list of safety plan exampls:
    # Too expensive to run these examples on every request, only passing one example.
    safety_plan_examples = [
        ("Trinity-Bellwoods", TRINITY_BELLWOODS_EXAMPLE),
        # ("Annex", ANNEX_EXAMPLE),
        # ("York University Heights", YORK_UNIVERSITY_HEIGHTS_EXAMPLE),
    ]
    
    
    # Define the prompt template
    FIRST_SAFETY_PROMPT = PromptTemplate.from_template("""
    You are a City of Toronto safety advisor specializing in crime prevention and public safety in Toronto, Ontario.

    Your goal is to brainstorm and analyze safety concerns and resources to generate actionable insights for creating an effective safety plan. The target audience is a member of the general public in Toronto.

    USER REQUEST:
    {input}

    RELEVANT TORONTO POLICE, CITY OF TORONTO, AND GOVERNMENT RESOURCES:
    {context}
    
    Please brainstorm and analyze the information, considering the perspective of the general public in Toronto.

    STEP 1 - UNDERSTANDING PUBLIC SAFETY NEEDS:
    - Evaluate how these safety concerns affect daily routines, activities, and community life.
    - Assess how time, location, and environmental factors (e.g., seasons, times of day) influence risks.
    - Reflect on the availability and accessibility of safety resources.

    STEP 2 - CRITICAL ANALYSIS:
    - Assess factors and how they influence the safety concerns.
    - Analyze how the user's specific context may affect their level of safety in the city of Toronto.
    - Identify practical and accessible safety measures, considering both immediate and preventive strategies.

    STEP 3 - EVIDENCE VERIFICATION:
    - Use the information provided as the foundation for your analysis.
    - Should there be any gaps in the information provided, explicitly state this, refrain from making assumptions.
    - Explicitly connect recommendations to the provided context and sources cited.
    - Highlight any areas where information is incomplete or additional data would be beneficial.

    Structure your response as:
    1. **Public Impact Assessment**
    - Impact of safety concerns on daily life.
    - Practical implications for residents.
    - Community-level considerations.

    2. **Situation Analysis**
    - Key safety concerns and their relationships.
    - Environmental and contextual factors.
    - User-specific considerations.

    3. **Resource Assessment**
    - Verified facts and recommendations (with citations).
    - Accessible public safety resources.
    - Information gaps.

    4. **Strategic Insights**
    - Brainstorm safety measures tailored to the user’s needs.
    - Priority areas based on risk levels.
    - Practical solutions accessible to the general public.

    Focus on providing practical, accessible guidance that Toronto residents can realistically implement in their daily lives. Maintain factual accuracy and proper citation of sources.
    
    Refrain from providing legal, medical, financial or personal or professional advice, stay within the scope of a safety plan and a role as a safety advisor.
    """)
    
    SECOND_SAFETY_PROMPT = PromptTemplate.from_template("""
    You are a City of Toronto safety advisor specializing in crime prevention and public safety in Toronto, Ontario. 
    
    Your goal is to transform the provided analysis into an actionable, tailored safety plan that supports the user's safety concerns and enhances their safety, in the City of Toronto. Your tone should be respectful and professional.
    
    You are provided the following information regarding the user:
    {input}
    
    I have conducted the following analysis regarding the user's request and the relevant resources:
    <analysis>
    {analysis}
    </analysis>

    Here are some example safety plans to guide your format:
    {example_safety_plans}
    
    Based on the provided information and examples above, create a detailed safety plan that adheres to the structure of the following 4 sections:

    1. NEIGHBOURHOOD-SPECIFIC ASSESSMENT:
    - Current safety landscape of the specified neighbourhood
    - Known risk factors and patterns
    - Specific areas or times that require extra caution
    - Provide a general overview of the neighbourhood, including its key features, landmarks, and any other relevant information. If user asks for neighbourhood-specific information and you cannot provide this information, focus on providing general information about the neighbourhood.

    2. TARGETED SAFETY RECOMMENDATIONS:
    Address all safety concerns requested by the user, regardless of crime level (i.e. low, medium, high).
    Follow the format for this section, with the following 4 points for each concern:
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
    - Where possible, refrain from repeating the same information and advice across different sections unless relevant and necessary
    - Include both preventive measures and emergency response protocols
    - Reference relevant Toronto Police Service programs or other city resources when applicable
    - Base your guidance on the information provided
    
    If certain information is not available in the knowledge base, acknowledge this and provide general best practices, while encouraging the user to contact Toronto Police Service's non-emergency line for more specific guidance. Refrain from introducing unverified information or making unsupported assumptions. 

    Refrain from providing legal, medical, financial, personal or professional advice, stay within the scope of a safety plan and a role as a safety advisor.

    Remember: Focus on prevention and awareness without causing undue alarm. Empower the user with knowledge and practical steps they can take to enhance their safety.
    """)
    
    # Format the crime concerns for the prompt
    formatted_crime_concerns = ", ".join(crime_type)
    formatted_context = "\n".join(user_context)
    
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )
    
    # Initialize the LLM and the VectorStore retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chat = ChatOpenAI(verbose=True, 
                      temperature=0.0,
                      model="gpt-4")
    
    # Create the analysis chain
    analysis_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=create_stuff_documents_chain(
            llm=chat,
            prompt=FIRST_SAFETY_PROMPT
        )
    )
    
    # Create the plan generation chain using LCEL
    plan_prompt = SECOND_SAFETY_PROMPT
    plan_chain = (
        {
            "input": lambda x: x["input"],
            "analysis": lambda x: x["analysis"],
            "example_safety_plans": lambda x: x["example_safety_plans"]
        } |
        plan_prompt | 
        chat | 
        StrOutputParser()
    )
    
    # Create a chain using LCEL syntax with complete plan formatting
    safety_plan_chain = (
        analysis_chain | 
        {
            "input": itemgetter("input"),
            "analysis": itemgetter("answer"),
            "example_safety_plans": lambda _: "\n\n".join(
                f"=== EXAMPLE: {name} ===\n{example}" 
                for name, example in safety_plan_examples
            ),
            "context": itemgetter("context")
        } | 
        {
            "plan": plan_chain,
            "context": itemgetter("context")
        } |
        # Add final transformation step to create the complete plan_string
        (lambda x: {
            "final_plan": f"""
            CITY OF TORONTO SERVICE SAFETY PLAN
            Neighbourhood: {neighbourhood}
            Primary Concerns: {formatted_crime_concerns}

            {x["plan"]}

            Sources Consulted:
            {chr(10).join([f"- {title} ({source})" for title, source in {
                (doc.metadata.get('title', 'Untitled'), doc.metadata['source'])
                for doc in x["context"]
            }])}
            
            ----
            
            Note: This safety plan is generated based on Toronto Police Service resources and general 
            safety guidelines. For emergencies, always call 911. For non-emergency police matters, 
            call 416-808-2222.
            """
        })
    )
    
    # Format the input
    formatted_user_input = {
        "input": f"""
        LOCATION: {neighbourhood}
        
        SAFETY CONCERNS:
        - {formatted_crime_concerns}
        
        ADDITIONAL USER CONTEXT:
        {formatted_context}
        """
    }
    
    # Run the chain
    chain_result = safety_plan_chain.invoke(
    formatted_user_input,
    config={
        "tags": [
            "application_example", 
            "one_shot_example",
            "trinity_bellwoods_york_examples",
            f"model_{chat.model_name}"
        ],
    }
    ) 

    # Put it into a plan_string.
    # This new format should allow us to see the full plan in LangSmith traces.
    plan_string = chain_result["final_plan"]
    
    # Return the final formatted plan
    return plan_string

# Main Control to run function:
if __name__ == "__main__":
    # Test Case - sample input agreed upon with group
    sample_input = {
        "neighbourhood": "York University Heights (27)",
        "crime_type": ["Assault: High", "Robbery: High"],
        "user_context": [
            "Q: Do you carry a fully charged phone?", 
            "A: Yes", 
            "Q: Do you have planned safe routes to and from your home?", 
            "A: No", 
            "Q: Do you avoid isolated, poorly lit areas while walking at night?", 
            "A: No",
            "Q: Would you like some neighbourhood-specific statistics?",
            "A: Yes"
        ]
    }

    # Run the function with the sample input
    result = generate_safety_plan(
        neighbourhood=sample_input["neighbourhood"],
        crime_type=sample_input["crime_type"],
        user_context=sample_input["user_context"]
    )

    # Print the result
    print(result) 