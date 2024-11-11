"""
evaluation_set.py
Comprehensive evaluation set for Toronto Safety RAG system
"""

EVALUATION_SET = {
    "direct_questions": [
        {
            "question": "What are the emergency numbers in Toronto?",
            "expected_contexts": ["emergency services", "police contacts"],
            "expected_elements": ["911", "non-emergency number", "specific services"],
            "complexity": "simple"
        },
        {
            "question": "Where is the nearest police station to Union Station?",
            "expected_contexts": ["police division locations", "downtown services"],
            "expected_elements": ["division number", "address", "contact information"],
            "complexity": "simple"
        },
        {
            "question": "How do I report a non-emergency incident in Toronto?",
            "expected_contexts": ["police reporting", "non-emergency procedures"],
            "expected_elements": ["online reporting", "phone numbers", "reporting process"],
            "complexity": "simple"
        },
        {
            "question": "What are Toronto Police's operating hours?",
            "expected_contexts": ["police services", "contact information"],
            "expected_elements": ["24/7 availability", "division hours", "service times"],
            "complexity": "simple"
        },
        {
            "question": "How do I contact Toronto Crime Stoppers?",
            "expected_contexts": ["crime reporting", "anonymous tips"],
            "expected_elements": ["phone number", "online reporting", "anonymity process"],
            "complexity": "simple"
        }
    ],
    
    "scenario_based": [
        {
            "question": "I'm walking alone at night near Ryerson University, what should I do to stay safe?",
            "expected_contexts": ["night safety", "university area safety", "street safety"],
            "expected_elements": ["well-lit paths", "emergency contacts", "safe routes"],
            "complexity": "medium"
        },
        {
            "question": "I think someone is following me on the TTC, what steps should I take?",
            "expected_contexts": ["TTC safety", "personal safety", "emergency procedures"],
            "expected_elements": ["emergency alarm", "conductor contact", "safe exit strategies"],
            "complexity": "medium"
        },
        {
            "question": "I witnessed a theft at Eaton Centre, what should I do?",
            "expected_contexts": ["crime reporting", "witness procedures", "mall safety"],
            "expected_elements": ["security contact", "police reporting", "witness safety"],
            "complexity": "medium"
        },
        {
            "question": "Someone suspicious is trying to enter my apartment building, what should I do?",
            "expected_contexts": ["building security", "residential safety", "emergency response"],
            "expected_elements": ["building security", "police contact", "safe protocols"],
            "complexity": "medium"
        },
        {
            "question": "I'm being harassed by someone on social media who claims to be in Toronto, what should I do?",
            "expected_contexts": ["cybercrime", "online harassment", "digital safety"],
            "expected_elements": ["reporting procedures", "evidence collection", "safety steps"],
            "complexity": "medium"
        }
    ],
    
    "edge_cases": [
        {
            "question": "What should I do during an active threat situation in downtown Toronto?",
            "expected_contexts": ["emergency procedures", "threat response", "public safety"],
            "expected_elements": ["run/hide/fight", "emergency contacts", "police response"],
            "complexity": "high"
        },
        {
            "question": "How do I report a suspected terrorist activity in Toronto?",
            "expected_contexts": ["terrorism reporting", "emergency procedures"],
            "expected_elements": ["reporting channels", "observation details", "immediate actions"],
            "complexity": "high"
        },
        {
            "question": "What should I do if I encounter a violent protest in Toronto?",
            "expected_contexts": ["public safety", "crowd management", "emergency response"],
            "expected_elements": ["safe distance", "police contact", "exit routes"],
            "complexity": "high"
        },
        {
            "question": "How do I handle a potential bomb threat at my workplace in Toronto?",
            "expected_contexts": ["emergency procedures", "threat assessment", "evacuation"],
            "expected_elements": ["evacuation procedures", "police contact", "threat reporting"],
            "complexity": "high"
        },
        {
            "question": "What should I do if I discover a cybercrime operation targeting Toronto residents?",
            "expected_contexts": ["cybercrime", "digital safety", "crime reporting"],
            "expected_elements": ["reporting procedures", "evidence preservation", "contact authorities"],
            "complexity": "high"
        }
    ],
    
    "multi_part": [
        {
            "question": "I'm new to Toronto and living alone. What basic safety measures should I take at home and when commuting?",
            "expected_contexts": ["home safety", "commuting safety", "newcomer information"],
            "expected_elements": ["home security", "safe travel routes", "emergency contacts"],
            "complexity": "complex"
        },
        {
            "question": "I'm organizing a public event in downtown Toronto. What security measures should I consider and who should I contact?",
            "expected_contexts": ["event safety", "public gatherings", "security planning"],
            "expected_elements": ["permits", "security requirements", "emergency planning"],
            "complexity": "complex"
        },
        {
            "question": "How can I protect my elderly parents who live alone in Toronto from scams and break-ins?",
            "expected_contexts": ["senior safety", "fraud prevention", "home security"],
            "expected_elements": ["scam awareness", "security measures", "emergency contacts"],
            "complexity": "complex"
        },
        {
            "question": "What safety measures should I take when running a business in downtown Toronto, both for cyber and physical security?",
            "expected_contexts": ["business security", "cybersecurity", "physical safety"],
            "expected_elements": ["security systems", "cyber protection", "emergency procedures"],
            "complexity": "complex"
        },
        {
            "question": "I'm a student moving to Toronto. What should I know about campus safety, housing security, and safe transportation?",
            "expected_contexts": ["campus safety", "student housing", "public transit"],
            "expected_elements": ["campus security", "housing tips", "safe travel"],
            "complexity": "complex"
        }
    ],
    
    "location_specific": [
        {
            "question": "What safety precautions should I take in the PATH system during off-hours?",
            "expected_contexts": ["PATH system", "downtown safety", "off-hours security"],
            "expected_elements": ["operating hours", "security presence", "safe routes"],
            "complexity": "medium"
        },
        {
            "question": "How can I stay safe while visiting Kensington Market at night?",
            "expected_contexts": ["neighborhood safety", "night security", "market area"],
            "expected_elements": ["safe routes", "emergency contacts", "area awareness"],
            "complexity": "medium"
        },
        {
            "question": "What safety measures should I take when using parking garages near Rogers Centre?",
            "expected_contexts": ["parking safety", "downtown parking", "event security"],
            "expected_elements": ["parking tips", "security presence", "emergency help"],
            "complexity": "medium"
        },
        {
            "question": "How can I protect myself when using ATMs in the Financial District after business hours?",
            "expected_contexts": ["ATM safety", "financial district", "night security"],
            "expected_elements": ["ATM precautions", "surroundings awareness", "emergency numbers"],
            "complexity": "medium"
        },
        {
            "question": "What safety precautions should I take when visiting High Park during early morning or late evening hours?",
            "expected_contexts": ["park safety", "recreational areas", "off-hours security"],
            "expected_elements": ["park rules", "emergency contacts", "safe areas"],
            "complexity": "medium"
        }
    ]
}

# Metadata about expected performance
EVALUATION_METRICS = {
    "simple": {
        "min_faithfulness": 0.8,
        "min_answer_relevancy": 0.8,
        "min_retrieval_precision": 0.7,
        "min_retrieval_recall": 0.7
    },
    "medium": {
        "min_faithfulness": 0.7,
        "min_answer_relevancy": 0.7,
        "min_retrieval_precision": 0.6,
        "min_retrieval_recall": 0.6
    },
    "complex": {
        "min_faithfulness": 0.6,
        "min_answer_relevancy": 0.6,
        "min_retrieval_precision": 0.5,
        "min_retrieval_recall": 0.5
    }
}

def get_evaluation_questions():
    """Return a flat list of all questions for evaluation"""
    questions = []
    for category in EVALUATION_SET.values():
        questions.extend([item["question"] for item in category])
    return questions

def get_questions_by_complexity(complexity):
    """Return all questions of a specific complexity level"""
    questions = []
    for category in EVALUATION_SET.values():
        questions.extend([item["question"] for item in category if item["complexity"] == complexity])
    return questions

def get_expected_metrics(complexity):
    """Get the expected metric thresholds for a given complexity"""
    return EVALUATION_METRICS.get(complexity, EVALUATION_METRICS["medium"]) 