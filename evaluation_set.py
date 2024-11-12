"""
evaluation_set.py
Defines the structure and metrics for evaluation
"""

EVALUATION_METRICS = {
    "simple": {
        "min_faithfulness": 0.8,
        "min_answer_relevancy": 0.8,
        "min_context_precision": 0.7,
        "min_context_recall": 0.7
    },
    "medium": {
        "min_faithfulness": 0.7,
        "min_answer_relevancy": 0.7,
        "min_context_precision": 0.6,
        "min_context_recall": 0.6
    },
    "complex": {
        "min_faithfulness": 0.6,
        "min_answer_relevancy": 0.6,
        "min_context_precision": 0.5,
        "min_context_recall": 0.5
    }
}

def get_expected_metrics(complexity: str):
    """Get the expected metric thresholds for a given complexity"""
    return EVALUATION_METRICS.get(complexity, EVALUATION_METRICS["medium"])