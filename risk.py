"""
Pure business-logic layer: turning a predicted RUL into a risk level and a
plain-language recommendation. No model, no I/O — this is exactly what a
FastAPI route handler calls on every prediction response.

The thresholds are a documented assumption, not a value read from the data
(the brief is explicit: no cost or scheduling data is provided by NASA).
"""


def rul_to_risk(rul: float) -> str:
    if rul < 30:
        return 'High'
    elif rul < 70:
        return 'Medium'
    else:
        return 'Low'


def recommendation(risk: str) -> str:
    return {
        'High': 'Prioritise this engine for inspection.',
        'Medium': 'Schedule a closer check soon.',
        'Low': 'No immediate action needed.',
    }[risk]
