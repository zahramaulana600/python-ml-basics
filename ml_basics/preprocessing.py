import math
import re
from typing import List, Tuple

def min_max_scale(data: List[float]) -> List[float]:
    """Scales a list of numbers to [0, 1] range."""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def standard_scale(data: List[float]) -> List[float]:
    """Standardizes data to mean 0 and standard deviation 1."""
    if not data:
        return []
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0] * n
    return [(x - mean) / std for x in data]

def clean_text(text: str) -> List[str]:
    """Basic NLP text cleaner: lowercase, strip punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return [token for token in text.split() if token]
