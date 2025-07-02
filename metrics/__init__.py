# Universal Translation Quality Metrics
# Language-agnostic metrics that work with any script or language

from .chrf import chrF_plus
from .edit_distance import edit_distance, normalized_edit_distance
from .ter import ter_score

__all__ = [
    # chrF+ (character n-gram F-score, enhanced)
    'chrF_plus',
    
    # Edit Distance (normalized Levenshtein)
    'edit_distance', 'normalized_edit_distance',
    
    # TER (Translation Error Rate)
    'ter_score',
] 