# ==================== CONFIGURATION ====================

CONFIG = {
    'main_score_offset_x': 20,
    'main_score_offset_y': -12,  # Upward offset from heading
    'main_score_fontsize': 16,   # Larger for visibility
    'main_score_color': (1, 0, 0),

    'criterion_score_offset_x': -18,
    'criterion_score_offset_y': 3,
    'criterion_score_fontsize': 11,
    'criterion_score_color': (1, 0, 0),

    'underline_color': (1, 0, 0),
    'comment_color': (1, 0, 0),
    'comment_offset': 35,
    'y_tolerance': 6,       # Points of vertical tolerance for same-line detection
    'search_tolerance': 0.8,  # Levenshtein ratio for fuzzy search
    'max_anchor_words': 6,  # Max words in an anchor chunk
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "with", "that", "this", "it", "its", "from", "into",
    "will", "would", "should", "can", "could", "may", "might", "there", "therefore",
}
