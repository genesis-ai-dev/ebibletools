def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Minimum number of single-character edits (insertions, deletions, substitutions)
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    len1, len2 = len(s1), len(s2)
    
    # Create distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize first row and column
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )
    
    return dp[len1][len2]

def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized edit distance (0.0 = identical, 1.0 = completely different)
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Normalized edit distance between 0.0 and 1.0
    """
    if not s1 and not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    
    return edit_distance(s1, s2) / max_len

def word_edit_distance(s1: str, s2: str) -> int:
    """
    Calculate edit distance at word level
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Word-level edit distance
    """
    words1 = s1.strip().split()
    words2 = s2.strip().split()
    
    return edit_distance(' '.join(words1), ' '.join(words2))

def normalized_word_edit_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized word-level edit distance
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Normalized word-level edit distance between 0.0 and 1.0
    """
    words1 = s1.strip().split()
    words2 = s2.strip().split()
    
    if not words1 and not words2:
        return 0.0
    
    max_words = max(len(words1), len(words2))
    if max_words == 0:
        return 0.0
    
    return word_edit_distance(s1, s2) / max_words 