def _minimum_edit_operations(hypothesis: list, reference: list) -> int:
    """
    Calculate minimum edit operations (insertions, deletions, substitutions, shifts)
    This is a simplified version focusing on basic edits without complex shift operations
    """
    if not hypothesis:
        return len(reference)
    if not reference:
        return len(hypothesis)
    
    m, n = len(hypothesis), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hypothesis[i-1] == reference[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )
    
    return dp[m][n]

def ter_score(hypothesis: str, reference: str) -> float:
    """
    Calculate TER (Translation Error Rate) score
    
    Args:
        hypothesis: Generated translation
        reference: Ground truth translation
    
    Returns:
        TER score (0.0 = perfect, higher = more errors)
    """
    hyp_tokens = hypothesis.strip().split()
    ref_tokens = reference.strip().split()
    
    if not ref_tokens:
        return 1.0 if hyp_tokens else 0.0
    
    edits = _minimum_edit_operations(hyp_tokens, ref_tokens)
    return edits / len(ref_tokens) 