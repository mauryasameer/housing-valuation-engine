import re


def sanitize_numerical_input(value: float | int | str, min_val: float = 0.0, max_val: float = 1e9) -> float:
    """Validate and sanitize a numerical input to prevent overflow or injection.

    Args:
        value: Input value.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Sanitized float.
    """
    try:
        val = float(value)
        if val < min_val:
            return float(min_val)
        if val > max_val:
            return float(max_val)
        return val
    except (ValueError, TypeError):
        return float(min_val)


def sanitize_categorical_input(value: str, allowed_categories: list[str]) -> str:
    """Validate that the category belongs to the whitelist of allowed categories.

    Args:
        value: Categorical input from user.
        allowed_categories: Whitelist of valid category names.

    Returns:
        Validated category string.
    """
    val_str = str(value).strip()
    if val_str in allowed_categories:
        return val_str
    # Fallback to the first allowed category if invalid
    return allowed_categories[0] if allowed_categories else "None"


def clean_llm_output(text: str) -> str:
    """Sanitize LLM-generated output to prevent Cross-Site Scripting (XSS)

    or malicious code rendering in HTML/Streamlit views.

    Args:
        text: LLM response text.

    Returns:
        Sanitized markdown string.
    """
    # Remove HTML script and style tags completely
    clean_text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<style.*?>.*?</style>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
    # Strip event handlers (e.g. onload, onerror, onmouseover)
    clean_text = re.sub(r"on\w+\s*=\s*\".*?\"", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"on\w+\s*=\s*'.*?'", "", clean_text, flags=re.IGNORECASE)
    return clean_text


def detect_prompt_injection(user_input: str) -> bool:
    """Check for common prompt injection patterns in user-submitted queries.

    Args:
        user_input: Text input from the user.

    Returns:
        True if prompt injection signatures are detected, False otherwise.
    """
    patterns = [
        r"ignore\s+(the\s+)?previous\s+instructions",
        r"ignore\s+above\s+instructions",
        r"bypass\s+the\s+system\s+prompt",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"override\s+system",
        r"forget\s+what\s+you\s+were\s+told",
        r"act\s+as\s+a\s+jailbroken",
        r"system\s+instructions:",
        r"do\s+anything\s+now",
    ]

    input_lower = user_input.lower()
    for pattern in patterns:
        if re.search(pattern, input_lower):
            return True
    return False
