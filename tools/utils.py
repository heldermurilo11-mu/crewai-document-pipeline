import os
import re
from typing import List, Optional, Sequence


def get_configured_prefixes(prefixes: Optional[Sequence[str]] = None) -> List[str]:
    """Return configured sample code prefixes from argument or SAMPLE_CODE_PREFIXES."""
    if prefixes is not None:
        return [str(p).strip() for p in prefixes if str(p).strip()]
    env_val = os.getenv("SAMPLE_CODE_PREFIXES", "")
    return [p.strip() for p in env_val.split(",") if p.strip()]


def normalize_time(time_str: str) -> str:
    """Normalize time strings to HH:MM format.

    Examples:
    - "10h35" -> "10:35"
    - "10:35:00" -> "10:35"
    - "10.35" -> "10:35"
    - "10 35" -> "10:35"
    - "10h" -> "10:00"
    - "9:5" -> "09:05"
    """
    if not time_str or not isinstance(time_str, str):
        return ""

    # Remove unwanted characters except digits and potential separators
    clean_time = re.sub(r"[^\d:hH\.\s]", "", time_str).strip()

    # Standardize separator to ":"
    clean_time = re.sub(r"[hH\.\s]+", ":", clean_time)

    parts = clean_time.split(":")
    # Filter out empty strings from split
    parts = [p for p in parts if p]

    if not parts:
        return ""

    hour = parts[0].zfill(2)
    minute = "00"

    if len(parts) > 1:
        minute = parts[1].zfill(2)

    # Final check: only return if it looks like a valid time (H:M)
    if len(hour) == 2 and len(minute) == 2:
        return f"{hour[:2]}:{minute[:2]}"

    return time_str  # Fallback to original if logic fails


def normalize_code(
    code_str: str,
    expected_length: int = 7,
    known_prefixes: Optional[Sequence[str]] = None,
) -> str:
    """Remove '.0' suffixes and repair configurable OCR boundary artifacts in sample codes.

    Expected pattern: numeric code of length `expected_length` (default: 7 digits).
    Optional `known_prefixes` (or `SAMPLE_CODE_PREFIXES` env var, comma-separated)
    allows repairing 1-digit border noise or missing leading digits when configured.
    """
    if code_str is None:
        return ""

    s = str(code_str).strip()

    # Remove trailing .0 or ,0 (common spreadsheet/OCR artifacts)
    s = re.sub(r"[\.,]0+$", "", s)

    # Remove any non-digit characters (like table borders | or /)
    s = re.sub(r"[^\d]", "", s)

    prefixes = get_configured_prefixes(known_prefixes)

    # Strip spurious leading border digit when length is expected_length + 1
    if len(s) == expected_length + 1:
        if prefixes and s[0] in ("0", "1") and any(s[1:].startswith(p) for p in prefixes):
            s = s[1:]
        elif s.startswith("0"):
            s = s[1:]
    # Restore a missing leading digit when length is expected_length - 1 and prefix matches
    elif len(s) == expected_length - 1 and prefixes:
        for prefix in prefixes:
            if len(prefix) >= 2 and s.startswith(prefix[1:]):
                s = prefix[0] + s
                break

    return s
