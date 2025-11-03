import re

def _collect_year_candidates(text):
    """Collect year candidates including ranges like 2024-25 and FY 2024/2025."""
    years = set()
    # Simple years
    for y in re.findall(r'20\d{2}', text):
        years.add(int(y))
    # Ranges: 2024-2025, 2024-25, 2024/2025, 2024/25, FY 2024-25, etc.
    for a, b in re.findall(r'(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})', text, flags=re.IGNORECASE):
        a_int = int(a)
        # Expand 2-digit year to 4-digit within same century if needed
        if len(b) == 2:
            b_int = int(str(a_int)[:2] + b)
        else:
            b_int = int(b)
        years.add(a_int)
        years.add(b_int)
    return sorted(years)

def filter_latest_year_content(content):
    """
    Extract ONLY the section belonging to the latest year and strip older years.

    Returns:
        tuple: (filtered_content, latest_year)
    """
    candidates = _collect_year_candidates(content)
    if not candidates:
        print("[FILTER] No years found in content.")
        return "", None

    latest_year = str(max(candidates))
    print(f"[FILTER] Latest year detected: {latest_year}")

    # Split into paragraphs and keep those tied to latest_year, plus small local context
    paragraphs = [p.strip() for p in re.split(r'\n{1,}', content) if p.strip()]
    keep_indexes = set()
    for idx, para in enumerate(paragraphs):
        # Mark paragraphs that explicitly reference latest_year or an FY range whose max equals latest_year
        mentions_year = re.search(rf'\b{latest_year}\b', para) is not None
        mentions_range_to_latest = False
        for a, b in re.findall(r'(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})', para):
            a_int = int(a)
            b_int = int(b) if len(b) == 4 else int(str(a_int)[:2] + b)
            if max(a_int, b_int) == int(latest_year):
                mentions_range_to_latest = True
                break
        if mentions_year or mentions_range_to_latest:
            keep_indexes.add(idx)
            # Also keep a small context window around the paragraph if those neighbors do NOT mention other years
            for j in (idx-2, idx-1, idx+1, idx+2):
                if 0 <= j < len(paragraphs):
                    other_years_in_neighbor = re.findall(r'20\d{2}', paragraphs[j])
                    if not any(y != latest_year for y in other_years_in_neighbor):
                        keep_indexes.add(j)

    kept = [paragraphs[i] for i in sorted(keep_indexes)]
    filtered_content = "\n".join(kept).strip()

    if not filtered_content:
        # Fallback: keep any lines mentioning latest_year
        lines = [ln for ln in content.splitlines() if re.search(rf'\b{latest_year}\b', ln)]
        filtered_content = "\n".join(lines).strip()

    # Heuristic expansion: ensure full Leave Allocation block is present
    # If Casual is present but Sick/Planned missing, include the allocation section from the source
    lc_missing = (
        ('Sick' not in filtered_content or 'Planned' not in filtered_content)
        and ('Casual' in filtered_content or re.search(r'Leave\s*Allocation', filtered_content, re.IGNORECASE))
    )
    if lc_missing:
        # Walk paragraphs to find 'Leave Allocation' and capture subsequent bullet lines until next heading
        allocation_block = []
        capturing = False
        for para in paragraphs:
            if not para:
                if capturing and allocation_block and allocation_block[-1] != '':
                    allocation_block.append('')
                continue
            if not capturing and re.search(r'^\s*Leave\s*Allocation\s*[:：-]?', para, re.IGNORECASE):
                capturing = True
                allocation_block.append(para)
                continue
            if capturing:
                # Stop at next major section heading
                if re.search(r'^\s*(Official\s*Holidays|Remarks)\s*[:：-]?', para, re.IGNORECASE):
                    break
                allocation_block.append(para)
        block_text = "\n".join([ln for ln in allocation_block if ln]).strip()
        if block_text and block_text not in filtered_content:
            # Prepend to emphasize structure
            filtered_content = f"{block_text}\n\n{filtered_content}".strip()

    # Remove any explicit other years that might still appear
    for y in set(re.findall(r'20\d{2}', filtered_content)):
        if y != latest_year:
            filtered_content = re.sub(rf'\b{y}\b', '', filtered_content)

    # Clean numbered policy prefixes to avoid the model referencing multiples
    filtered_content = re.sub(r'(?i)\bPolicy\s*\d+\s*[:：-]?\s*', '', filtered_content)

    # Add a clear data marker for later filtering
    filtered_content = f"YEAR_{latest_year}_ONLY_DATA\n\n{filtered_content.strip()}"

    print(f"[FILTER] Final filtered content length: {len(filtered_content)} chars")
    return filtered_content.strip(), latest_year


def extract_leave_days(filtered_content, leave_type):
    """Extract number of days for a leave type from filtered content.

    Args:
        filtered_content: text that should already be limited to YEAR_<latest> data
        leave_type: one of 'sick', 'planned', 'casual'

    Returns:
        str like '7 days' or None if not found
    """
    if not filtered_content or not leave_type:
        return None
    lt = leave_type.strip().lower()
    # Build robust regex for the leave type and days (inline or start of line)
    pattern = rf"(?i)\b{lt}\s*leave\s*[:：-]?\s*(\d{1,3})\s*days?\b"
    m = re.search(pattern, filtered_content)
    if m:
        return f"{m.group(1)} days"
    # Secondary: allow the word 'days' to precede the number or varied punctuation
    pattern2 = rf"(?i)\b{lt}\s*leave\b[^\n\r]*?\b(\d{1,3})\s*days?\b"
    m2 = re.search(pattern2, filtered_content)
    if m2:
        return f"{m2.group(1)} days"
    return None


def detect_leave_type_from_question(question):
    """Detect leave type (sick/planned/casual) from a free-form question.

    Supports common synonyms and abbreviations (SL/PL/CL) and simple Hinglish cues.
    """
    if not question:
        return None
    q = question.lower()

    # Abbreviations
    if re.search(r'\bsl\b', q):
        return 'sick'
    if re.search(r'\bpl\b', q):
        return 'planned'
    if re.search(r'\bcl\b', q):
        return 'casual'

    # Full words and common variants
    if re.search(r'\b(sick|medical)\s*leave\b', q):
        return 'sick'
    if re.search(r'\bplanned\s*leave\b', q):
        return 'planned'
    if re.search(r'\bcasual\s*leave\b', q):
        return 'casual'

    # Minimal forms like just the type word
    if re.search(r'\bsick\b', q):
        return 'sick'
    if re.search(r'\bplanned\b', q):
        return 'planned'
    if re.search(r'\bcasual\b', q):
        return 'casual'

    return None


def get_latest_year_prompt(question, latest_year):
    """Generate a strict instruction to respond ONLY with latest-year data."""
    if not latest_year:
        print("[PROMPT] No latest year detected. Returning plain question.")
        return question

    print(f"[PROMPT] Enforcing strict {latest_year}-only rules")

    return f"""
### SYSTEM ENFORCEMENT ###
You are analyzing a document that may contain multiple policy versions across years.

CRITICAL RULES:
1. Use ONLY the content that appears after "YEAR_{latest_year}_ONLY_DATA" in the provided context.
2. Completely ignore and exclude any information from other years; do not compare or mention them.
3. Provide a single, self-contained answer for {latest_year} only.
4. If no {latest_year} information exists in the context, respond with exactly: No specific information available for {latest_year}.

OUTPUT REQUIREMENTS (for {latest_year} only):
- Show a complete leave policy summary with the following sections when available:
  - Leave Allocation: Sick Leave (days), Planned Leave (days), Casual Leave (days)
  - Official Holidays: list each with date and weekday
  - Remarks: include any advisory notes
- Do not omit any of these subsections if present in the context. If a subsection is truly missing, output "Not specified" for that subsection.
- Do NOT mention other years, do NOT merge or compare policies, and do NOT say there are multiple policies.

Answer clearly and completely.

Question: {question.strip()} (Respond strictly for {latest_year} ONLY)
"""
