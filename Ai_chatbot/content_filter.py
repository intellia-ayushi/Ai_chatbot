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
        print("[FILTER] No years found in content. Passing through full content.")
        # Preserve existing functionality for docs without explicit years
        return content.strip(), None

    latest_year = str(max(candidates))
    print(f"[FILTER] Latest year detected: {latest_year}")

    # Try to capture the entire section starting at the first heading mentioning the latest year
    lines = [ln for ln in content.splitlines()]
    start_idx = None
    for i, ln in enumerate(lines):
        if re.search(rf'\b{latest_year}\b', ln, flags=re.IGNORECASE):
            # Prefer lines that look like a heading
            if re.search(r'(?i)company\s+leave\s+calendar', ln) or re.search(r'(?i)leave\s+policy|calendar|description', ln):
                start_idx = i
                break
            if start_idx is None:
                start_idx = i
    end_idx = None
    if start_idx is not None:
        for j in range(start_idx + 1, len(lines)):
            years_in_line = re.findall(r'20\d{2}', lines[j])
            # Stop when another explicit different 20xx year header appears
            if any(y != latest_year for y in years_in_line):
                end_idx = j
                break
        if end_idx is None:
            end_idx = len(lines)
        section_text = "\n".join(lines[start_idx:end_idx]).strip()
    else:
        section_text = ""

    # Fallback proximity-based keep if section extraction failed or too short
    if not section_text or len(section_text) < 40:
        paragraphs = [p.strip() for p in re.split(r'\n{1,}', content) if p.strip()]
        keep_indexes = set()
        for idx, para in enumerate(paragraphs):
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
                for j in (idx-5, idx-4, idx-3, idx-2, idx-1, idx+1, idx+2, idx+3, idx+4, idx+5):
                    if 0 <= j < len(paragraphs):
                        other_years_in_neighbor = re.findall(r'20\d{2}', paragraphs[j])
                        if not any(y != latest_year for y in other_years_in_neighbor):
                            keep_indexes.add(j)
        kept = [paragraphs[i] for i in sorted(keep_indexes)]
        section_text = "\n".join(kept).strip()

    filtered_content = section_text

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

    # Add a clear data marker for later filtering (only when a latest year exists)
    filtered_content = f"YEAR_{latest_year}_ONLY_DATA\n\n{filtered_content.strip()}" if latest_year else filtered_content.strip()

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


def parse_leave_allocation(filtered_content):
    """Parse leave allocation numbers into a dict {sick:int, planned:int, casual:int}."""
    if not filtered_content:
        return {}
    alloc = {}
    for lt in ('sick', 'planned', 'casual'):
        days = extract_leave_days(filtered_content, lt)
        if days:
            try:
                alloc[lt] = int(re.search(r'(\d+)', days).group(1))
            except Exception:
                pass
    return alloc


def extract_max_leave_type(filtered_content):
    """Return (type, days) for the maximum among sick/planned/casual, or (None, None)."""
    alloc = parse_leave_allocation(filtered_content)
    if not alloc:
        return None, None
    max_type = max(alloc, key=lambda k: alloc[k])
    return max_type, alloc[max_type]


WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


def extract_holiday_info(filtered_content, holiday_name, year=None):
    """Extract holiday date string and weekday for a given holiday name and optional year.

    Returns (date_str, weekday) or (None, None).
    """
    if not filtered_content or not holiday_name:
        return None, None
    name = holiday_name.strip()
    # Build regex with optional year constraint
    year_part = f"\s*{year}" if year else "\s*20\\d{2}"
    # e.g., Diwali Holiday: 20 October 2025 (Monday)
    pattern = rf"(?i){re.escape(name)}\s*Holiday\s*[:：-]?\s*(\d{1,2}\s+[A-Za-z]+{year_part})\s*\(({'|'.join([w.capitalize() for w in WEEKDAYS])})\)"
    m = re.search(pattern, filtered_content)
    if m:
        return m.group(1), m.group(2)
    # Fallback format: "Diwali – 20 October 2025 (Monday)" (no 'Holiday' word)
    pattern_alt = rf"(?i){re.escape(name)}\s*[–\-]\s*(\d{1,2}\s+[A-Za-z]+{year_part})\s*\(({'|'.join([w.capitalize() for w in WEEKDAYS])})\)"
    m_alt = re.search(pattern_alt, filtered_content)
    if m_alt:
        return m_alt.group(1), m_alt.group(2)
    # Fallback: date without weekday (both variants)
    pattern2 = rf"(?i){re.escape(name)}\s*Holiday\s*[:：-]?\s*(\d{1,2}\s+[A-Za-z]+{year_part})"
    m2 = re.search(pattern2, filtered_content)
    if m2:
        return m2.group(1), None
    pattern2b = rf"(?i){re.escape(name)}\s*[–\-]\s*(\d{1,2}\s+[A-Za-z]+{year_part})"
    m2b = re.search(pattern2b, filtered_content)
    if m2b:
        return m2b.group(1), None
    return None, None


def parse_official_holidays(text, year: str = None):
    """Parse the Official Holidays section and return a dict: {name: (date_str, weekday)}.

    - Works with lines like:
      "Official Holidays:"
      "Diwali – 20 October 2025 (Monday)"
      "Diwali Holiday: 20 October 2025 (Monday)"
    - If year is provided, prefers entries that include that year in the date.
    """
    if not text:
        return {}
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.search(r'(?i)^\s*official\s*holidays\s*[:：-]?', ln):
            start = i + 1
            break
    if start is None:
        # try to find any line that looks like a holiday entry even without heading
        start = 0
    # stop at next heading
    stop = len(lines)
    for j in range(start, len(lines)):
        if re.search(r'(?i)^(remarks|leave\s*allocation|additional\s*guidelines)\s*[:：-]?', lines[j]):
            stop = j
            break
    entries = {}
    body = lines[start:stop]
    # Join wrapped bullet lines
    merged = []
    buf = ''
    for ln in body:
        if re.match(r'^\s*[-•*]', ln) or re.match(r'^[A-Za-z]', ln):
            if buf:
                merged.append(buf.strip())
            buf = ln
        else:
            buf += ' ' + ln.strip()
    if buf:
        merged.append(buf.strip())

    # Regex patterns
    day_group = '(' + '|'.join([w.capitalize() for w in WEEKDAYS]) + ')'
    date_pat = r'(\d{1,2}\s+[A-Za-z]+\s+20\d{2})'
    # e.g., Name – 20 October 2025 (Monday) OR Name Holiday: 20 October 2025 (Monday)
    pat1 = re.compile(rf'(?i)^\s*([A-Za-z][A-Za-z\s&]+?)\s*(?:Holiday)?\s*[:–-]\s*{date_pat}\s*\({day_group}\)')
    pat2 = re.compile(rf'(?i)^\s*([A-Za-z][A-Za-z\s&]+?)\s*(?:Holiday)?\s*[:–-]\s*{date_pat}\b')

    for ln in merged:
        m = pat1.search(ln) or pat2.search(ln)
        if m:
            name = m.group(1).strip()
            date_str = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else None
            weekday = m.group(3).strip() if m.lastindex and m.lastindex >= 3 else None
            if year and date_str and (year not in date_str):
                continue
            # prefer first match per name for given year
            key = name.lower()
            if key not in entries:
                entries[key] = (date_str, weekday)
    return entries


def extract_holiday_from_question(text, question: str, year: str = None):
    """Extract a single holiday's date/weekday based on a free-form question.

    - Parses all holidays, finds which holiday name appears in the question.
    - Returns (date_str, weekday, matched_name) or (None, None, None).
    """
    holidays = parse_official_holidays(text, year)
    if not holidays:
        return None, None, None
    ql = (question or '').lower()
    matches = [name for name in holidays.keys() if name in ql]
    if len(matches) == 1:
        return (*holidays[matches[0]], matches[0])
    # If multiple or none match, but only one holiday exists in that year, return it
    if len(holidays) == 1:
        only = next(iter(holidays.keys()))
        return (*holidays[only], only)
    return None, None, None


def find_holiday_by_date(text, date_query: str, year: str = None):
    """Given a date fragment like '26 January' (and optional year), return holiday name.

    Returns the matched holiday name or None.
    """
    if not text or not date_query:
        return None
    # Normalize spaces and case
    dq = re.sub(r'\s+', ' ', date_query.strip())
    # Build year-aware pattern
    if year:
        pat = re.compile(rf'(?i)\b([A-Za-z][A-Za-z\s&]+?)\b.*?(?:Holiday)?\s*[:–-]?\s*{re.escape(dq)}\s+{year}\b')
    else:
        pat = re.compile(rf'(?i)\b([A-Za-z][A-Za-z\s&]+?)\b.*?(?:Holiday)?\s*[:–-]?\s*{re.escape(dq)}\s+20\d{2}\b')
    # Search within official holidays block first for precision
    hols_block = '\n'.join(text.splitlines())
    m = pat.search(hols_block)
    if m:
        return m.group(1).strip()
    return None


def extract_application_system(text):
    """Extract the system/portal used to apply for leave (e.g., HRMS portal)."""
    if not text:
        return None
    # Look for sentences mentioning apply/applications and a portal/system name
    m = re.search(r'(?i)apply\s+for\s+leave.*?(?:through|via|using)\s+the\s+([A-Z][A-Z]+\s+portal|[A-Za-z]+\s+portal|[A-Za-z]+\s+system)', text)
    if m:
        return m.group(1).strip()
    # Direct HRMS portal mention
    m2 = re.search(r'(?i)HRMS\s+portal', text)
    if m2:
        return 'HRMS portal'
    return None


def extract_carry_forward_days(text, year: str = None):
    """Extract the maximum number of unused leaves that can be carried forward."""
    if not text:
        return None
    pat = re.compile(r'(?i)carried\s+forward\s+(?:up\s+to\s+a\s+maximum\s+of\s+)?(\d{1,3})\s+days')
    if year:
        # Prefer matches near the specified year
        # Simple heuristic: search in a window around the year occurrence
        idx = text.lower().find(str(year))
        if idx != -1:
            window = text[max(0, idx-500): idx+500]
            m = pat.search(window)
            if m:
                return f"{m.group(1)} days"
    m2 = pat.search(text)
    if m2:
        return f"{m2.group(1)} days"
    return None


def list_official_holidays(filtered_content):
    """Return list of holiday names mentioned under Official Holidays."""
    if not filtered_content:
        return []
    holidays = []
    for name in ("Diwali", "Christmas"):
        d, _w = extract_holiday_info(filtered_content, name)
        if d:
            holidays.append(name)
    return holidays


def extract_remarks(filtered_content):
    """Extract the remarks block text."""
    if not filtered_content:
        return ""
    # Find 'Remarks:' line and capture until next double newline or section heading
    parts = re.split(r'(?im)^\s*remarks\s*[:：-]?\s*', filtered_content)
    if len(parts) < 2:
        return ""
    tail = parts[1]
    # Stop at next heading keyword or end
    split_tail = re.split(r'(?im)^(official\s*holidays|leave\s*allocation)\s*[:：-]?', tail)
    remarks = split_tail[0].strip()
    return remarks


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
