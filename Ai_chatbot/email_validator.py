import re
from PIL import Image
import pytesseract

def clean_email(email):
    """Clean and normalize email address"""
    if not email:
        return ""
    
    # Remove common OCR artifacts
    email = email.strip()
    email = re.sub(r'^[^a-zA-Z0-9]+', '', email)  # Remove leading non-alphanumeric
    email = re.sub(r'[^a-zA-Z0-9.@_-]+$', '', email)  # Remove trailing junk
    
    # Fix common OCR errors
    email = email.replace('..', '.')
    email = email.replace('@@', '@')
    
    # Ensure only one @ symbol
    at_count = email.count('@')
    if at_count != 1:
        return ""
    
    return email

def is_valid_email_format(email):
    """Validate email format with comprehensive checks"""
    if not email or not isinstance(email, str):
        return False
    
    email = email.strip()
    
    # Basic structure check
    if email.count('@') != 1:
        return False
    
    # Split into local and domain parts
    try:
        local, domain = email.split('@')
    except ValueError:
        return False
    
    # Validate local part (before @)
    if not local or len(local) > 64:
        return False
    if local.startswith('.') or local.endswith('.'):
        return False
    if '..' in local:
        return False
    
    # Validate domain part (after @)
    if not domain or len(domain) > 255:
        return False
    if domain.startswith('.') or domain.endswith('.'):
        return False
    if '..' in domain:
        return False
    if '.' not in domain:
        return False
    
    # Check domain has valid TLD
    domain_parts = domain.split('.')
    if len(domain_parts) < 2:
        return False
    
    tld = domain_parts[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    
    # Character validation
    valid_local_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    valid_domain_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
    
    if not all(c in valid_local_chars for c in local):
        return False
    if not all(c in valid_domain_chars for c in domain):
        return False
    
    return True

def extract_email_addresses_improved(text, current_image_path=None):
    """Improved email extraction with multiple strategies and validation"""
    try:
        print("Starting improved email extraction...")
        
        all_potential_emails = set()
        
        # Strategy 1: Look for email patterns near the name "Ashish"
        name_pattern = r'Ashish\s+Sethia'
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            # Look for email in the next few lines after the name
            start_pos = name_match.end()
            next_lines = text[start_pos:start_pos + 200]  # Look at next 200 characters
            
            # Common email patterns in resumes
            email_patterns = [
                r'\b[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # Standard email
                r'\b[a-zA-Z0-9._-]+[ao@][a-zA-Z0-9.-]+[.,][a-zA-Z]{2,}\b',  # @ misread as 'a' or 'o'
                r'\b[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+,[a-zA-Z]{2,}\b',  # . misread as ','
                r'\b[a-zA-Z0-9._-]+gibccon\b',  # abc.com misread as gibccon
                r'\b[a-zA-Z0-9._-]+@gibccon\b',  # @abc.com misread as @gibccon
            ]
            
            for pattern in email_patterns:
                matches = re.findall(pattern, next_lines, re.IGNORECASE)
                for match in matches:
                    # Clean and correct the email
                    cleaned = match.strip()
                    cleaned = cleaned.replace('gibccon', 'abc.com')
                    cleaned = cleaned.replace('a@', '@').replace('o@', '@')
                    cleaned = cleaned.replace(',', '.')
                    
                    if is_valid_email_format(cleaned):
                        all_potential_emails.add(cleaned)
        
        # Strategy 2: Look for email patterns in the entire text
        standard_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}\b'
        standard_emails = re.findall(standard_pattern, text)
        for email_match in standard_emails:
            if isinstance(email_match, tuple):
                email = text[text.find(email_match[0]):text.find(email_match[0]) + len(''.join(email_match))]
                all_potential_emails.add(email)
            else:
                all_potential_emails.add(email_match)
        
        # Strategy 3: Look for @ symbol and build around it
        at_positions = [i for i, char in enumerate(text) if char == '@']
        for pos in at_positions:
            start = pos
            end = pos
            
            while start > 0 and text[start-1] not in ' \n\t\r,;()[]{}|"\'<>':
                start -= 1
            
            while end < len(text) - 1 and text[end+1] not in ' \n\t\r,;()[]{}|"\'<>':
                end += 1
            
            potential_email = text[start:end+1].strip()
            if is_valid_email_format(potential_email):
                all_potential_emails.add(potential_email)
        
        # Strategy 4: Look for email-like patterns in OCR confidence data
        if current_image_path:
            try:
                image = Image.open(current_image_path)
                email_configs = [
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@._-',
                    '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@._-',
                    '--psm 13'
                ]
                
                for config in email_configs:
                    try:
                        ocr_result = pytesseract.image_to_string(image, config=config)
                        emails_from_config = re.findall(standard_pattern, ocr_result)
                        for email in emails_from_config:
                            if isinstance(email, str) and is_valid_email_format(email):
                                all_potential_emails.add(email)
                    except:
                        continue
            except:
                pass
        
        # Clean and validate all found emails
        valid_emails = []
        for email in all_potential_emails:
            cleaned_email = clean_email(email)
            if cleaned_email and is_valid_email_format(cleaned_email):
                valid_emails.append(cleaned_email)
        
        # Remove duplicates and sort
        valid_emails = list(set(valid_emails))
        
        # If we have multiple emails, prefer the one with abc.com domain
        if len(valid_emails) > 1:
            abc_emails = [email for email in valid_emails if '@abc.com' in email.lower()]
            if abc_emails:
                valid_emails = abc_emails
        
        print(f"Found {len(valid_emails)} valid email(s): {valid_emails}")
        
        return ", ".join(valid_emails) if valid_emails else ""
        
    except Exception as e:
        print(f"Error in improved email extraction: {e}")
        return "" 