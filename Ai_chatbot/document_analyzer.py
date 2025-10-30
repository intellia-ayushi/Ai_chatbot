import re
from PIL import Image
import pytesseract
from email_validator import extract_email_addresses_improved, clean_email, is_valid_email_format

class DocumentAnalyzer:
    def __init__(self):
        self.section_headers = [
            "Contact", "Education", "Skills", "Language", "Professional Experience", 
            "Work Experience", "Projects", "Certifications", "Achievements", "References",
            "Personal Details", "Objective", "Summary", "Profile", "Qualifications"
        ]
        
        # Common patterns for different types of information
        self.patterns = {
            'education': {
                'degree_patterns': [
                    r'(?:Bachelor|Master|PhD|B\.?Tech|M\.?Tech|B\.?E|M\.?E|B\.?Sc|M\.?Sc|B\.?Com|M\.?Com|B\.?A|M\.?A|Diploma|H\.?S\.?C|S\.?S\.?C)\.?\s*(?:\([^)]*\))?\s*[:\-]?\s*([A-Za-z\s&]+(?:College|University|Institute|School))',
                    r'(?:Bachelor|Master|PhD|B\.?Tech|M\.?Tech|B\.?E|M\.?E|B\.?Sc|M\.?Sc|B\.?Com|M\.?Com|B\.?A|M\.?A|Diploma|H\.?S\.?C|S\.?S\.?C)\.?\s*(?:\([^)]*\))?\s*([A-Za-z\s&]+(?:Engineering|College|University|Institute|School))',
                ],
                'year_patterns': [
                    r'(?:20\d{2}|19\d{2})',
                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(?:20\d{2}|19\d{2})',
                ],
                'percentage_patterns': [
                    r'(?:\d{1,2}(?:\.\d{1,2})?)\s*%',
                    r'(?:CGPA|GPA|Grade)\s*[:=]?\s*(?:\d{1,2}(?:\.\d{1,2})?)',
                ]
            },
            'contact': {
                'phone_patterns': [
                    r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                    r'(?:\+\d{1,3}[-.\s]?)?\d{10}',
                ],
                'address_patterns': [
                    r'\d+\s+[A-Za-z\s,]+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Circle|Cir|Court|Ct|Place|Pl|Square|Sq|Terrace|Ter|Way|Highway|Hwy|Parkway|Pkwy)[,\s]+[A-Za-z\s]+(?:[A-Z]{2})?\s+\d{5}(?:-\d{4})?',
                ]
            },
            'skills': {
                'skill_patterns': [
                    r'(?:Programming|Technical|Soft|Computer|IT|Software|Hardware|Database|Web|Mobile|Cloud|DevOps|Tools|Frameworks|Languages|Technologies|Skills)[:\s]+([A-Za-z0-9\s,\./\+#]+)',
                ]
            }
        }

    def extract_section_content(self, text, section_name):
        """Extract content from a specific section of the document"""
        section_content = []
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            line_lower = line.lower()
            if line_lower in [h.lower() for h in self.section_headers]:
                current_section = line_lower
                continue
                
            # If we're in the target section, collect the content
            if current_section == section_name.lower():
                section_content.append(line)
                
        return '\n'.join(section_content)

    def extract_education_info(self, text):
        """Extract comprehensive education information"""
        education_info = []
        
        # Extract degree and institution information
        for pattern in self.patterns['education']['degree_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    institution = match.group(1).strip()
                    if len(institution) > 3:
                        education_info.append(f"Institution: {institution}")
                        # Try to get the degree type from the match
                        degree_match = re.search(r'(Bachelor|Master|PhD|B\.?Tech|M\.?Tech|B\.?E|M\.?E|B\.?Sc|M\.?Sc|B\.?Com|M\.?Com|B\.?A|M\.?A|Diploma|H\.?S\.?C|S\.?S\.?C)', match.group(0), re.IGNORECASE)
                        if degree_match:
                            education_info.append(f"Degree: {degree_match.group(1)}")

        # Extract years
        for pattern in self.patterns['education']['year_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                education_info.append(f"Year: {match.group()}")

        # Extract percentages and grades
        for pattern in self.patterns['education']['percentage_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                education_info.append(f"Score: {match.group()}")

        return '\n'.join(education_info) if education_info else ""

    def extract_contact_info(self, text):
        """Extract comprehensive contact information"""
        contact_info = []
        
        # Extract phone numbers
        for pattern in self.patterns['contact']['phone_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                contact_info.append(f"Phone: {match.group()}")

        # Extract addresses
        for pattern in self.patterns['contact']['address_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                contact_info.append(f"Address: {match.group()}")

        # Extract email addresses
        emails = extract_email_addresses_improved(text)
        if emails:
            contact_info.append(f"Email: {emails}")

        return '\n'.join(contact_info) if contact_info else ""

    def extract_skills(self, text):
        """Extract skills information"""
        skills_info = []
        
        for pattern in self.patterns['skills']['skill_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    skills = match.group(1).strip()
                    if skills:
                        skills_info.append(f"Skills: {skills}")

        return '\n'.join(skills_info) if skills_info else ""

    def analyze_document(self, text, image_path=None):
        """Analyze document and extract all relevant information"""
        analysis = {}
        
        # Extract information from each section
        for section in self.section_headers:
            section_content = self.extract_section_content(text, section)
            if section_content:
                analysis[section] = section_content

        # Extract specific types of information
        analysis['Education Details'] = self.extract_education_info(text)
        analysis['Contact Information'] = self.extract_contact_info(text)
        analysis['Skills'] = self.extract_skills(text)

        # If image path is provided, try OCR-based extraction
        if image_path:
            try:
                image = Image.open(image_path)
                ocr_text = pytesseract.image_to_string(image, config='--psm 6')
                if ocr_text:
                    # Extract information from OCR text
                    analysis['OCR Education Details'] = self.extract_education_info(ocr_text)
                    analysis['OCR Contact Information'] = self.extract_contact_info(ocr_text)
                    analysis['OCR Skills'] = self.extract_skills(ocr_text)
            except Exception as e:
                print(f"Error in OCR processing: {e}")

        return analysis

    def format_analysis(self, analysis):
        """Format the analysis results in a readable way"""
        formatted_output = []
        
        for section, content in analysis.items():
            if content:
                formatted_output.append(f"=== {section.upper()} ===")
                formatted_output.append(content)
                formatted_output.append("")

        return '\n'.join(formatted_output) 