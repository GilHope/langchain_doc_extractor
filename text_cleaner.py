import re

def clean_extracted_text(text):
    # Remove repeated copyright notices.
    text = re.sub(r'©\d{4}\s*Palantir Technologies Inc\.', '', text)
    # Remove the word "Appendix" if it appears standalone.
    text = re.sub(r'\bAppendix\b', '', text)
    # Optionally, remove other boilerplate patterns.
    return text
