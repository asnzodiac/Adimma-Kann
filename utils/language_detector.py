"""
Language Detection Module
Detects English, Malayalam, and Manglish
"""

import re
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detect language from text"""
    
    # Malayalam Unicode range
    MALAYALAM_RANGE = r'[\u0D00-\u0D7F]'
    
    # Common Manglish words
    MANGLISH_PATTERNS = [
        r'\bentha\b', r'\bengane\b', r'\balle\b', r'\bseri\b',
        r'\bpinne\b', r'\bkandu\b', r'\bvaran\b', r'\bpoyi\b',
        r'\billa\b', r'\bundo\b', r'\baano\b', r'\bennu\b',
        r'\bcheyyam\b', r'\bcheythu\b', r'\bvenda\b', r'\bsir\b',
        r'\bmone\b', r'\bmola\b', r'\bmyre\b', r'\beda\b',
        r'\bpoda\b', r'\bpodo\b', r'\balle\b', r'\banallo\b'
    ]
    
    async def detect(self, text: str) -> str:
        """
        Detect language from text
        Returns: 'ml' (Malayalam), 'manglish', or 'en' (English)
        """
        text_lower = text.lower()
        
        # Check for Malayalam script
        malayalam_chars = len(re.findall(self.MALAYALAM_RANGE, text))
        total_chars = len(re.findall(r'\w', text))
        
        if total_chars == 0:
            return 'en'
        
        malayalam_ratio = malayalam_chars / total_chars
        
        # If more than 30% Malayalam characters, it's Malayalam
        if malayalam_ratio > 0.3:
            logger.info(f"Detected Malayalam (ratio: {malayalam_ratio:.2f})")
            return 'ml'
        
        # Check for Manglish patterns
        manglish_matches = sum(
            1 for pattern in self.MANGLISH_PATTERNS 
            if re.search(pattern, text_lower)
        )
        
        if manglish_matches >= 2:
            logger.info(f"Detected Manglish ({manglish_matches} matches)")
            return 'manglish'
        
        # Default to English
        logger.info("Detected English")
        return 'en'
