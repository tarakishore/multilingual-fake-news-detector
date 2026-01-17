"""
Data preprocessing module for Multilingual Fake News Detection.
Handles text cleaning, script normalization, and tokenization preparation.
"""

import re
import unicodedata
import logging
from typing import List, Optional, Union
import sys
import os

# Add parent directory to path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from config import PreprocessingConfig
except ImportError:
    # Fallback if config not found (e.g. running standalone)
    class PreprocessingConfig:
        REMOVE_URLS = True
        REMOVE_MENTIONS = True
        REMOVE_HASHTAGS = False
        REMOVE_EMOJIS = False
        LOWERCASE = False
        NORMALIZE_UNICODE = True
        TRANSLITERATE = True

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Handles cleaning and normalization of multilingual text.
    """
    
    def __init__(self, config=None):
        self.config = config or PreprocessingConfig
        
        # Regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        # Emoji pattern (simplified)
        self.emoji_pattern = re.compile(r'[^\w\s,.:;"\'?!-]') 
        
        # Initialize transliterator if needed
        self.transliterator = None
        if self.config.TRANSLITERATE:
            self._init_transliterator()

    def _init_transliterator(self):
        """Initialize Indic transliteration library if available."""
        try:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            self.transliterator = UnicodeIndicTransliterator
            logger.info("Indic NLP library loaded for transliteration.")
        except ImportError:
            logger.warning("indic-nlp-library not found. Transliteration will be skipped.")
            self.transliterator = None

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: URLs, mentions, whitespace.
        """
        if not text:
            return ""
            
        # Normalize unicode
        if self.config.NORMALIZE_UNICODE:
            text = unicodedata.normalize('NFKC', text)
            
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = self.url_pattern.sub('<URL>', text)
            
        # Remove User Mentions
        if self.config.REMOVE_MENTIONS:
            text = self.mention_pattern.sub('<USER>', text)
            
        # Remove Hashtags (Optional)
        if self.config.REMOVE_HASHTAGS:
            text = self.hashtag_pattern.sub('', text)
            
        # Remove Emojis/Special chars (Optional)
        if self.config.REMOVE_EMOJIS:
             # This is a broad filter, might remove non-latin scripts if not careful
             # Using a safer approach usually requires the `emoji` library
             pass

        # Lowercase (Optional - Models like XLM-R are case sensitive usually)
        if self.config.LOWERCASE:
            text = text.lower()
            
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def normalize_script(self, text: str, lang_code: str) -> str:
        """
        Handles Code-Switching by transliterating to native script.
        E.g., "main ghar ja raha hoon" (Hindi in Latin) -> "मैं घर जा रहा हूँ"
        """
        if not self.config.TRANSLITERATE or not self.transliterator:
            return text
            
        # Map common language codes to Indic NLP library codes if needed
        # This is a simplified mapping
        indic_map = {
            'hi': 'hi', 'bn': 'bn', 'ta': 'ta', 'te': 'te', 
            'mr': 'mr', 'gu': 'gu', 'kn': 'kn', 'ml': 'ml', 'pa': 'pa'
        }
        
        if lang_code in indic_map:
            try:
                # Note: This is an oversimplification. 
                # Real transliteration from Latin -> Native requires deeper models (like indic-trans).
                # The UnicodeIndicTransliterator usually handles Script-to-Script.
                # For Latin-to-Native, we would ideally use a model like LibIndic or Google's API.
                # Here we placeholder a logic check.
                pass
            except Exception as e:
                logger.error(f"Transliteration error for {lang_code}: {e}")
                
        return text

    def process(self, text: Union[str, List[str]], lang_code: str = 'en') -> Union[str, List[str]]:
        """
        Main processing pipeline.
        """
        is_list = isinstance(text, list)
        texts = text if is_list else [text]
        
        processed_texts = []
        for t in texts:
            cleaned = self.clean_text(t)
            # Apply script normalization if it's an Indic language
            normalized = self.normalize_script(cleaned, lang_code)
            processed_texts.append(normalized)
            
        return processed_texts if is_list else processed_texts[0]
