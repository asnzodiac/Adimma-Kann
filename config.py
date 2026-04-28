"""
Configuration management
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Telegram
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    OWNER_ID = 733340342
    
    # Groq AI
    GROQ_API_KEYS = os.getenv('GROQ_API_KEYS', '').split(',')
    GROQ_MODEL = 'llama-3.3-70b-versatile'
    
    # Webhook
    WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
    PORT = int(os.getenv('PORT', 8443))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    CHARACTER_FILE = BASE_DIR / 'character.txt'
    TTS_CACHE_DIR = BASE_DIR / 'tts_cache'
    
    # Conversation
    MAX_HISTORY = 20
    
    # Media limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_PDF_PAGES = 10
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.BOT_TOKEN:
            raise ValueError("BOT_TOKEN is required")
        if not cls.GROQ_API_KEYS or cls.GROQ_API_KEYS == ['']:
            raise ValueError("GROQ_API_KEYS is required")
        if not cls.WEBHOOK_URL:
            raise ValueError("WEBHOOK_URL is required")
