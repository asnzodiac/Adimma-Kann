"""
Text-to-Speech Handler using edge-tts
Supports English and Malayalam with caching and retry logic.
Falls back to gTTS if edge-tts keeps failing (e.g. Render IP blocked by Microsoft).
"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import edge_tts

logger = logging.getLogger(__name__)


class TTSHandler:
    """Handle text-to-speech conversion"""

    VOICES = {
        'en': 'en-GB-RyanNeural',
        'ml': 'ml-IN-SobhanaNeural',
        'manglish': 'en-IN-NeerjaNeural',
    }

    # gTTS language codes as fallback
    GTTS_LANG = {
        'en': 'en',
        'ml': 'ml',
        'manglish': 'en',
    }

    CACHE_DIR = Path('tts_cache')

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        logger.info("TTS Handler initialized")

    def _get_cache_path(self, text: str, language: str) -> Path:
        text_hash = hashlib.md5(f"{text}_{language}".encode()).hexdigest()
        return self.CACHE_DIR / f"{text_hash}.mp3"

    async def _try_edge_tts(self, text: str, language: str, cache_path: Path) -> bool:
        """
        Try generating speech with edge-tts.
        Retries up to 3 times with a short delay.
        Returns True on success, False on failure.
        """
        voice = self.VOICES.get(language, self.VOICES['en'])
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(cache_path))
                logger.info(f"edge-tts succeeded on attempt {attempt + 1}")
                return True
            except Exception as e:
                logger.warning(f"edge-tts attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.5)
        return False

    def _try_gtts_fallback(self, text: str, language: str, cache_path: Path) -> bool:
        """
        Fallback: generate speech with gTTS (Google TTS).
        Returns True on success, False on failure.
        """
        try:
            from gtts import gTTS
            lang_code = self.GTTS_LANG.get(language, 'en')
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(str(cache_path))
            logger.info(f"gTTS fallback succeeded for language: {language}")
            return True
        except ImportError:
            logger.error("gTTS not installed. Add 'gTTS' to requirements.txt for fallback TTS.")
            return False
        except Exception as e:
            logger.error(f"gTTS fallback failed: {e}")
            return False

    async def generate_speech(self, text: str, language: str = 'en') -> str:
        """
        Generate speech from text.
        Tries edge-tts first (3 attempts), falls back to gTTS, returns None if both fail.
        Returns: path to audio file or None
        """
        try:
            cache_path = self._get_cache_path(text, language)

            if cache_path.exists():
                logger.info(f"Using cached TTS for: {text[:30]}...")
                return str(cache_path)

            # Try edge-tts first
            success = await self._try_edge_tts(text, language, cache_path)

            # Fall back to gTTS if edge-tts failed
            if not success:
                logger.warning("edge-tts failed, trying gTTS fallback...")
                success = self._try_gtts_fallback(text, language, cache_path)

            if success and cache_path.exists():
                logger.info(f"TTS generated and cached: {cache_path}")
                return str(cache_path)

            logger.error("Both edge-tts and gTTS failed. Returning text-only response.")
            return None

        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return None
