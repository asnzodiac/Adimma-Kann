"""
Speech-to-Text Handler
Supports Malayalam and English with fallback
"""

import speech_recognition as sr
import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class STTHandler:
    """Handle speech-to-text conversion"""
    
    def __init__(self):
        """Initialize STT handler"""
        self.recognizer = sr.Recognizer()
        # Optimize for Malayalam accents
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        logger.info("STT Handler initialized")
    
    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        Tries Malayalam first, then English
        """
        try:
            # Convert OGG to WAV using ffmpeg
            wav_path = audio_path.replace('.ogg', '.wav')
            
            subprocess.run([
                'ffmpeg', '-i', audio_path, 
                '-acodec', 'pcm_s16le', 
                '-ar', '16000', 
                '-ac', '1',
                wav_path, 
                '-y'
            ], check=True, capture_output=True)
            
            # Load audio
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            
            # Try Malayalam first
            try:
                text = self.recognizer.recognize_google(
                    audio_data, 
                    language='ml-IN'
                )
                logger.info(f"Malayalam STT: {text}")
                
                # Clean up
                self._cleanup(wav_path)
                return text
                
            except sr.UnknownValueError:
                # Fallback to English (India)
                try:
                    text = self.recognizer.recognize_google(
                        audio_data, 
                        language='en-IN'
                    )
                    logger.info(f"English STT: {text}")
                    
                    # Clean up
                    self._cleanup(wav_path)
                    return text
                    
                except sr.UnknownValueError:
                    logger.warning("Could not understand audio")
                    self._cleanup(wav_path)
                    return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            return None
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
    
    def _cleanup(self, wav_path: str):
        """Clean up temporary WAV file"""
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
