"""
Text-to-Speech Handler
Multiple voices including Jarvis-style, Sarvam AI for Malayalam.
Falls back: edge-tts → gTTS on failure.

Fix: Voice preferences are stored per language slot (EN vs ML) so the voice
always matches the language of the text — no more garbled audio from feeding
Malayalam into an English voice model or vice versa.
"""

import asyncio
import base64
import hashlib
import io
import logging
import os
from pathlib import Path

import edge_tts
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# VOICE CATALOGUE
# Format: "key": (display_name, engine, voice_id, language_tag)
# language_tag is "en" or "ml" — determines which slot this voice belongs to.
# ============================================================================

VOICE_CATALOGUE = {
    # ── English ──────────────────────────────────────────────────────────
    "jarvis":           ("🤖 Jarvis (Deep Male EN)",        "edge",   "en-US-GuyNeural",      "en"),
    "ryan":             ("🎩 Ryan (British Male EN)",        "edge",   "en-GB-RyanNeural",     "en"),
    "davis":            ("🎙️ Davis (Male EN)",               "edge",   "en-US-DavisNeural",    "en"),
    "tony":             ("🦾 Tony (Male EN)",                "edge",   "en-US-TonyNeural",     "en"),
    "aria":             ("🌸 Aria (Female EN)",              "edge",   "en-US-AriaNeural",     "en"),
    "jenny":            ("💁 Jenny (Female EN)",             "edge",   "en-US-JennyNeural",    "en"),
    "neerja":           ("🇮🇳 Neerja (Indian Female EN)",    "edge",   "en-IN-NeerjaNeural",   "en"),
    "prabhat":          ("🇮🇳 Prabhat (Indian Male EN)",     "edge",   "en-IN-PrabhatNeural",  "en"),

    # ── Malayalam (Edge TTS) ─────────────────────────────────────────────
    "sobhana":          ("🌺 Sobhana (ML Female)",           "edge",   "ml-IN-SobhanaNeural",  "ml"),
    "midhun":           ("🎤 Midhun (ML Male)",              "edge",   "ml-IN-MidhunNeural",   "ml"),

    # ── Malayalam (Sarvam AI) ─────────────────────────────────────────────
    "sarvam_anushka":   ("✨ Sarvam Anushka (ML Female)",    "sarvam", "anushka",              "ml"),
    "sarvam_arvind":    ("💪 Sarvam Arvind (ML Male)",       "sarvam", "arvind",               "ml"),
    "sarvam_neel":      ("🎵 Sarvam Neel (ML Male)",         "sarvam", "neel",                 "ml"),
    "sarvam_misha":     ("🌸 Sarvam Misha (ML Female)",      "sarvam", "misha",                "ml"),
    "sarvam_amol":      ("🔊 Sarvam Amol (ML Male)",         "sarvam", "amol",                 "ml"),
    "sarvam_diya":      ("🌟 Sarvam Diya (ML Female)",       "sarvam", "diya",                 "ml"),
}

# Voices that belong to each language slot.
# "manglish" is treated as English for TTS purposes.
_SLOT_FOR_LANG = {
    "en":       "en",
    "manglish": "en",
    "ml":       "ml",
}

# Default voice key per slot — used when the user hasn't set a preference.
DEFAULT_VOICE_PER_SLOT = {
    "en": "jarvis",
    "ml": "sarvam_anushka",
}


def _slot(language: str) -> str:
    """Map a detected language string to a voice slot ('en' or 'ml')."""
    return _SLOT_FOR_LANG.get(language, "en")


class TTSHandler:
    """
    Handle text-to-speech with per-language voice slots.

    _user_voices[chat_id] is now a dict:  {"en": "jarvis", "ml": "sarvam_arvind"}
    Setting /voice jarvis  → writes the EN slot.
    Setting /voice midhun  → writes the ML slot.
    Both slots are independent so the bot automatically picks the right voice
    based on the language it detected in each message.
    """

    CACHE_DIR = Path('tts_cache')
    SARVAM_API_KEY = os.getenv('SARVAM_API_KEY', '')

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        # chat_id → {"en": voice_key, "ml": voice_key}
        self._user_voices: dict[int, dict[str, str]] = {}
        logger.info("TTS Handler initialised")

    # ── Voice management ─────────────────────────────────────────────────

    def set_voice(self, chat_id: int, voice_key: str) -> bool:
        """
        Store the voice preference in the correct language slot for this chat.
        Returns False if voice_key is not in the catalogue.
        """
        if voice_key not in VOICE_CATALOGUE:
            return False

        voice_lang_slot = VOICE_CATALOGUE[voice_key][3]  # "en" or "ml"

        if chat_id not in self._user_voices:
            self._user_voices[chat_id] = {}

        self._user_voices[chat_id][voice_lang_slot] = voice_key
        logger.info(
            f"Voice slot '{voice_lang_slot}' set to '{voice_key}' for chat {chat_id}"
        )
        return True

    def get_voice_key(self, chat_id: int, language: str) -> str:
        """
        Return the best voice key for this chat + detected language.
        Looks up the correct slot (en/ml) so we never feed text to the
        wrong language model.
        """
        slot = _slot(language)
        user_prefs = self._user_voices.get(chat_id, {})
        return user_prefs.get(slot, DEFAULT_VOICE_PER_SLOT[slot])

    def get_current_voice_name(self, chat_id: int, language: str) -> str:
        key = self.get_voice_key(chat_id, language)
        return VOICE_CATALOGUE[key][0]

    def get_voice_menu(self) -> str:
        lines = [
            "🎙️ *Available Voices*\n",
            "_Each voice belongs to a language slot._",
            "_EN voices fire for English/Manglish; ML voices for Malayalam._",
            "_Both slots are saved independently — set one for each!_\n",
        ]
        sections = [
            ("🇬🇧🇺🇸 English / Manglish slot", ["jarvis", "ryan", "davis", "tony",
                                                   "aria", "jenny", "neerja", "prabhat"]),
            ("🇮🇳 Malayalam slot — Edge TTS",   ["sobhana", "midhun"]),
            ("✨ Malayalam slot — Sarvam AI",    ["sarvam_anushka", "sarvam_arvind",
                                                   "sarvam_neel", "sarvam_misha",
                                                   "sarvam_amol", "sarvam_diya"]),
        ]
        for section_title, keys in sections:
            lines.append(f"\n*{section_title}*")
            for k in keys:
                name = VOICE_CATALOGUE[k][0]
                lines.append(f"  `{k}` — {name}")
        lines.append("\n*Usage:*")
        lines.append("`/voice jarvis`        — English voice → Jarvis")
        lines.append("`/voice sarvam_arvind` — Malayalam voice → Arvind")
        lines.append("_Both slots stay set independently until you change them._")
        return "\n".join(lines)

    # ── Main generate ────────────────────────────────────────────────────

    async def generate_speech(self, text: str, language: str = 'en',
                               chat_id: int = 0):
        """
        Generate speech for `text` in `language`.
        Always picks a voice whose language tag matches the detected language,
        so we never send Malayalam text to an English voice model.

        Returns path to mp3 file, or None on total failure (text-only fallback).
        """
        voice_key = self.get_voice_key(chat_id, language)
        display_name, engine, voice_id, voice_lang = VOICE_CATALOGUE[voice_key]

        # Safety guard: if somehow the voice slot language doesn't match the
        # text language, fall back to the appropriate default so we never
        # feed Malayalam text into an English voice or vice versa.
        text_slot = _slot(language)
        if voice_lang != text_slot:
            safe_key  = DEFAULT_VOICE_PER_SLOT[text_slot]
            logger.warning(
                f"Voice '{voice_key}' (lang={voice_lang}) mismatches text lang "
                f"'{language}' — falling back to '{safe_key}'"
            )
            voice_key   = safe_key
            display_name, engine, voice_id, voice_lang = VOICE_CATALOGUE[voice_key]

        cache_key  = hashlib.md5(f"{voice_key}|{text}".encode()).hexdigest()
        cache_path = self.CACHE_DIR / f"{cache_key}.mp3"

        if cache_path.exists():
            logger.info(f"TTS cache hit ({voice_key})")
            return str(cache_path)

        logger.info(f"Generating TTS: voice='{voice_key}' engine='{engine}' lang='{language}'")

        success = False
        if engine == "sarvam":
            success = await self._sarvam_tts(text, voice_id, cache_path)
            if not success:
                logger.warning("Sarvam failed, trying edge-tts Malayalam fallback")
                success = await self._edge_tts(text, "ml-IN-MidhunNeural", cache_path)
        else:
            success = await self._edge_tts(text, voice_id, cache_path)

        if not success:
            logger.warning("edge-tts failed, trying gTTS last resort")
            success = self._gtts_fallback(text, language, cache_path)

        if success and cache_path.exists():
            return str(cache_path)

        logger.error("All TTS engines failed — text-only response")
        return None

    # ── Engines ──────────────────────────────────────────────────────────

    async def _edge_tts(self, text: str, voice: str, out: Path) -> bool:
        """edge-tts with 3 retries and back-off."""
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(out))
                logger.info(f"edge-tts OK (voice={voice}, attempt={attempt+1})")
                return True
            except Exception as e:
                logger.warning(f"edge-tts attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.5)
        return False

    async def _sarvam_tts(self, text: str, speaker: str, out: Path) -> bool:
        """
        Sarvam AI TTS — high-quality Malayalam neural voices.
        https://docs.sarvam.ai/api-reference-docs/text-to-speech
        """
        if not self.SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not configured")
            return False
        try:
            payload = {
                "inputs": [text],
                "target_language_code": "ml-IN",
                "speaker": speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.5,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v1",
            }
            headers = {
                "api-subscription-key": self.SARVAM_API_KEY,
                "Content-Type": "application/json",
            }
            resp = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                json=payload,
                headers=headers,
                timeout=20,
            )
            if resp.status_code != 200:
                logger.error(f"Sarvam TTS {resp.status_code}: {resp.text[:200]}")
                return False

            audio_b64  = resp.json()["audios"][0]
            audio_bytes = base64.b64decode(audio_b64)

            # Convert WAV → MP3
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                seg.export(str(out), format="mp3")
            except Exception:
                # pydub unavailable — raw WAV works fine in Telegram
                out.write_bytes(audio_bytes)

            logger.info(f"Sarvam TTS OK (speaker={speaker})")
            return True

        except Exception as e:
            logger.error(f"Sarvam TTS exception: {e}")
            return False

    def _gtts_fallback(self, text: str, language: str, out: Path) -> bool:
        """Last-resort Google TTS fallback."""
        lang_map = {"en": "en", "ml": "ml", "manglish": "en"}
        try:
            from gtts import gTTS
            gTTS(text=text, lang=lang_map.get(language, "en"), slow=False).save(str(out))
            logger.info("gTTS fallback OK")
            return True
        except ImportError:
            logger.error("gTTS not installed — add 'gTTS' to requirements.txt")
        except Exception as e:
            logger.error(f"gTTS fallback failed: {e}")
        return False
