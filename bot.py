"""
Adimma-Kann: Voice-First Telegram Bot
A witty, sarcastic AI assistant for 'sir'
"""

import os
import asyncio
import logging
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
import random
from datetime import datetime

# Import custom utilities
from utils.language_detector import LanguageDetector
from utils.tts_handler import TTSHandler
from utils.stt_handler import STTHandler
from utils.media_processor import MediaProcessor
from utils.conversation_manager import ConversationManager

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
BOT_TOKEN = os.getenv('TELEGRAM_TOKEN') or os.getenv('BOT_TOKEN')
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
PORT = int(os.getenv('PORT', 10000))
OWNER_ID = int(os.getenv('OWNER_ID', 733340342))

def get_groq_keys():
    """Get all Groq API keys from environment"""
    keys = []
    multi_keys = os.getenv('GROQ_API_KEYS', '')
    if multi_keys:
        keys.extend([k.strip() for k in multi_keys.split(',') if k.strip()])
    for i in range(4):
        key_name = f'GROQ_API_KEY{i}' if i > 0 else 'GROQ_API_KEY'
        key = os.getenv(key_name, '')
        if key and key not in keys:
            keys.append(key.strip())
    return keys

GROQ_API_KEYS = get_groq_keys()

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_TOKEN or BOT_TOKEN environment variable is required")
if not GROQ_API_KEYS:
    raise ValueError("At least one GROQ_API_KEY is required")

logger.info(f"Loaded {len(GROQ_API_KEYS)} Groq API keys")

# ============================================================================
# PERSISTENT EVENT LOOP  ← FIX #1: single loop reused across all requests
# ============================================================================

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

app = Flask(__name__)

lang_detector = LanguageDetector()
tts_handler = TTSHandler()
stt_handler = STTHandler()
media_processor = MediaProcessor()
conversation_manager = ConversationManager()

CHARACTER_PROMPT = ""
try:
    with open('character.txt', 'r', encoding='utf-8') as f:
        CHARACTER_PROMPT = f.read().strip()
    logger.info("Character prompt loaded successfully")
except Exception as e:
    logger.error(f"Failed to load character.txt: {e}")
    CHARACTER_PROMPT = "You are Adimma Kann, a witty and sarcastic AI assistant."

bot_state = {}

# ============================================================================
# GROQ CLIENT MANAGER
# ============================================================================

class GroqClientManager:
    """Manages multiple Groq API keys with rotation"""

    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        logger.info(f"Initialized GroqClientManager with {len(self.api_keys)} keys")

    def get_client(self):
        client = Groq(api_key=self.api_keys[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return client

    def get_completion(self, messages, model="llama-3.3-70b-versatile", max_retries=3):
        for attempt in range(max_retries):
            try:
                client = self.get_client()
                response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.8,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Sorry sir, I'm having trouble thinking right now. Try again in a moment!"
                continue

groq_manager = GroqClientManager(GROQ_API_KEYS)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bot_state(chat_id):
    if chat_id not in bot_state:
        bot_state[chat_id] = {"active": True, "language": "en"}
    return bot_state[chat_id]

def should_sleep(text):
    sleep_commands = [
        "bye", "standby", "stop listening", "sleep",
        "good night", "goodnight", "നല്ല രാത്രി", "പോയി വരാം"
    ]
    text_lower = text.lower().strip()
    return any(cmd in text_lower for cmd in sleep_commands)

def should_wake(text):
    wake_commands = [
        "hi", "hello", "wake up", "adimma", "hey",
        "ഹലോ", "എണീക്ക്", "അടിമ്മ"
    ]
    text_lower = text.lower().strip()
    return any(cmd in text_lower for cmd in wake_commands)

async def notify_owner_new_user(context, user):
    try:
        message = (
            f"🆕 *New User Started Bot*\n\n"
            f"👤 Name: {user.first_name} {user.last_name or ''}\n"
            f"🆔 User ID: `{user.id}`\n"
            f"📱 Username: @{user.username or 'N/A'}\n"
            f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await context.bot.send_message(
            chat_id=OWNER_ID,
            text=message,
            parse_mode='Markdown'
        )
        logger.info(f"Notified owner about new user: {user.id}")
    except Exception as e:
        logger.error(f"Failed to notify owner: {e}")

def build_system_prompt(language="en"):
    lang_instruction = {
        "en": "Respond in English.",
        "ml": "മലയാളത്തിൽ മറുപടി നൽകുക. Respond in Malayalam.",
        "manglish": "Respond in Manglish (Romanized Malayalam mixed with English)."
    }
    instruction = lang_instruction.get(language, lang_instruction["en"])
    return f"{CHARACTER_PROMPT}\n\nIMPORTANT: {instruction}"

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    state = get_bot_state(chat_id)
    state["active"] = True

    if user.id != OWNER_ID:
        await notify_owner_new_user(context, user)

    welcome_message = (
        "🎭 *Adimma Kann at your service\\!*\n\n"
        "I'm your witty, slightly sarcastic AI assistant\\.\n\n"
        "💬 Send me:\n"
        "• Voice messages \\(English/Malayalam/Manglish\\)\n"
        "• Text messages\n"
        "• Photos\n"
        "• Documents \\(PDFs\\)\n\n"
        "🌐 I'll respond in the same language you use\\!\n\n"
        "⌨️ Commands:\n"
        "/help \\- Show instructions\n"
        "/clear \\- Clear conversation history\n\n"
        "😴 Say 'bye' or 'sleep' to put me on standby\n"
        "👋 Say 'hi' or 'wake up' to activate me again"
    )

    await update.message.reply_text(welcome_message, parse_mode='MarkdownV2')
    logger.info(f"User {user.id} started the bot")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # FIX #2: Use plain text to avoid Markdown parse errors crashing the reply
    help_text = (
        "🎭 Adimma Kann - Usage Instructions\n\n"
        "How to Use:\n"
        "1. Just talk to me! Send voice or text messages\n"
        "2. I understand English, Malayalam, and Manglish\n"
        "3. I'll reply in the same language you use\n\n"
        "Voice Messages:\n"
        "🎤 Send voice notes - I'll transcribe and respond with voice + text\n\n"
        "Text Messages:\n"
        "💬 Type anything - I'll reply with wit and sarcasm\n\n"
        "Images & Documents:\n"
        "🖼 Send photos - I'll analyze and comment\n"
        "📄 Send PDFs - I'll read and discuss them\n\n"
        "Sleep/Wake Commands:\n"
        "😴 Sleep: 'bye', 'standby', 'good night', 'sleep'\n"
        "👋 Wake: 'hi', 'hello', 'wake up', 'adimma'\n\n"
        "Commands:\n"
        "/start - Restart the bot\n"
        "/help - Show this message\n"
        "/clear - Clear conversation history\n\n"
        "Ready to chat? 🚀"
    )
    await update.message.reply_text(help_text)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_manager.clear_history(chat_id)
    await update.message.reply_text("🗑️ Conversation history cleared!\n\nStarting fresh, sir! 🎬")
    logger.info(f"Cleared conversation history for chat {chat_id}")

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    try:
        voice_file = await update.message.voice.get_file()
        voice_path = f"voice_{chat_id}_{datetime.now().timestamp()}.ogg"
        await voice_file.download_to_drive(voice_path)

        await update.message.reply_text("🎧 Listening...")
        transcription = await stt_handler.transcribe(voice_path)

        if os.path.exists(voice_path):
            os.remove(voice_path)

        if not transcription:
            await update.message.reply_text("😕 Sorry sir, couldn't hear you clearly. Try again?")
            return

        logger.info(f"Voice transcribed: {transcription}")
        await process_message(update, context, transcription)

    except Exception as e:
        logger.error(f"Voice handling error: {e}")
        await update.message.reply_text("❌ Oops! Something went wrong with the voice message.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_bot_state(chat_id)

    if not state["active"]:
        return

    try:
        await update.message.reply_text("🖼️ Analyzing image...")

        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_path = f"photo_{chat_id}_{datetime.now().timestamp()}.jpg"
        await photo_file.download_to_drive(photo_path)

        description = await media_processor.process_image(photo_path)

        if os.path.exists(photo_path):
            os.remove(photo_path)

        caption = update.message.caption or "What do you think about this image?"
        user_message = f"{caption}\n\n[Image description: {description}]"
        await process_message(update, context, user_message)

    except Exception as e:
        logger.error(f"Photo handling error: {e}")
        await update.message.reply_text("❌ Couldn't process the image, sir!")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_bot_state(chat_id)

    if not state["active"]:
        return

    try:
        document = update.message.document

        if document.file_size > 10 * 1024 * 1024:
            await update.message.reply_text("📄 File too large! Please send files under 10MB.")
            return

        await update.message.reply_text("📄 Processing document...")

        doc_file = await document.get_file()
        doc_path = f"doc_{chat_id}_{datetime.now().timestamp()}_{document.file_name}"
        await doc_file.download_to_drive(doc_path)

        content = await media_processor.process_document(doc_path, document.file_name)

        if os.path.exists(doc_path):
            os.remove(doc_path)

        if not content:
            await update.message.reply_text("😕 Couldn't read the document. Is it a valid PDF or text file?")
            return

        caption = update.message.caption or "Please analyze this document."

        if len(content) > 4000:
            content = content[:4000] + "... (truncated)"

        user_message = f"{caption}\n\n[Document content:\n{content}]"
        await process_message(update, context, user_message)

    except Exception as e:
        logger.error(f"Document handling error: {e}")
        await update.message.reply_text("❌ Couldn't process the document, sir!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    await process_message(update, context, text)

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    chat_id = update.effective_chat.id
    state = get_bot_state(chat_id)

    if should_sleep(text):
        state["active"] = False
        response = random.choice([
            "😴 Going on standby, sir. Wake me when you need me!",
            "💤 Alright, taking a power nap. Just say 'hi' when you're back!",
            "🌙 Good night, sir! Standing by...",
        ])
        await update.message.reply_text(response)
        logger.info(f"Bot sleeping for chat {chat_id}")
        return

    if should_wake(text):
        if not state["active"]:
            state["active"] = True
            response = random.choice([
                "👋 Wide awake, sir! What can I do for you?",
                "⚡ Back in action! What's up?",
                "🎯 Activated and ready, sir!",
            ])
            await update.message.reply_text(response)
            logger.info(f"Bot waking up for chat {chat_id}")
            return

    if not state["active"]:
        return

    try:
        detected_lang = await lang_detector.detect(text)
        state["language"] = detected_lang
        logger.info(f"Detected language: {detected_lang}")

        history = conversation_manager.get_history(chat_id)

        messages = [{"role": "system", "content": build_system_prompt(detected_lang)}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": text})

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        response = groq_manager.get_completion(messages)

        conversation_manager.add_message(chat_id, "user", text)
        conversation_manager.add_message(chat_id, "assistant", response)

        await update.message.reply_text(response)

        # Generate and send voice response
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")

        voice_file = await tts_handler.generate_speech(response, detected_lang)

        if voice_file and os.path.exists(voice_file):
            with open(voice_file, 'rb') as audio:
                await update.message.reply_voice(voice=audio)
            logger.info(f"Sent voice response in {detected_lang}")
        else:
            logger.info("TTS unavailable, sent text-only response")

    except Exception as e:
        logger.error(f"Message processing error: {e}")
        await update.message.reply_text("❌ Oops! My circuits got tangled. Give me a moment, sir!")

# ============================================================================
# WEBHOOK
# ============================================================================

@app.route('/')
def index():
    return "Adimma Kann is alive! 🎭", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    """Handle incoming updates — FIX #1: reuse persistent loop, never close it"""
    try:
        json_data = request.get_json(force=True)
        update = Update.de_json(json_data, application.bot)
        _loop.run_until_complete(application.process_update(update))
        return "OK", 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return "Error", 500

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

application = Application.builder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start_command))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(CommandHandler("instruction", help_command))
application.add_handler(CommandHandler("clear", clear_command))
application.add_handler(MessageHandler(filters.VOICE, handle_voice))
application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def setup_webhook():
    """Setup webhook using the persistent loop"""
    async def _setup():
        await application.initialize()
        webhook_url = f"{WEBHOOK_URL}/{BOT_TOKEN}"
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set to: {webhook_url}")

    _loop.run_until_complete(_setup())

if __name__ == '__main__':
    if WEBHOOK_URL:
        setup_webhook()
    else:
        logger.warning("No WEBHOOK_URL set - bot will not receive updates!")

    app.run(host='0.0.0.0', port=PORT)
