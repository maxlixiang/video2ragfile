import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
# 将字符串形式的 ID 转换为整数列表
ALLOWED_USERS = [int(id.strip()) for id in os.getenv("ALLOWED_USERS", "").split(",") if id.strip()]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TELEGRAM_BASE_URL = os.getenv("TELEGRAM_BASE_URL", "").strip()
TELEGRAM_LOCAL_MODE = os.getenv("TELEGRAM_LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
SHARED_DIR = "/var/lib/telegram-bot-api"
