"""
Configuration file for API keys and other settings.
Copy this file to config.py and fill in your actual API keys.
"""

# API Keys - Replace with your actual keys
GEMINI_API_KEY = "your-gemini-api-key-here"
CLAUDE_API_KEY = "your-claude-api-key-here"
CHATGPT_API_KEY = "your-openai-api-key-here"

# API Endpoints
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Model Settings
CLAUDE_MODEL = "claude-3-opus-20240229"
CHATGPT_MODEL = "gpt-3.5-turbo"

# Google Cloud Storage
BUCKET_NAME = "your-bucket-name"
GOOGLE_CREDENTIALS = "./google-cloud-key.json"

# Data
DATA_DIR = "./user_data"
