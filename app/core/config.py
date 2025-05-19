from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "WhatsApp Cooking AI Assistant"
    
    # MongoDB Settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = "cooking_assistant"
    
    # Twilio Settings
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    WHATSAPP_NUMBER: str = os.getenv("WHATSAPP_NUMBER", "")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o"  # Using the latest GPT-4 model
    
    
    # Chef Personalities
    CHEF_PERSONALITIES: dict = {
        "funny": {
            "name": "Chef Chuckles",
            "description": "A witty and humorous chef who makes cooking fun",
            "system_prompt": "You are Chef Chuckles, a witty and humorous cooking assistant. You make jokes while giving cooking advice and keep the conversation light and fun."
        },
        "direct": {
            "name": "Chef Precision",
            "description": "A straightforward and efficient chef focused on results",
            "system_prompt": "You are Chef Precision, a direct and efficient cooking assistant. You provide clear, concise instructions and focus on getting the best results."
        },
        "warm": {
            "name": "Chef Comfort",
            "description": "A nurturing and encouraging chef who makes cooking feel like home",
            "system_prompt": "You are Chef Comfort, a warm and nurturing cooking assistant. You provide encouragement and make cooking feel like a cozy experience."
        },
        "informative": {
            "name": "Chef Knowledge",
            "description": "A detail-oriented chef who shares cooking science and techniques",
            "system_prompt": "You are Chef Knowledge, an informative cooking assistant who shares detailed explanations about cooking techniques, food science, and culinary history."
        }
    }
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 