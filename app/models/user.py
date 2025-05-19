from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class ChefPersonality(str, Enum):
    FUNNY = "funny"
    DIRECT = "direct"
    WARM = "warm"
    INFORMATIVE = "informative"

class UserPreferences(BaseModel):
    chef_personality: ChefPersonality = ChefPersonality.WARM
    dietary_restrictions: List[str] = []
    favorite_cuisines: List[str] = []
    cooking_skill_level: str = "beginner"
    spice_preference: str = "medium"
    allergies: List[str] = []

class KitchenInventory(BaseModel):
    ingredients: List[str] = []
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class User(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    phone_number: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    kitchen_inventory: KitchenInventory = Field(default_factory=KitchenInventory)
    conversation_history: List[Dict] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "user_id": "USER_20240101123456_7890",
                "phone_number": "+1234567890",
                "name": "John Doe",
                "email": "john@example.com",
                "preferences": {
                    "chef_personality": "warm",
                    "dietary_restrictions": ["vegetarian"],
                    "favorite_cuisines": ["italian", "indian"],
                    "cooking_skill_level": "intermediate",
                    "spice_preference": "medium",
                    "allergies": ["peanuts"]
                },
                "kitchen_inventory": {
                    "ingredients": ["tomatoes", "onions", "garlic"],
                    "last_updated": "2024-01-01T00:00:00Z"
                }
            }
        } 