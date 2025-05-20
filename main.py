from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from app.services.whatsapp_service import whatsapp_service
from app.services.ai_service import ai_service
from app.db.mongodb import mongodb
from app.models.user import User, UserPreferences, KitchenInventory
from typing import Optional, Dict
import logging
from datetime import datetime
import json
from bson import ObjectId
from pydantic import BaseModel
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Cooking AI Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    await mongodb.connect_to_database()

@app.on_event("shutdown")
async def shutdown_db_client():
    await mongodb.close_database_connection()

# Helper functions
async def get_user(phone_number: str) -> Optional[Dict]:
    """Get user from database or create new user if doesn't exist"""
    users_collection = mongodb.get_collection("users")
    user = await users_collection.find_one({"phone_number": phone_number})
    
    if not user:
        # Generate unique user_id using timestamp and phone number
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        user_id = f"USER_{timestamp}_{phone_number[-4:]}"  # Using last 4 digits of phone number
        
        # Create new user with proper ObjectId for _id
        new_user = User(
            user_id=user_id,  # This is our custom user_id
            phone_number=phone_number,
            preferences=UserPreferences(),
            kitchen_inventory=KitchenInventory(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert the user document
        result = await users_collection.insert_one(new_user.dict(by_alias=True))
        user = await users_collection.find_one({"_id": result.inserted_id})
        print(user)
    return user

# Request models
class WhatsAppWebhook(BaseModel):
    MessageSid: str
    From: str
    Body: str
    MessageType: str = "text"
    MediaUrl0: Optional[str] = None
    MessageTimestamp: str

class UpdatePreferences(BaseModel):
    chef_personality: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None
    favorite_cuisines: Optional[List[str]] = None
    cooking_skill_level: Optional[str] = None
    spice_preference: Optional[str] = None
    allergies: Optional[List[str]] = None

# Routes
@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    x_twilio_signature: str = Header(None)
):
    """Handle incoming WhatsApp messages"""
    # Get form data
    form_data = await request.form()
    message_data = dict(form_data)
    # print("WhatsApp webhook data:", message_data)
    
    # Validate Twilio signature
    if not whatsapp_service.validate_request(
        x_twilio_signature,
        str(request.url),
        message_data
    ):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        
    
    # Parse message for normal message handling
    parsed_message = whatsapp_service.parse_incoming_message(message_data)
    phone_number = parsed_message["from_number"]
    
    # Get or create user
    user = await get_user(phone_number)
    users_collection = mongodb.get_collection("users")
    
    try:
        # Handle different message types
        if parsed_message["type"] == "text":
            # Classify the message
            classification = await ai_service.classify_message(
                message=parsed_message["message_body"],
                user_preferences=user["preferences"]
            )
            print("classification", classification)
            
            # Process the message based on classification
            result = await ai_service.process_classified_message(
                classification=classification,
                user=user,
                message=parsed_message["message_body"]
            )
            
            # Update conversation history
            await users_collection.update_one(
                {"user_id": user["user_id"]},
                {
                    "$set": {
                        "conversation_history": result["conversation_history"],
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Send response back to user
            await whatsapp_service.send_text_message(
                to_number=phone_number,
                message=result["response"]
            )
            
        elif parsed_message["type"] == "image":
            # Analyze image
            print(parsed_message)
            image_analysis = await ai_service.analyze_image(parsed_message["media_url"])
            await whatsapp_service.send_text_message(
                to_number=phone_number,
                message=f"I've analyzed your image. Here's what I found:\n\n{image_analysis['analysis']}"
            )
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await whatsapp_service.send_text_message(
            to_number=phone_number,
            message="❌ Sorry, something went wrong. Please try again."
        )
        raise HTTPException(status_code=500, detail="Failed to process message")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 