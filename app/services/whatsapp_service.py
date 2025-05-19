from twilio.rest import Client
from twilio.request_validator import RequestValidator
from ..core.config import settings
from typing import Dict, Optional, List
import logging
from fastapi import HTTPException
import json

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self):
        self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
        self.whatsapp_number = settings.WHATSAPP_NUMBER

    def validate_request(self, signature: str, url: str, params: Dict) -> bool:
        """Validate that the request is coming from Twilio"""
        return self.validator.validate(url, params, signature)

    async def send_text_message(self, to_number: str, message: str) -> Dict:
        """Send a text message via WhatsApp"""
        try:
            message = self.client.messages.create(
                from_=f"whatsapp:{self.whatsapp_number}",
                body=message,
                to=f"whatsapp:{to_number}"
            )
            return {"status": "success", "message_sid": message.sid}
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            raise HTTPException(status_code=500, detail="Failed to send WhatsApp message")

    async def send_quick_reply_buttons(self, to_number: str, message: str, buttons: List[Dict]) -> Dict:
        """Send a message with quick reply buttons"""
        try:
            # Format buttons according to Twilio's WhatsApp API
            button_list = []
            for button in buttons:
                button_list.append({
                    "type": "reply",
                    "title": button["title"],
                    "id": button["id"]
                })

            message = self.client.messages.create(
                from_=f"whatsapp:{self.whatsapp_number}",
                body=message,
                to=f"whatsapp:{to_number}",
                content_sid="HX" + json.dumps({"buttons": button_list})  # Custom content SID for buttons
            )
            return {"status": "success", "message_sid": message.sid}
        except Exception as e:
            logger.error(f"Error sending WhatsApp quick reply buttons: {e}")
            raise HTTPException(status_code=500, detail="Failed to send WhatsApp quick reply buttons")

    async def send_image(self, to_number: str, image_url: str, caption: Optional[str] = None) -> Dict:
        """Send an image via WhatsApp"""
        try:
            message = self.client.messages.create(
                from_=f"whatsapp:{self.whatsapp_number}",
                media_url=[image_url],
                body=caption if caption else "",
                to=f"whatsapp:{to_number}"
            )
            return {"status": "success", "message_sid": message.sid}
        except Exception as e:
            logger.error(f"Error sending WhatsApp image: {e}")
            raise HTTPException(status_code=500, detail="Failed to send WhatsApp image")

    def parse_incoming_message(self, message_data: Dict) -> Dict:
        """Parse incoming WhatsApp message data"""
        try:
            message_type = message_data.get("MessageType", "text")
            from_number = message_data.get("From", "").replace("whatsapp:", "")
            message_body = message_data.get("Body", "")
            media_url = message_data.get("MediaUrl0", None)
            
            return {
                "type": message_type,
                "from_number": from_number,
                "message_body": message_body,
                "media_url": media_url,
                "timestamp": message_data.get("MessageTimestamp", ""),
                "message_sid": message_data.get("MessageSid", "")
            }
        except Exception as e:
            logger.error(f"Error parsing incoming message: {e}")
            raise HTTPException(status_code=400, detail="Invalid message format")

# Create WhatsApp service instance
whatsapp_service = WhatsAppService() 