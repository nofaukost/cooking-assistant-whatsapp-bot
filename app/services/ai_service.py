from openai import AsyncOpenAI
from ..core.config import settings
from typing import Dict, List, Optional
import logging
from fastapi import HTTPException
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.chef_personalities = settings.CHEF_PERSONALITIES

    def _get_system_prompt(self, personality: str, user_preferences: Dict) -> str:
        """Generate system prompt based on chef personality and user preferences"""
        personality_config = self.chef_personalities.get(personality, self.chef_personalities["warm"])
        base_prompt = personality_config["system_prompt"]
        
        # Add user preferences to the system prompt
        preferences_prompt = f"""
        User Preferences:
        - Cooking Skill Level: {user_preferences.get('cooking_skill_level', 'beginner')}
        - Dietary Restrictions: {', '.join(user_preferences.get('dietary_restrictions', []))}
        - Favorite Cuisines: {', '.join(user_preferences.get('favorite_cuisines', []))}
        - Spice Preference: {user_preferences.get('spice_preference', 'medium')}
        - Allergies: {', '.join(user_preferences.get('allergies', []))}
        
        Please adapt your responses according to these preferences while maintaining your {personality_config['name']} personality.
        """
        
        return base_prompt + preferences_prompt

    async def generate_response(
        self,
        message: str,
        conversation_history: List[Dict],
        personality: str,
        user_preferences: Dict,
        available_ingredients: Optional[List[str]] = None
    ) -> Dict:
        """Generate AI response based on user message and context"""
        try:
            system_prompt = self._get_system_prompt(personality, user_preferences)
            
            # Prepare conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Add available ingredients if provided
            if available_ingredients:
                messages.append({
                    "role": "system",
                    "content": f"Available ingredients: {', '.join(available_ingredients)}"
                })

            # Add character limit instruction to system prompt
            messages[0]["content"] += "\n\nIMPORTANT: Keep your response under 1500 characters. If you need to provide detailed information, break it into multiple messages or be more concise."
            # Generate response with retry logic for long responses
            max_retries = 3
            retry_count = 0
            ai_response = None

            while retry_count < max_retries:
                # Generate response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

                ai_response = response.choices[0].message.content

                # Check if response is within character limit
                if len(ai_response) <= 1500:
                    break

                # If response is too long, add instruction to be more concise
                messages.append({
                    "role": "assistant",
                    "content": ai_response
                })
                messages.append({
                    "role": "user",
                    "content": "Your response was too long. Please provide a more concise version under 1500 characters."
                })
                retry_count += 1

            # If still too long after retries, truncate the response
            if len(ai_response) > 1500:
                ai_response = ai_response[:1497] + "..."

            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "response": ai_response,
                "conversation_history": conversation_history
            }

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate AI response")

    async def analyze_image(self, image_url: str) -> Dict:
        """Analyze an image using OpenAI's vision capabilities"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this image and identify any food items, ingredients, or cooking-related elements. Provide a detailed description."
                            },
                            {
                                "type": "image_url",
                                "image_url": image_url
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            return {
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise HTTPException(status_code=500, detail="Failed to analyze image")

    async def generate_recipe_recommendations(
        self,
        ingredients: List[str],
        preferences: Dict,
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate recipe recommendations based on available ingredients and preferences"""
        try:
            system_prompt = f"""
            You are a culinary expert. Based on the following information, suggest 3 recipes:
            
            Available Ingredients: {', '.join(ingredients)}
            User Preferences:
            - Cooking Skill Level: {preferences.get('cooking_skill_level', 'beginner')}
            - Dietary Restrictions: {', '.join(preferences.get('dietary_restrictions', []))}
            - Favorite Cuisines: {', '.join(preferences.get('favorite_cuisines', []))}
            - Spice Preference: {preferences.get('spice_preference', 'medium')}
            
            For each recipe, provide:
            1. Recipe name
            2. Brief description
            3. Required ingredients (including what's available and what needs to be purchased)
            4. Difficulty level
            5. Estimated cooking time
            6. Basic steps
            
            Format the response as a JSON object with an array of recipes.
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please suggest recipes based on the available ingredients and preferences."}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # Parse the response to ensure it's valid JSON
            try:
                recipes = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, format it manually
                recipes = {
                    "recipes": [
                        {
                            "name": "Recipe from AI",
                            "description": response.choices[0].message.content,
                            "ingredients": ingredients,
                            "difficulty": "medium",
                            "cooking_time": "30 minutes",
                            "steps": ["Follow the instructions in the description"]
                        }
                    ]
                }

            return {
                "recommendations": recipes,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating recipe recommendations: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate recipe recommendations")

# Create AI service instance
ai_service = AIService() 