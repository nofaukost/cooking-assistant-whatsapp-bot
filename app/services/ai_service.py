from openai import AsyncOpenAI
from ..core.config import settings
from typing import Dict, List, Optional
import logging
from fastapi import HTTPException
import json
from datetime import datetime
from app.db.mongodb import mongodb

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

    async def generate_general_response(
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
            
            # Add character limit instruction to system prompt
            system_prompt += "\n\nIMPORTANT: Keep your response under 1500 characters. Be concise and clear."
            
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
            ai_response = None

            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            ai_response = response.choices[0].message.content

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
                "content": str(ai_response),
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
        message: str,
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate recipe recommendations based on available ingredients and preferences"""
        try:
            system_prompt = """You are a culinary expert. Generate recipe recommendations in JSON format.
            You MUST respond with a valid JSON object only.
            
            Response MUST follow this exact structure:
            {
                "recipes": [
                    {
                        "name": "string",
                        "description": "string (max 100 chars)",
                        "ingredients": {
                            "available": ["string"],
                            "needed": ["string"]
                        },
                        "difficulty": "beginner/intermediate/advanced",
                        "cooking_time": "string (e.g. '30 minutes')",
                        "steps": [
                            "string (detailed step, max 150 chars each)",
                            "string (include measurements and timing)",
                            "string (include temperature if applicable)",
                            "string (include tips or warnings if needed)"
                        ]
                    }
                ]
            }
            
            Guidelines:
            - Keep descriptions under 100 characters
            - Make steps detailed but concise (max 150 chars each)
            - Include measurements, timing, and temperature in steps
            - Add cooking tips or warnings in steps when relevant
            - List only essential ingredients
            - Separate available and needed ingredients
            - Use simple, clear language
            - Generate 1-2 recipes maximum
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                Generate detailed recipes based on:
                Available Ingredients: {json.dumps(ingredients)}
                User Preferences: {json.dumps(preferences)}
                User Message: {message}
                Conversation History: {json.dumps(conversation_history)}
                
                Respond with a valid JSON object only."""}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                response_format={ "type": "json_object" }  # Force JSON response
            )

            # Parse and validate the response
            content = response.choices[0].message.content.strip()
            print("Raw recipe response:", content)
            
            try:
                recipes = json.loads(content)
                
                # Validate required structure
                if "recipes" not in recipes or not isinstance(recipes["recipes"], list):
                    raise ValueError("Response must contain 'recipes' array")
                
                # Validate each recipe
                for recipe in recipes["recipes"]:
                    required_fields = ["name", "description", "ingredients", "difficulty", 
                                     "cooking_time", "steps"]
                    if not all(field in recipe for field in required_fields):
                        raise ValueError(f"Recipe missing required fields: {required_fields}")
                    
                    # Validate ingredients structure
                    if not isinstance(recipe["ingredients"], dict) or \
                       "available" not in recipe["ingredients"] or \
                       "needed" not in recipe["ingredients"]:
                        raise ValueError("Recipe ingredients must have 'available' and 'needed' lists")
                    
                    # Validate arrays
                    if not isinstance(recipe["steps"], list):
                        raise ValueError("Recipe steps must be an array")
                    
                    # Validate string lengths
                    if len(recipe["description"]) > 100:
                        recipe["description"] = recipe["description"][:97] + "..."
                    
                    # Truncate long steps but keep more detail (150 chars)
                    recipe["steps"] = [step[:147] + "..." if len(step) > 150 else step 
                                     for step in recipe["steps"]]
                

                conversation_history.append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow().isoformat()
                })

                conversation_history.append({
                    "role": "assistant",
                    "content": str(recipes),
                    "timestamp": datetime.utcnow().isoformat()
                })

                return {
                    "recommendations": recipes,
                    "timestamp": datetime.utcnow().isoformat()
                }

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {e}")
                raise ValueError(f"Invalid JSON response: {content}")
                
        except Exception as e:
            logger.error(f"Error generating recipe recommendations: {e}")
            # Return a valid JSON response for error case
            return {
                "recommendations": {
                    "recipes": [{
                        "name": "Error generating recipes",
                        "description": "Please try again later",
                        "ingredients": {
                            "available": [],
                            "needed": []
                        },
                        "difficulty": "beginner",
                        "cooking_time": "0 minutes",
                        "steps": ["Try again later"]
                    }]
                },
                "timestamp": datetime.utcnow().isoformat()
            }

    async def classify_message(self, message: str, user_preferences: Dict) -> Dict:
        """Classify user message to determine the appropriate action"""
        system_prompt = """You are a message classifier for a cooking assistant WhatsApp bot. 
        Analyze the user's message and determine the appropriate action to take.
        You MUST respond in valid JSON format only.
        
        Possible actions are:
        1. general_conversation - General cooking-related conversation
        2. get_recipes - User wants recipe recommendations
        
        Response MUST be a valid JSON object with this exact structure:
        {
            "action": "action_type",
            }
        }"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            User Preferences: {json.dumps(user_preferences)}
            User Message: {message}
            
            Classify this message and respond with a valid JSON object only."""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=150,
                response_format={ "type": "json_object" }  # Force JSON response
            )

            # Parse and validate the response
            content = response.choices[0].message.content.strip()
            print("Raw response:", content)
            
            # Ensure the response is valid JSON
            try:
                classification = json.loads(content)
                
                # Validate required fields
                required_fields = ["action"]
                if not all(field in classification for field in required_fields):
                    raise ValueError("Missing required fields in classification")
                
                # Validate action type
                valid_actions = ["get_recipes", "general_conversation"]
                if classification["action"] not in valid_actions:
                    raise ValueError(f"Invalid action type: {classification['action']}")
                
                
                print("Validated classification:", classification)
                return classification
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {e}")
                raise ValueError(f"Invalid JSON response: {content}")
                
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            # Return a valid JSON response for error case
            return {
                "action": "general_conversation",
            }

    async def process_classified_message(
        self,
        classification: Dict,
        user: Dict,
        message: str
    ) -> Dict:
        """Process the message based on its classification"""
        action = classification["action"]

        if action == "get_recipes":
            # Generate recipe recommendations
            recommendations = await self.generate_recipe_recommendations(
                ingredients=user["kitchen_inventory"]["ingredients"],
                preferences=user["preferences"],
                message=message,
                conversation_history=user["conversation_history"]
            )
            
            # Format recommendations for WhatsApp with character limit
            response = "🍳 Recipe suggestions:\n\n"
            total_length = len(response)
            recipes_to_show = []
            
            for recipe in recommendations["recommendations"]["recipes"]:
                # Format ingredients lists
                available_ingredients = "\n    • ".join([""] + recipe["ingredients"]["available"])
                needed_ingredients = "\n    • ".join([""] + recipe["ingredients"]["needed"])
                
                # Format steps with numbers
                steps = "\n    ".join([f"{i+1}. {step}" for i, step in enumerate(recipe["steps"])])
                
                recipe_text = (
                    f"📌 {recipe['name']}\n"
                    f"📝 {recipe['description']}\n\n"
                    f"🛒 Ingredients:\n"
                    f"  Available:{available_ingredients}\n"
                    f"  Needed:{needed_ingredients}\n\n"
                    f"⏱️  Time: {recipe['cooking_time']}\n"
                    f"⭐ Difficulty: {recipe['difficulty']}\n\n"
                    f"📋 Steps:\n    {steps}\n\n"
                    f"{'─' * 30}\n\n"  # Separator between recipes
                )
                
                if total_length + len(recipe_text) > 1400:  # Leave room for "more recipes available" message
                    break
                    
                recipes_to_show.append(recipe_text)
                total_length += len(recipe_text)
            
            response += "".join(recipes_to_show)
            
            # Add note if not all recipes are shown
            if len(recipes_to_show) < len(recommendations["recommendations"]["recipes"]):
                remaining = len(recommendations["recommendations"]["recipes"]) - len(recipes_to_show)
                response += f"\n...and {remaining} more recipe{'s' if remaining > 1 else ''} available. Ask for more!"
            
            return {
                "response": response,
                "conversation_history": user.get("conversation_history", [])
            }

        else:
            # Handle general conversation with character limit
            response = await self.generate_general_response(
                message=message,
                conversation_history=user.get("conversation_history", []),
                personality=user["preferences"]["chef_personality"],
                user_preferences=user["preferences"],
                available_ingredients=user["kitchen_inventory"]["ingredients"]
            )
            
            # Ensure response is under limit
            if len(response["response"]) > 1500:
                response["response"] = response["response"][:1497] + "..."
            
            return response

# Create AI service instance
ai_service = AIService() 