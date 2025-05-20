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
                "content": f"Available ingredients: {', '.join([item['name']+' , '+item['amount'] for item in available_ingredients])}"
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
            print(ai_response)
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
                model="gpt-4o",
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
        3. update_inventory - User wants to add or update kitchen inventory items
        4. get_inventory - User wants to view their current inventory
        5. remove_inventory - User wants to remove items from their inventory
        
        Response MUST be a valid JSON object with this exact structure:
        {
            "action": "action_type",
            "items": [  # Include for update_inventory or remove_inventory actions
                {
                    "name": "string",
                    "amount": "string (e.g., '2 kg', '500g', '3 pieces', or just '5' for pieces)"
                }
            ],
            "remove_all": boolean  # Set to true if user wants to remove all items
        }

        For inventory updates:
        - If user mentions a number without a unit (e.g., "5 tomatoes"), use "piece" as the default unit
        - If user mentions "some" or "a few", use "2 pieces" as the default amount
        - If user mentions "many" or "a lot", use "5 pieces" as the default amount
        - If user mentions "a couple", use "2 pieces" as the default amount
        - If no amount is mentioned, use "1 piece" as the default amount"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            User Preferences: {json.dumps(user_preferences)}
            User Message: {message}
            
            Classify this message and respond with a valid JSON object only.
            For update_inventory action, extract the items and their amounts mentioned in the message.
            For remove_inventory action, classify messages about removing items and extract the items and amounts to remove.
            For get_inventory action, classify messages like "show my inventory", "what ingredients do I have", etc.
            If amount is not specified for removal, remove the entire item.
            If message includes "all" or "everything" for removal, set remove_all to true.
            
            Examples:
            - "I have 2 kg of tomatoes" -> {{"action": "update_inventory", "items": [{{"name": "tomatoes", "amount": "2 kg"}}]}}
            - "I have 5 tomatoes" -> {{"action": "update_inventory", "items": [{{"name": "tomatoes", "amount": "5"}}]}}
            - "Add some onions" -> {{"action": "update_inventory", "items": [{{"name": "onions", "amount": "2"}}]}}
            - "Remove 500g of garlic" -> {{"action": "remove_inventory", "items": [{{"name": "garlic", "amount": "500g"}}]}}
            - "Remove all items" -> {{"action": "remove_inventory", "items": [], "remove_all": true}}
            - "Clear my inventory" -> {{"action": "remove_inventory", "items": [], "remove_all": true}}
            - "Show my inventory" -> {{"action": "get_inventory"}}"""}
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
                valid_actions = ["get_recipes", "general_conversation", "update_inventory", "get_inventory", "remove_inventory"]
                if classification["action"] not in valid_actions:
                    raise ValueError(f"Invalid action type: {classification['action']}")
                
                # Validate items field for update_inventory and remove_inventory
                if classification["action"] in ["update_inventory", "remove_inventory"]:
                    if "items" not in classification or not isinstance(classification["items"], list):
                        raise ValueError("Missing or invalid items field for inventory action")
                    
                    # Validate each item has name and amount if items are specified
                    for item in classification["items"]:
                        if not isinstance(item, dict) or "name" not in item or "amount" not in item:
                            raise ValueError("Each inventory item must have 'name' and 'amount' fields")
                
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

    def _validate_unit(self, unit: str) -> tuple[bool, str]:
        """Validate if a unit is supported and return its normalized form"""
        # Normalize unit (remove plural, lowercase)
        unit = unit.lower().rstrip('s')
        
        # Check if unit is in our conversion table
        if unit in self._get_unit_conversions():
            return True, unit
            
        # Try to find similar units
        similar_units = []
        for valid_unit in self._get_unit_conversions().keys():
            if unit in valid_unit or valid_unit in unit:
                similar_units.append(valid_unit)
        
        if similar_units:
            return False, f"Did you mean: {', '.join(similar_units)}?"
        return False, "Unsupported unit. Supported units are: g, kg, ml, l, piece"

    def _get_unit_conversions(self) -> Dict[str, float]:
        """Get the unit conversion factors"""
        return {
            # Weight conversions (base unit: grams)
            "g": 1,
            "gram": 1,
            "grams": 1,
            "kg": 1000,
            "kilo": 1000,
            "kilos": 1000,
            "kilogram": 1000,
            "kilograms": 1000,
            # Volume conversions (base unit: milliliters)
            "ml": 1,
            "milliliter": 1,
            "milliliters": 1,
            "l": 1000,
            "liter": 1000,
            "liters": 1000,
            # Count units
            "piece": 1,
            "pieces": 1,
            "pcs": 1,
            "pc": 1,
            "whole": 1,
            "wholes": 1
        }

    def _convert_unit(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert between common units of measurement"""
        # Get conversion factors
        unit_conversions = self._get_unit_conversions()
        
        # Normalize units
        from_unit = from_unit.lower().rstrip('s')  # Remove plural
        to_unit = to_unit.lower().rstrip('s')
        
        # Check if units are in our conversion table
        if from_unit not in unit_conversions or to_unit not in unit_conversions:
            return None
            
        # Check if units are of the same type (weight, volume, or count)
        from_type = "weight" if from_unit in ["g", "gram", "kg", "kilo", "kilogram"] else \
                   "volume" if from_unit in ["ml", "milliliter", "l", "liter"] else \
                   "count"
        to_type = "weight" if to_unit in ["g", "gram", "kg", "kilo", "kilogram"] else \
                 "volume" if to_unit in ["ml", "milliliter", "l", "liter"] else \
                 "count"
                 
        if from_type != to_type:
            return None
            
        # Convert to base unit then to target unit
        base_value = value * unit_conversions[from_unit]
        converted_value = base_value / unit_conversions[to_unit]
        
        return converted_value

    def _format_unit(self, value: float, unit: str) -> str:
        """Format the value and unit nicely"""
        # Round to 2 decimal places if needed
        if value.is_integer():
            value = int(value)
        else:
            value = round(value, 2)
            
        # # Use singular form for 1
        # if value == 1:
        #     unit = unit.rstrip('s')
        # else:
        #     # Ensure plural form
        #     if not unit.endswith('s'):
        #         unit += 's'
                
        return f"{value} {unit}"

    def _parse_amount_unit(self, amount_str: str) -> tuple[float, str]:
        """Parse a string containing amount and unit, handling cases with or without space"""
        # Remove any extra spaces
        amount_str = amount_str.strip()
        
        # If amount_str is just a number, return it with default unit "piece"
        try:
            value = float(amount_str)
            return value, "piece"
        except ValueError:
            pass
        
        # Try to find where the number ends and unit begins
        import re
        match = re.match(r'^(\d+(?:\.\d+)?)([a-zA-Z]+)$', amount_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value, unit
            
        # If no match, try splitting by space
        parts = amount_str.split()
        if len(parts) == 2:
            try:
                value = float(parts[0])
                unit = parts[1]
                return value, unit
            except ValueError:
                pass
                
        raise ValueError(f"Invalid amount format: {amount_str}")

    async def normalize_item_name(self, item_name: str) -> str:
        """Normalize item name using OpenAI to handle plurals, synonyms, and variations"""
        try:
            system_prompt = """You are a food item name normalizer. Your task is to convert any food item name to its standard singular form.
            Rules:
            1. Convert plural to singular (e.g., 'apples' -> 'apple')
            2. Use standard/common names (e.g., 'taters' -> 'potato')
            3. Remove unnecessary words (e.g., 'fresh tomatoes' -> 'tomato')
            4. Use consistent spelling (e.g., 'chilli' -> 'chili')
            5. Return ONLY the normalized name, nothing else
            
            Examples:
            - "apples" -> "apple"
            - "fresh red tomatoes" -> "tomato"
            - "minced garlic" -> "garlic"
            - "green onions" -> "onion"
            - "bell peppers" -> "bell pepper"
            - "chicken breasts" -> "chicken breast"
            - "ground beef" -> "ground beef"
            - "olive oil" -> "olive oil"
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Normalize this food item name: {item_name}"}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=50
            )

            normalized_name = response.choices[0].message.content.strip().lower()
            return normalized_name

        except Exception as e:
            logger.error(f"Error normalizing item name: {e}")
            # Fallback to simple lowercase if normalization fails
            return item_name.lower()

    async def process_classified_message(
        self,
        classification: Dict,
        user: Dict,
        message: str
    ) -> Dict:
        """Process the message based on its classification"""
        action = classification["action"]

        if action == "remove_inventory":
            try:
                # Get current inventory
                current_inventory = user.get("kitchen_inventory", {}).get("ingredients", [])
                # Create a mapping of normalized names to items
                current_items = {}
                for item in current_inventory:
                    normalized_name = await self.normalize_item_name(item["name"])
                    current_items[normalized_name] = item
                
                # Process items to remove
                items_to_remove = classification.get("items", [])
                removed_items = []
                
                for item in items_to_remove:
                    normalized_name = await self.normalize_item_name(item["name"])
                    if normalized_name in current_items:
                        current_item = current_items[normalized_name]
                        
                        if item["amount"].lower() == "all":
                            # Remove entire item
                            removed_items.append({
                                "name": current_item["name"],
                                "amount": current_item["amount"]
                            })
                            del current_items[normalized_name]
                        else:
                            # Try to parse amounts for partial removal
                            try:
                                # Parse current and remove amounts
                                current_value, current_unit = self._parse_amount_unit(current_item["amount"])
                                remove_value, remove_unit = self._parse_amount_unit(item["amount"])
                                
                                print(f"Current: {current_value} {current_unit}, Remove: {remove_value} {remove_unit}")
                                
                                # Try to convert units if they're different
                                if current_unit != remove_unit:
                                    converted_value = self._convert_unit(remove_value, remove_unit, current_unit)
                                    if converted_value is None:
                                        removed_items.append({
                                            "name": current_item["name"],
                                            "amount": current_item["amount"],
                                            "error": f"Cannot convert between {remove_unit} and {current_unit}"
                                        })
                                        continue
                                    remove_value = converted_value
                                
                                if current_value > remove_value:
                                    # Update remaining amount
                                    remaining = current_value - remove_value
                                    current_items[normalized_name]["amount"] = self._format_unit(remaining, current_unit)
                                    removed_items.append({
                                        "name": current_item["name"],
                                        "amount": self._format_unit(remove_value, current_unit)
                                    })
                                elif current_value == remove_value:
                                    # Remove entire item
                                    removed_items.append(current_item)
                                    del current_items[normalized_name]
                                else:
                                    # Can't remove more than available
                                    removed_items.append({
                                        "name": current_item["name"],
                                        "amount": current_item["amount"],
                                        "error": "Requested amount exceeds available amount"
                                    })
                            except ValueError as e:
                                # If amount parsing fails, remove entire item
                                logger.error(f"Error parsing amount: {e}")
                                removed_items.append({
                                    "name": current_item["name"],
                                    "amount": current_item["amount"],
                                    "error": "Invalid amount format"
                                })
                                del current_items[normalized_name]
                    else:
                        removed_items.append({
                            "name": item["name"],
                            "error": "Item not found in inventory"
                        })
                
                # Convert back to list
                updated_inventory = list(current_items.values())
                
                # Update user's inventory in database
                users_collection = mongodb.get_collection("users")
                await users_collection.update_one(
                    {"user_id": user["user_id"]},
                    {
                        "$set": {
                            "kitchen_inventory.ingredients": updated_inventory,
                            "kitchen_inventory.last_updated": datetime.utcnow()
                        }
                    }
                )
                
                # Format response
                if removed_items:
                    response = "🗑️ Updated your kitchen inventory!\n\n"
                    response += "Removed items:\n"
                    for item in removed_items:
                        if "error" in item:
                            response += f"• {item['name']} - {item['error']}\n"
                        else:
                            response += f"• {item['name']} ({item['amount']})\n"
                    response += f"\nRemaining items: {len(updated_inventory)}"
                else:
                    response = "No items were removed from your inventory."
                
                return {
                    "response": response,
                    "conversation_history": user.get("conversation_history", [])
                }
            except Exception as e:
                logger.error(f"Error removing inventory items: {e}")
                return {
                    "response": "❌ Sorry, there was an error updating your inventory. Please try again.",
                    "conversation_history": user.get("conversation_history", [])
                }

        elif action == "get_inventory":
            try:
                # Get current inventory
                inventory = user.get("kitchen_inventory", {}).get("ingredients", [])
                
                if not inventory:
                    response = "Your kitchen inventory is empty. Add some ingredients to get started!"
                else:
                    # Analyze message to determine what items to show
                    system_prompt = """You are an inventory analyzer. Analyze the user's message to determine what items they want to see.
                    Rules:
                    1. If message is general (e.g., "show inventory"), return "all"
                    2. If message mentions specific items, return those items
                    3. If message mentions categories (e.g., "vegetables", "spices"), return items in those categories
                    4. If message mentions recipes or cooking, return items relevant to cooking
                    5. Return a JSON object with this structure:
                    {
                        "type": "all|specific|category",
                        "items": ["item1", "item2"],  # For specific items
                        "categories": ["category1", "category2"]  # For categories
                    }
                    
                    Examples:
                    - "show my inventory" -> {"type": "all"}
                    - "do I have tomatoes?" -> {"type": "specific", "items": ["tomato"]}
                    - "show my vegetables" -> {"type": "category", "categories": ["vegetable"]}
                    - "what can I cook with?" -> {"type": "category", "categories": ["cooking"]}
                    """

                    analysis_response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Analyze this message: {message}"}
                        ],
                        temperature=0.1,
                        max_tokens=100,
                        response_format={ "type": "json_object" }
                    )

                    analysis = json.loads(analysis_response.choices[0].message.content)
                    
                    # Group items by normalized names
                    normalized_groups = {}
                    for item in inventory:
                        normalized_name = await self.normalize_item_name(item["name"])
                        if normalized_name not in normalized_groups:
                            normalized_groups[normalized_name] = []
                        normalized_groups[normalized_name].append(item)

                    # Filter items based on analysis
                    filtered_groups = {}
                    if analysis["type"] == "all":
                        filtered_groups = normalized_groups
                    elif analysis["type"] == "specific":
                        for item in analysis["items"]:
                            normalized_item = await self.normalize_item_name(item)
                            if normalized_item in normalized_groups:
                                filtered_groups[normalized_item] = normalized_groups[normalized_item]
                    elif analysis["type"] == "category":
                        # Get category for each item
                        category_prompt = """You are a food categorizer. Categorize each food item into one or more categories.
                        Categories: vegetable, fruit, meat, seafood, dairy, grain, spice, herb, oil, other
                        Return a JSON array of categories for each item.
                        Example: ["apple"] -> ["fruit"]"""

                        for normalized_name, items in normalized_groups.items():
                            category_response = await self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": category_prompt},
                                    {"role": "user", "content": f"Categorize: {normalized_name}"}
                                ],
                                temperature=0.1,
                                max_tokens=50,
                                response_format={ "type": "json_object" }
                            )
                            
                            item_categories = json.loads(category_response.choices[0].message.content)
                            # Check if any of the item's categories match the requested categories
                            if any(cat in analysis["categories"] for cat in item_categories):
                                filtered_groups[normalized_name] = items

                    # Sort items alphabetically by normalized name
                    sorted_groups = sorted(filtered_groups.items(), key=lambda x: x[0])
                    
                    # Format response
                    if not filtered_groups:
                        if analysis["type"] == "specific":
                            response = "❌ None of the requested items were found in your inventory."
                        elif analysis["type"] == "category":
                            response = f"❌ No items found in the requested categories: {', '.join(analysis['categories'])}"
                        else:
                            response = "Your kitchen inventory is empty. Add some ingredients to get started!"
                    else:
                        response = "📋 Your Kitchen Inventory:\n\n"
                        for normalized_name, items in sorted_groups:
                            if len(items) == 1:
                                response += f"• {items[0]['name']} ({items[0]['amount']})\n"
                            else:
                                # If we have multiple variations of the same item
                                response += f"• {normalized_name.title()}:\n"
                                for item in items:
                                    response += f"  - {item['name']} ({item['amount']})\n"
                        
                        response += f"\nTotal items shown: {len(filtered_groups)}"
                        if len(filtered_groups) < len(normalized_groups):
                            response += f" (of {len(normalized_groups)} total items)"
                        response += "\n\nLast updated: " + user["kitchen_inventory"]["last_updated"].strftime("%Y-%m-%d %H:%M")
                
                return {
                    "response": response,
                    "conversation_history": user.get("conversation_history", [])
                }
            except Exception as e:
                logger.error(f"Error getting inventory: {e}")
                return {
                    "response": "❌ Sorry, there was an error retrieving your inventory. Please try again.",
                    "conversation_history": user.get("conversation_history", [])
                }

        elif action == "update_inventory":
            try:
                # Get current inventory
                current_inventory = user.get("kitchen_inventory", {}).get("ingredients", [])
                # Create a mapping of normalized names to items
                current_items = {}
                for item in current_inventory:
                    normalized_name = await self.normalize_item_name(item["name"])
                    current_items[normalized_name] = item
                
                # Process new items
                new_items = classification.get("items", [])
                updated_items = []
                has_errors = False
                
                # Update or add items
                for new_item in new_items:
                    normalized_name = await self.normalize_item_name(new_item["name"])
                    try:
                        # Parse the new amount
                        new_value, new_unit = self._parse_amount_unit(new_item["amount"])
                        
                        # Validate unit
                        is_valid, unit_message = self._validate_unit(new_unit)
                        if not is_valid:
                            updated_items.append({
                                "name": new_item["name"],
                                "amount": new_item["amount"],
                                "error": unit_message
                            })
                            has_errors = True
                            continue
                        
                        if normalized_name in current_items:
                            # Parse current amount
                            current_value, current_unit = self._parse_amount_unit(current_items[normalized_name]["amount"])
                            
                            # Validate current unit
                            is_valid, unit_message = self._validate_unit(current_unit)
                            if not is_valid:
                                updated_items.append({
                                    "name": new_item["name"],
                                    "amount": current_items[normalized_name]["amount"],
                                    "error": f"Current amount has {unit_message}"
                                })
                                has_errors = True
                                continue
                            
                            # Try to convert units if they're different
                            if current_unit != new_unit:
                                converted_value = self._convert_unit(new_value, new_unit, current_unit)
                                if converted_value is None:
                                    updated_items.append({
                                        "name": new_item["name"],
                                        "amount": new_item["amount"],
                                        "error": f"Cannot convert between {new_unit} and {current_unit}"
                                    })
                                    has_errors = True
                                    continue
                                new_value = converted_value
                                new_unit = current_unit
                            
                            # Add the amounts
                            total_value = current_value + new_value
                            current_items[normalized_name]["amount"] = self._format_unit(total_value, current_unit)
                            updated_items.append({
                                "name": new_item["name"],
                                "amount": self._format_unit(total_value, current_unit),
                                "added": self._format_unit(new_value, new_unit)
                            })
                        else:
                            # Add new item
                            current_items[normalized_name] = {
                                "name": new_item["name"],
                                "amount": self._format_unit(new_value, new_unit)
                            }
                            updated_items.append({
                                "name": new_item["name"],
                                "amount": self._format_unit(new_value, new_unit),
                                "added": self._format_unit(new_value, new_unit)
                            })
                    except ValueError as e:
                        logger.error(f"Error parsing amount: {e}")
                        updated_items.append({
                            "name": new_item["name"],
                            "amount": new_item["amount"],
                            "error": "Invalid amount format"
                        })
                        has_errors = True
                
                # Only update database if there are no errors
                if not has_errors:
                    # Convert back to list
                    updated_inventory = list(current_items.values())
                    
                    # Update user's inventory in database
                    users_collection = mongodb.get_collection("users")
                    await users_collection.update_one(
                        {"user_id": user["user_id"]},
                        {
                            "$set": {
                                "kitchen_inventory.ingredients": updated_inventory,
                                "kitchen_inventory.last_updated": datetime.utcnow()
                            }
                        }
                    )
                
                # Format response
                if updated_items:
                    if has_errors:
                        response = "⚠️ Some items couldn't be added to your inventory:\n\n"
                    else:
                        response = "✅ Updated your kitchen inventory!\n\n"
                    
                    response += "Items:\n"
                    for item in updated_items:
                        if "error" in item:
                            response += f"• {item['name']} ({item['amount']}) - {item['error']}\n"
                        else:
                            response += f"• {item['name']} ({item['amount']}) [+{item['added']}]\n"
                    
                    if not has_errors:
                        response += f"\nTotal items in inventory: {len(current_items)}"
                else:
                    response = "No new items were added to your inventory."
                
                return {
                    "response": response,
                    "conversation_history": user.get("conversation_history", [])
                }
            except Exception as e:
                logger.error(f"Error updating inventory: {e}")
                return {
                    "response": "❌ Sorry, there was an error updating your inventory. Please try again.",
                    "conversation_history": user.get("conversation_history", [])
                }

        elif action == "get_recipes":
            # Update recipe generation to use item names only
            ingredients = [item["name"] for item in user["kitchen_inventory"]["ingredients"]]
            recommendations = await self.generate_recipe_recommendations(
                ingredients=ingredients,
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