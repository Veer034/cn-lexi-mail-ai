import json
import re
import httpx
import logging
from typing import List, Dict, Any, Optional
from config import MISTRAL_CONFIG


# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

class SuggestionProcessor:
    def __init__(self):
        """
        Initialize the SuggestionProcessor
        """
        pass

    async def extract_suggestions(self, email_content: str, type: str, subType: Optional[str],  language: str = "en") -> List[str]:
        """
        Extract suggestions from emails - simplified version without document search
        
        Args:
            email_content: The content of the email to analyze
            type: The predefined type of suggestion
            subType: The optional predefined subtype of suggestion
            language: The language of the email content
            
        Returns:
            List of extracted suggestions
        """
        try:
            # Extract suggestions using Mistral
            suggestions_list = await self._extract_suggestions_with_mistral(
                email_content, type, subType, language
            )
            
            logger.info(f"Extracted {len(suggestions_list)} suggestions")
            return suggestions_list
        
        except Exception as e:
            logger.error(f"Error extracting suggestions: {e}", exc_info=True)
            return []

    
    async def _extract_suggestions_with_mistral(self, email_content: str, type: str, 
                                        subType: Optional[str], 
                                        language: str) -> List[str]:
        """
        Use Mistral AI to extract suggestions from email content with token optimization
        """
        # Truncate email content if too long to avoid token issues
        if len(email_content) > 1000:
            email_content = email_content[:1000] + "..."
        
        # Build context
        type_context = f"This email relates to {type}"
        if subType:
            type_context += f", specifically about {subType}"
        
        system_prompt = f"""Extract customer suggestions from email in {language}.

    Extract only explicit suggestions and recommendations for improvements.

    Return a simple JSON array format: ["suggestion1", "suggestion2", "suggestion3"]

    If no suggestions found, return: []

    Focus on actual improvement ideas and constructive feedback."""
        
        user_prompt = f"""Email content: {email_content}

    Extract suggestions for improvements. Return JSON array only."""
        
        # Check token usage
        estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
        logger.info(f"Suggestion extraction tokens: {estimated_tokens}")
        
        # Create API payload
        data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=300)
        data["model"] = MISTRAL_CONFIG['model']
        data["temperature"] = 0.1
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MISTRAL_CONFIG['service_url'],
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=MISTRAL_CONFIG['timeout']
            )
        
        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.status_code} - {response.text}")
            return []
        
        response_data = response.json()
        logger.info(f"Suggestion extraction response: {response_data}")
        
        try:
            content = response_data['message']['content']
            logger.info(f"Raw content: {content}")
            
            # Clean the content by removing markdown code blocks if present
            cleaned_content = content.strip()
            
            # Remove ```json and ``` if present
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            elif cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]   # Remove ```
                
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                
            # Remove any remaining whitespace/newlines
            cleaned_content = cleaned_content.strip()
            
            logger.info(f"Cleaned content: {cleaned_content}")
            
            # Parse the JSON
            parsed_data = json.loads(cleaned_content)
            
            # Handle both direct array and object with suggestions key
            if isinstance(parsed_data, list):
                # Direct array format: ["suggestion1", "suggestion2"]
                suggestions = parsed_data
            elif isinstance(parsed_data, dict) and "suggestions" in parsed_data:
                # Object format: {"suggestions": ["suggestion1", "suggestion2"]}
                suggestions = parsed_data["suggestions"]
            else:
                logger.error(f"Unexpected JSON format: {parsed_data}")
                return []
            
            # Validate that suggestions is a list
            if not isinstance(suggestions, list):
                logger.error(f"Expected list but got {type(suggestions)}: {suggestions}")
                return []
            
            # Filter out empty strings and ensure all items are strings
            valid_suggestions = [s.strip() for s in suggestions if isinstance(s, str) and s.strip()]
            
            logger.info(f"Extracted suggestions: {valid_suggestions}")
            return valid_suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Content that failed to parse: {repr(content)}")
            
            # Fallback: Try to extract JSON using regex
            try:
                # Look for JSON array pattern in the content
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    suggestions = json.loads(json_str)
                    if isinstance(suggestions, list):
                        valid_suggestions = [s.strip() for s in suggestions if isinstance(s, str) and s.strip()]
                        logger.info(f"Extracted suggestions via regex: {valid_suggestions}")
                        return valid_suggestions
            except Exception as fallback_error:
                logger.error(f"Fallback regex extraction failed: {fallback_error}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error processing suggestion extraction response: {e}")
            logger.error(f"Content: {repr(content)}")
            return []

    async def generate_suggestion_response(self, sender_name: str, email_content: str, 
                                    suggestions: List[str], language: str, 
                                    template: str = None, 
                                    suggestion_regards: str = None) -> str:
        """Generate acknowledgment response for suggestions with template and custom regards support"""
        try:
            # Base absolute rules
            base_rules = f"""You are responding to a customer email in {language}. This is the ACTUAL REPLY email.

    ABSOLUTE RULES - VIOLATION IS FORBIDDEN:
    1. ONLY acknowledge what the customer suggested - do not promise any implementations, reviews, or future actions
    2. Do not make commitments about what the company will do with the suggestions
    3. NEVER invent or add information not provided in the customer's suggestions
    4. Be polite, professional, and appreciative
    5. Use only first person ("I", "we", "our team")"""

            # Closing instructions
            template_closing = template if template else ''
            default_closing = 'Best regards,\nCustomer Service Team' if not suggestion_regards else ''
            
            if suggestion_regards:
                closing_instruction = "DO NOT add any closing/regards/signature — they will be added separately"
            else:
                closing_instruction = "End with a short, professional closing (e.g., 'Best regards, Customer Service Team')"

            # Build system prompt based on template and regards availability
            if template:
                system_prompt = f"""{base_rules}
    6. Follow this email template structure: {template_closing}
    7. {closing_instruction}

    Response format must be:
    Thank you for contacting us.

    Your suggestions:
    - [Acknowledge specific suggestion]
    - [Acknowledge specific suggestion]

    We appreciate your feedback."""
            else:
                system_prompt = f"""{base_rules}
    6. {closing_instruction}

    Response format must be:
    Dear {sender_name},

    Thank you for taking the time to provide your suggestions.

    Your suggestions:
    - [Acknowledge specific suggestion]
    - [Acknowledge specific suggestion]

    We appreciate your input.

    {default_closing}"""

            # Build user prompt with suggestions
            user_prompt = f"""CUSTOMER EMAIL FROM: {sender_name}
    EMAIL CONTENT: {email_content}

    SUGGESTIONS PROVIDED:
    """
            
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    user_prompt += f"{i}. {suggestion}\n"
            else:
                user_prompt += "No specific suggestions identified.\n"

            user_prompt += f"""

    CRITICAL INSTRUCTIONS:
    - Write complete email reply in {language}
    - Acknowledge each suggestion provided above
    - Show appreciation for their feedback
    - NEVER promise implementations, reviews, or future actions
    - Be professional and grateful within strict acknowledgment constraints"""

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=600)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.0
            
            # Call API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    MISTRAL_CONFIG['service_url'],
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=MISTRAL_CONFIG['timeout']
                )
            
            if response.status_code != 200:
                logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return f"Thank you for your suggestions, {sender_name}. We appreciate your feedback."
            
            response_data = response.json()
            content = response_data['message']['content'].strip()
            
            # Add regards if needed (exact phrase matching)
            if suggestion_regards:
                content_lower = content.lower()
                if suggestion_regards.lower() not in content_lower:
                    content = content.rstrip() + f"\n\n{suggestion_regards}"
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating suggestion response: {str(e)}", exc_info=True)
            return f"Thank you for your suggestions, {sender_name}. We appreciate your feedback."
    

    def _get_fallback_suggestion_response(self, sender_name: str, language: str) -> str:
        """Generate simple fallback response for suggestions"""
        return f"Thank you for your suggestions, {sender_name}. We will review them carefully and appreciate your feedback."
    


    def create_mistral_payload(self,system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Dict[str, Any]:
        """
        Create Mistral API payload
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Maximum response tokens
            
        Returns:
            API payload dictionary
        """
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": max_tokens
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (1 token ≈ 4 characters + 20% buffer)"""
        return int((len(text) / 4) * 1.2)
