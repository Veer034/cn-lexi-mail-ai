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

    RULES:
    1. Extract ONLY explicit suggestions and recommendations
    2. Focus on improvement ideas and constructive feedback
    3. Consider {type_context}.
    4. Return JSON array: ["suggestion1", "suggestion2", ...]
    5. If no suggestions, return empty array"""
        
        user_prompt = f"""EMAIL: {email_content}

    Extract suggestions and recommendations for improvements.

    Focus on:
    - Explicit suggestions for improvement
    - New feature recommendations  
    - Constructive feedback as suggestions
    - Ideas specific to {type} {f'and {subType}' if subType else ''}

    Return JSON array of suggestion strings."""
        
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
            suggestions = json.loads(cleaned_content)
            
            # Validate that suggestions is a list
            if not isinstance(suggestions, list):
                logger.error(f"Expected list but got {type(suggestions)}: {suggestions}")
                return []
                
            logger.info(f"Extracted suggestions: {suggestions}")
            return suggestions
            
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
                        logger.info(f"Extracted suggestions via regex: {suggestions}")
                        return suggestions
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
            # Build system prompt based on template availability
            if template:
                system_prompt = f"""You are a professional customer service AI assistant responding to customer suggestions in {language}.

    Generate a professional suggestion acknowledgment response following this template format:
    {template}

    Use the provided suggestions to fill the content appropriately.
    Show appreciation for the customer's feedback and suggestions.
    Maintain the template structure while expressing gratitude.
    Respond entirely in {language}."""
            else:
                if suggestion_regards:
                    system_prompt = f"""You are a professional customer service AI assistant responding to customer suggestions in {language}.

    Generate a professional suggestion acknowledgment response with:
    1. Thank {sender_name} for taking time to provide suggestions
    2. Acknowledge the specific suggestions they made
    3. Assure them the company will review and consider their suggestions
    4. Express appreciation for their continued engagement
    5. DO NOT add any closing, regards, or signature - stop immediately after expressing appreciation

    Be polite, professional, and appreciative.
    Do NOT promise specific implementations or timelines.
    Respond entirely in {language}."""
                else:
                    system_prompt = f"""You are a professional customer service AI assistant responding to customer suggestions in {language}.

    Generate a professional suggestion acknowledgment response with:
    1. Thank {sender_name} for taking time to provide suggestions
    2. Acknowledge the specific suggestions they made
    3. Assure them the company will review and consider their suggestions
    4. Express appreciation for their continued engagement
    5. Professional closing offering further assistance

    Be polite, professional, and appreciative.
    Do NOT promise specific implementations or timelines.
    Respond entirely in {language}."""

            # Build user prompt with suggestions
            user_prompt = f"""CUSTOMER EMAIL:
    From: {sender_name}
    Content: {email_content}

    SUGGESTIONS PROVIDED:
    """
            
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    user_prompt += f"{i}. {suggestion}\n"
            else:
                user_prompt += "No specific suggestions identified.\n"

            user_prompt += f"\nGenerate professional acknowledgment response in {language} thanking {sender_name} for their suggestions."
            
            # Add specific instruction for suggestion_regards case
            if suggestion_regards:
                user_prompt += " DO NOT add any closing or regards - end after expressing appreciation."

            # Check token usage
            estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
            if estimated_tokens > 7000:
                user_prompt = user_prompt[:4000] + "..."
            
            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=600)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.3
            
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
                return f"Thank you for your suggestions, {sender_name}. We will review them carefully and appreciate your feedback."
            
            response_data = response.json()
            content = response_data['message']['content']
            
            # Clean any markdown formatting
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```.*?\n', '', content)
            if content.endswith('```'):
                content = re.sub(r'\n```$', '', content)
            content = content.strip()
            
            # Add suggestion_regards if provided and no template
            if not template and suggestion_regards:
                content = content.strip() + f"\n\n{suggestion_regards}"
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating suggestion response: {str(e)}", exc_info=True)
            return f"Thank you for your suggestions, {sender_name}. We will review them carefully and appreciate your feedback."   


    def _create_suggestion_system_prompt(self, language: str, suggestion_ai_mode: str = None, template: str = None) -> str:
        """Create system prompt for suggestion acknowledgment responses"""
        
        base_prompt = f"""You are a professional customer service AI assistant responding to customer suggestions in {language}.

    RESPONSE GUIDELINES:
    1. Be polite, professional, and appreciative
    2. Acknowledge the specific suggestions made by the customer
    3. Thank them for taking the time to provide feedback
    4. Assure them that the company will review and consider their suggestions
    5. Keep the tone warm but professional
    6. Do NOT promise specific implementations or timelines
    7. Do NOT provide solutions - only acknowledge receipt and review
    8. Respond in {language} language

    RESPONSE STRUCTURE:
    - Thank the customer for their suggestions
    - Acknowledge the specific suggestions they made
    - Assure them the company will look into the suggestions
    - Express appreciation for their continued engagement
    - Professional closing

    Keep the response concise but thoughtful."""

        if suggestion_ai_mode:
            base_prompt += f"\n\nADDITIONAL MODE: {suggestion_ai_mode}"
        
        if template:
            base_prompt += f"\n\nTEMPLATE GUIDANCE: {template}"
        
        return base_prompt

    def _create_suggestion_user_prompt(self, sender_name: str, email_content: str, 
                                    suggestions_context: str, template: str = None, 
                                    suggestion_ai_mode: str = None) -> str:
        """Create user prompt for suggestion acknowledgment"""
        
        prompt = f"""CUSTOMER EMAIL:
    From: {sender_name}
    Content: {email_content}

    {suggestions_context}

    Generate a professional acknowledgment response that:
    1. Thanks {sender_name} for their valuable suggestions
    2. Acknowledges that the company will review and consider the suggestions
    3. Expresses appreciation for their feedback
    4. Maintains a professional and courteous tone

    The response should be a direct email reply, not a JSON format."""

        return prompt

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
