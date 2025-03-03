import json
import httpx
import logging
from typing import List, Dict, Any, Optional
from config import  MISTRAL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

class SuggestionExtractor:
    def __init__(self, embedding_model=None):
        """
        Initialize the SuggestionExtractor with required clients and models.
        
        Args:
            embedding_model: Model for generating embeddings
        """

        self.embedding_model = embedding_model
        

    async def extract_suggestions(self, email_content: str, type: str, subType: Optional[str], sector: Optional[str],language: str = "en") -> List[str]:
        """
        Extract suggestions from emails and search for matching solutions in Elasticsearch using Mistral AI.

        Args:
            email_content: The content of the email to analyze
            type: The predefined type of suggestion
            subType: The optional predefined subtype of suggestion
            sector: The optional business domain/sector the email relates to
            language: The language of the email content

        Returns:
            List of suggestions
        """

        # Extract suggestions from the email using Mistral AI
        suggestions_list = await self._extract_suggestions_with_mistral(email_content,type,subType,sector, language)

        return suggestions_list



    async def _extract_suggestions_with_mistral(self, email_content: str, type: str, subType: Optional[str], sector: Optional[str], language: str) -> List[str]:
        """
        Use Mistral AI to extract ONLY suggestions from email content based on type, subtype, sector and language.
        
        Args:
            email_content: The content of the email to analyze
            type: The predefined type of suggestion
            subType: The optional predefined subtype of suggestion
            sector: The optional business domain/sector the email relates to
            language: The language of the email content
            
        Returns:
            List[str]: A list of extracted suggestions
        """
        
        # Build type-specific context for the prompt
        type_context = f"This email relates to {type}"
        if subType:
            type_context += f", specifically about {subType}"
        
        # Add business domain context if available
        domain_context = f"The email is from the {sector} sector." if sector else ""
        
        system_prompt = f"""You are a multilingual email analysis assistant specialized in identifying customer suggestions.
    Your task is to find and extract ONLY the explicit improvement suggestions or recommendations in an email.
    Focus exclusively on constructive suggestions, improvement ideas, and recommendations made by the sender.
    Do NOT extract issues, problems, or complaints unless they are explicitly framed as suggestions for improvement.

    {type_context}. {domain_context}

    You must consider the cultural and linguistic context of the {language} language when analyzing this email."""

        user_prompt = f"""
    ORIGINAL EMAIL:
    {email_content}

    TASK:
    Analyze the above email in {language} language. Extract ONLY the suggestions or recommendations being expressed.
    Focus exclusively on:
    - Explicit suggestions for improvement
    - Recommendations for new features or services
    - Ideas proposed by the customer
    - Constructive feedback framed as suggestions
    - Recommendations specific to the {type} category {f'and {subType} subcategory' if subType else ''}
    - Suggestions related to the {sector if sector else 'business'} domain

    Return your response as a JSON array of strings containing ONLY the suggestions found in the ORIGINAL EMAIL:
    []

    If no suggestions are found, return an empty array.
    """

        # Call Mistral API
        data = {
            "model": MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 500,
            "temperature": 0.1  # Lower temperature for more deterministic output
        }

        logger.info(f"Sending suggestion extraction request to Mistral API with type={type}, subType={subType}, sector={sector}, language={language}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MISTRAL_CONFIG['service_url'],
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_CONFIG['api_key']}"},
                json=data,
                timeout=MISTRAL_CONFIG['timeout']
            )

        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.status_code} - {response.text}")
            return []

        response_data = response.json()
        logger.debug(f"Received response from Mistral API")

        try:
            content = response_data['message']['content']
            complaints = json.loads(content)
            return complaints
        except Exception as e:
            logger.error(f"Error processing complaints: {e}")
            return []
                
        """Synchronous method to generate embeddings (runs in a thread)"""
        embeddings = self.embedding_model.encode(query)
        return embeddings.tolist()


        """
        Format the Elasticsearch matches into a clean structure.
        
        Args:
            matches: List of Elasticsearch hit documents
            
        Returns:
            List of formatted Q&A pairs with metadata
        """
        formatted_results = []
        
        for match in matches:
            formatted_result = {
                "content": match.get("content", ""),
                "title": match.get("title", ""),
                "score": match.get("score", 0),
                "url": match.get("url", ""),
                "metadata": match.get("metadata", {})
            }
            
            formatted_results.append(formatted_result)
            
        return formatted_results