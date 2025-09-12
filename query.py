import json
import re
import httpx
import logging
import datetime
import asyncio
from typing import List, Dict, Any, Optional, Set
from config import  ES_CONFIG, MISTRAL_CONFIG
from pydantic import BaseModel, Field

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)


class QueryProcessor:
    def __init__(self, es_client=None, embedding_model=None):
        """
        Initialize the EmailProcessor with required clients and models.
        
        Args:
            es_client: Elasticsearch client
            embedding_model: Model for generating embeddings
        """
        self.es_client = es_client
        self.embedding_model = embedding_model
        

    async def extract_query_generate_responses(self, tenant_id: str, thread_id:str, sender_name:str, email_content: str, language: str, type: str, subtype: Optional[str] = None, query_ai_mode: str = None, template: str = None, query_regards: str= None) -> Dict[str,Any]:
        
         # Extract all questions from the email
        questions_batch = await self.extract_multiple_questions(email_content,language,type,subtype)
        
        if not questions_batch:
            logger.warning(f"No questions found in the email for tenantId: {tenant_id}, ThreadId: {thread_id}")
            # Generate professional response even without questions
            return await self.generate_no_questions_response(sender_name, email_content, language, query_ai_mode, template)
        

        query_response = await self.batch_generate_responses(sender_name,email_content,questions_batch,tenant_id,language,query_ai_mode,template,query_regards)
        
        # query_response is already a dictionary, no need to parse JSON
        return query_response

    async def generate_no_questions_response(self, sender_name: str, email_content: str, 
                                           language: str, query_ai_mode: str = None, 
                                           template: str = None) -> Dict[str, Any]:
        """
        Generate professional response when no questions are found
        """
        try:
            system_prompt = f"""You are a professional customer service AI assistant responding in {language}.

Generate a polite, professional email response that:
1. Thanks the customer for their email
2. Acknowledges receipt of their message
3. Mentions that if they have specific questions, they're welcome to ask
4. Maintains a helpful and courteous tone
5. Ends with professional closing

Respond directly in {language} language with a complete email response."""

            user_prompt = f"""CUSTOMER EMAIL:
From: {sender_name}
Content: {email_content}

Generate a professional acknowledgment response thanking {sender_name} for their email and letting them know we're here to help with any questions they may have."""

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=400)
            data["model"] = MISTRAL_CONFIG['model']
            
            # Call API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    MISTRAL_CONFIG['service_url'],
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=MISTRAL_CONFIG['timeout']
                )
            
            if response.status_code != 200:
                logger.error(f"Mistral API error: {response.status_code}")
                return {"message": {"content": f"Thank you for your email, {sender_name}. We appreciate you reaching out to us."}}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error generating no-questions response: {str(e)}")
            return {"message": {"content": f"Thank you for your email, {sender_name}. We appreciate you reaching out to us."}}


    async def extract_multiple_questions(self, email_content: str, language: str, type: str, subtype: Optional[str] = None):
        """
        Extract questions from an email content if multiple questions are present.
        
        Args:
            email_content (str): The content of the email to analyze
            language (str): The language of the email content
            type (str): The type of the email content
            subtype (Optional[str]): The subtype of the email content
        
        Returns:
            List[str]: A list of extracted questions
        """
        system_prompt = """You are a multilingual email analysis assistant specialized in identifying questions. 
    Your task is to find and extract ONLY the explicit questions in an email.

    An explicit question is a sentence that directly asks for information, permission, or action.
    Statements expressing needs (like "I need help") are NOT questions unless phrased as questions.

    Be careful to only extract ACTUAL questions from the provided email, nothing else."""
        
        context_description = f"{type}" + (f", subtype: {subtype}" if subtype else "")
        
        user_prompt = f"""
    ORIGINAL EMAIL:
    {email_content}

    TASK:
    Analyze the above email of type {context_description} and in {language} language. Extract ONLY the explicit questions.
    Do not consider statements of need or general expressions of desire as questions.

    Return your response as a JSON array of strings containing ONLY questions found in the ORIGINAL EMAIL:
    []

    If no questions are found, return an empty array.
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
        logger.info(f"response_data: {response_data}")
        
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
            questions = json.loads(cleaned_content)
            
            # Validate that questions is a list
            if not isinstance(questions, list):
                logger.error(f"Expected list but got {type(questions)}: {questions}")
                return []
                
            logger.info(f"Extracted questions: {questions}")
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Content that failed to parse: {repr(content)}")
            
            # Fallback: Try to extract JSON using regex
            try:
                # Look for JSON array pattern in the content
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    questions = json.loads(json_str)
                    if isinstance(questions, list):
                        logger.info(f"Extracted questions via regex: {questions}")
                        return questions
            except Exception as fallback_error:
                logger.error(f"Fallback regex extraction failed: {fallback_error}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error processing questions: {e}")
            logger.error(f"Content: {repr(content)}")
            return []

    async def batch_generate_responses(self, sender_name: str, email_content:str, questions_batch: List[str], tenant_id: str, language: str, query_ai_mode: str = None,template: str = None,query_regards: str= None) -> Dict[str, Any]:
        """
        Process multiple questions in batches to minimize API calls
        
        Args:
            sender_name: Name of the email sender
            questions_batch: List of questions to process
            tenant_id: Tenant ID for knowledge base search
            language: Language to generate responses in
            
        Returns:
            Dictionary containing the response data
        """
        
        # First, gather all documents for all non-API questions in the batch
        question_documents = {}
        for question in questions_batch:
            # Search knowledge base
            search_results = await self.search_knowledge_base(
                question=question,
                tenant_id=tenant_id,
                language=language
            )
            question_documents[question] = search_results
        
        # Process questions in smaller batches that fit within Mistral's context limits
        return await self.process_questions(sender_name, email_content, question_documents, language,query_ai_mode,template,query_regards)
        


    async def process_questions(self, sender_name: str, email_content: str, question_documents: Dict[str, List[Dict]], 
                            language: str, query_ai_mode: str = None,template: str = None,query_regards: str= None) -> Dict[str, Any]:
        """
        Process email using RAG approach - provide email body and relevant documents to Mistral.
        
        Args:
            sender_name: Name of the email sender
            email_content: Original email content
            question_documents: Dictionary mapping questions to their relevant documents
            responses: List to store generated responses
            language: Language to generate responses in
            
        Returns:
            Dictionary containing the response data
        """
        # Token limit configuration
        MAX_TOKENS = 8192
        RESERVE_TOKENS = 1000
        EFFECTIVE_LIMIT = MAX_TOKENS - RESERVE_TOKENS
        TOKEN_PER_CHAR_APPROX = 0.3
        
        # Collect all documents from all questions into a single consolidated list
        all_documents = []
        
        # Track which questions we're processing
        processed_questions = set()
        

        logger.info(f"question_documents: {question_documents}")

        # Collect documents from all questions
        for question, documents in question_documents.items():
            processed_questions.add(question)
            for doc in documents:
                if doc not in all_documents:  # Avoid duplicates
                    all_documents.append(doc)
        

        # Check token limits for the entire context
        current_token_estimate = len(email_content) * TOKEN_PER_CHAR_APPROX

        # Add documents until we approach token limit
        documents_to_use = []
        for doc in all_documents:
            doc_content = doc['content']  # Just use the content directly
            doc_url = doc.get('url', '') 
           
            # Calculate tokens using safe string lengths
            doc_tokens = len(doc_content) * TOKEN_PER_CHAR_APPROX
            if doc_url:  # Only add URL tokens if URL exists
                doc_tokens += len(doc_url) * TOKEN_PER_CHAR_APPROX
  
            
            if current_token_estimate + doc_tokens > EFFECTIVE_LIMIT:
                logger.warning(f"Skipping document due to token limit")
                continue
                
            documents_to_use.append(doc)
            current_token_estimate += doc_tokens

        
        # Generate comprehensive response  and return
        return await self.generate_query_response(sender_name, email_content, question_documents, language,template,query_regards)
        

    async def generate_query_response(self, sender_name: str, email_content: str, 
                            question_documents: Dict[str, List[Dict]], 
                            language: str, template: str = None, query_regards: str = None) -> Dict[str, Any]:
        """Generate structured query response with document-based answers"""
        try:
            # Build system prompt based on template availability
            if template:
                system_prompt = f"""You are a professional customer service AI assistant responding in {language}.

    Generate an email response following this template format:
    {template}

    Use ONLY information found in the provided documents. If information is not available in documents, clearly state "This information is not available in our current documentation."

    Do not make promises or provide information not explicitly stated in the documents. Respond entirely in {language}."""
            else:
                if query_regards:
                    system_prompt = f"""You are a professional customer service AI assistant responding in {language}.

    Generate a professional email response with:
    1. Professional greeting thanking {sender_name} for their email
    2. Answer each question using ONLY information from provided documents
    3. For questions without document support, state "This information is not available in our current documentation"
    4. DO NOT add any closing, regards, or signature - stop immediately after the last answer

    Use only facts from provided documents. Do not make promises not explicitly stated in documents. Respond entirely in {language}."""
                else:
                    system_prompt = f"""You are a professional customer service AI assistant responding in {language}.

    Generate a professional email response with:
    1. Professional greeting thanking {sender_name} for their email  
    2. Answer each question using ONLY information from provided documents
    3. For questions without document support, state "This information is not available in our current documentation"
    4. Professional closing offering to help find additional information if needed

    Use only facts from provided documents. Do not make promises not explicitly stated in documents. Respond entirely in {language}."""

            # Build user prompt with document content
            user_prompt = f"""CUSTOMER EMAIL:
    From: {sender_name}
    Content: {email_content}

    QUESTIONS AND AVAILABLE DOCUMENTATION:
    """
            
            for i, (question, docs) in enumerate(question_documents.items(), 1):
                user_prompt += f"\nQuestion {i}: {question}\n"
                if docs and docs[0].get('content'):
                    user_prompt += f"Available information: {docs[0]['content'][:400]}\n"
                    if docs[0].get('url'):
                        user_prompt += f"Source: {docs[0]['url']}\n"
                else:
                    user_prompt += "No relevant documentation available for this question.\n"

            user_prompt += f"\nGenerate professional email response in {language} using only the documentation provided above."

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=800)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.1
            
            # Call API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    MISTRAL_CONFIG['service_url'],
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=MISTRAL_CONFIG['timeout']
                )
            
            if response.status_code != 200:
                logger.error(f"Mistral API error: {response.status_code}")
                return {"error": "Sorry, I couldn't generate a response at this time."}
            
            response_data = response.json()
            
            # Add query_regards if provided and no template
            if not template and query_regards:
                content = response_data['message']['content'].strip()
                content = content + f"\n\n{query_regards}"
                response_data['message']['content'] = content
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating query response: {str(e)}")
            return {"error": "Sorry, I couldn't generate a response at this time."}

    async def search_knowledge_base(self, question: str, tenant_id: str, language: str) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch for relevant documents using vector search
        
        Args:
            question (str): The question to search for
            tenant_id (str): The tenant ID (compulsory field)
            language (str, optional): The language of the question for potential language-specific handling
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents
        """
        try:
            # Step 1: Generate embedding for the question
            embeddings = await self.generate_embeddings(question)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate embeddings for the question")
                return []
            
            # Step 2: Prepare metadata filters if needed (e.g., language-specific filtering)
            metadata_filters = {}
            if language and language != "unknown":
                metadata_filters["language"] = language
            
            
            
            # Step 3: Perform vector search
            search_results = await self.vector_search(
                embedding=embeddings,  # Get the first embedding
                tenant_id=tenant_id,
                top_k=3,  # Get top 3 results as in original function
                threshold=0.55,  # Cosine similarity threshold
                metadata_filters=metadata_filters
            )
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching knowledge base with vector search: {str(e)}", exc_info=True)
            return []

    async def vector_search(
        self,
        embedding: List[float],
        tenant_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search in Elasticsearch
        
        Args:
            embedding (List[float]): The question embedding vector
            tenant_id (str): The tenant ID (compulsory field)
            top_k (int): Number of top results to return
            threshold (float): Similarity threshold for filtering results
            metadata_filters (Dict[str, Any], optional): Additional filters for metadata
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        try:
            # Build the filter conditions - tenant_id is required
            filter_conditions = [{"term": {"tenantId": tenant_id}}]
            
            # Add metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    if isinstance(value, list):
                        filter_conditions.append({"terms": {f"metadata.{key}": value}})
                    else:
                        filter_conditions.append({"term": {f"metadata.{key}": value}})
            
            # Build the query with cosine similarity script scoring
            query = {
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "filter": filter_conditions
                            }
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'contentVector')",
                            "params": {"query_vector": embedding}
                        }
                    }
                }
            }
   
            
            # Execute search
            response = await self.es_client.search(
                index=ES_CONFIG['tenant_document_index_name'],
                body=query,
                size=top_k
            )


            # Process results
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                if score >= threshold:
                    results.append({
                        "content": hit['_source']['content'],
                        "url": hit['_source'].get('url', None)
                    })
            
            logger.info(f"Vector search returned {len(results)} results above threshold {threshold}")
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            return []
              

    async def generate_embeddings(self, query: str) -> List[List[float]]:
        """Generate embeddings for a query using a thread pool"""
        try:
            start_time = datetime.datetime.now()
            
            # Move the embedding generation to a separate thread 
            # since SentenceTransformer is not async-compatible
            embeddings = await asyncio.to_thread(self._generate_embeddings_sync, query)
            
            end_time = datetime.datetime.now()
            logger.info(f"Generated {len(query)} query embeddings in {(end_time - start_time).total_seconds()} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating query embeddings: {str(e)}", exc_info=True)
            raise
    
    def _generate_embeddings_sync(self, query: str) -> List[List[float]]:
        """Synchronous method to generate embeddings (runs in a thread)"""
        embeddings = self.embedding_model.encode(query)
        return embeddings.tolist()
    



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