import json
import re
import httpx
import torch
import numpy as np
import datetime
import asyncio
from typing import List, Dict, Any, Optional, Set
from config import  ES_CONFIG, MISTRAL_CONFIG
from pydantic import BaseModel, Field
import traceback
from lang_utils import LangUtil 
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
        self.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
        logger.info(f"Using device: {self.device}")

        if embedding_model and hasattr(embedding_model, 'to'):
            self.embedding_model = embedding_model.to(self.device)
        else:
            self.embedding_model = embedding_model
        

    async def extract_query_generate_responses(self, tenant_id: str, thread_id:str, sender_name:str, email_content: str, language: str, language_code: str,type: str, subtype: Optional[str] = None, query_ai_mode: str = None, template: str = None, query_regards: str= None) -> Dict[str,Any]:
        
         # Extract all questions from the email
        questions_batch = await self.extract_multiple_questions(email_content,language,type,subtype)
        
        if not questions_batch:
            logger.warning(f"No questions found in the email for tenantId: {tenant_id}, ThreadId: {thread_id}")
            # Generate professional response even without questions
            return await self.generate_no_questions_response(sender_name, email_content, language, query_ai_mode, template)
        

        query_response = await self.batch_generate_responses(sender_name,email_content,questions_batch,tenant_id,language,language_code,query_ai_mode,template,query_regards)
        
        # query_response is already a dictionary, no need to parse JSON
        return query_response
    
    async def generate_no_questions_response(self, sender_name: str, email_content: str, 
                                        language: str, query_ai_mode: str = None, 
                                        template: str = None) -> Dict[str, Any]:
        """
        Generate professional response when no questions are found
        """
        try:
            system_prompt = f"""You are a customer service representative responding in {language}.

    Write a warm, professional email response that:
    1. Thanks {sender_name} for reaching out
    2. Acknowledges you received their message
    3. Shows you understand their communication
    4. Offers assistance for any future questions
    5. Maintains a helpful and friendly tone

    Style guidelines:
    - Write as "I" or "we" (representing the company)
    - Be conversational and warm, not robotic
    - Sound like a real person responding
    - Keep it concise but genuine

    Respond directly in {language}."""

            user_prompt = f"""Customer: {sender_name}
    Customer's message: {email_content}

    Write a natural, friendly acknowledgment response that shows you received and appreciate their email. Let them know you're available to help with any questions they might have."""

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=300)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.2  # Natural language
            
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
                return {"message": {"content": f"Hi {sender_name},\n\nThank you for your email. I received your message and appreciate you reaching out to us. If you have any questions or need assistance with anything, please don't hesitate to let me know.\n\nI'm here to help!"}}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error generating no-questions response: {str(e)}")
            return {"message": {"content": f"Hi {sender_name},\n\nThank you for your email. I received your message and appreciate you reaching out to us. If you have any questions or need assistance with anything, please don't hesitate to let me know.\n\nI'm here to help!"}}

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
        system_prompt = f"""Extract questions from emails in {language}. Return only a JSON array format.

    Extract:
    - Direct questions (What plans are available?)
    - Information requests (Can you provide contact details?)
    - Requests that need specific answers

    Do NOT extract:
    - General statements (I need help, I am looking for)
    - Greetings or thanks

    Return format: ["question 1", "question 2"] or [] if no questions found."""

        context_description = f"{type}" + (f", subtype: {subtype}" if subtype else "")
        
        user_prompt = f"""Email Type: {context_description}
    Language: {language}

    Email Content:
    {email_content}

    Extract all questions that request specific information or action. Return only JSON array format."""

        try:
            # Call Mistral API with simpler parameters
            data = {
                "model": MISTRAL_CONFIG['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    MISTRAL_CONFIG['service_url'],
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=60
                )
            
            if response.status_code != 200:
                logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return []
            
            response_data = response.json()
            content = response_data['message']['content']
            logger.info(f"Raw API response: {repr(content)}")
            
            # Aggressive cleaning for JSON extraction
            cleaned_content = content.strip()
            
            # Remove common prefixes/suffixes that break JSON
            prefixes_to_remove = ['```json', '```', 'json', 'JSON:', 'Questions:', 'Array:']
            suffixes_to_remove = ['```', '```json']
            
            for prefix in prefixes_to_remove:
                if cleaned_content.startswith(prefix):
                    cleaned_content = cleaned_content[len(prefix):].strip()
            
            for suffix in suffixes_to_remove:
                if cleaned_content.endswith(suffix):
                    cleaned_content = cleaned_content[:-len(suffix)].strip()
            
            logger.info(f"Cleaned content: {repr(cleaned_content)}")
            
            # Multiple parsing attempts
            parsing_attempts = [
                # Attempt 1: Direct parsing
                lambda: json.loads(cleaned_content),
                
                # Attempt 2: Extract JSON array with regex
                lambda: json.loads(re.search(r'\[.*?\]', cleaned_content, re.DOTALL).group(0)),
                
                # Attempt 3: Fix common JSON issues and parse
                lambda: json.loads(cleaned_content.replace("'", '"').replace('`', '')),
                
                # Attempt 4: Extract and fix quotes
                lambda: json.loads(re.sub(r'["""]', '"', cleaned_content)),
            ]
            
            questions = []
            for i, attempt in enumerate(parsing_attempts, 1):
                try:
                    result = attempt()
                    if isinstance(result, list):
                        questions = result
                        logger.info(f"Parsing attempt {i} successful: {questions}")
                        break
                    else:
                        logger.warning(f"Attempt {i} returned non-list: {type(result)}")
                except Exception as e:
                    logger.debug(f"Parsing attempt {i} failed: {e}")
                    continue
            
            # If all parsing attempts failed, try manual extraction
            if not questions:
                logger.warning("All JSON parsing attempts failed, trying manual extraction")
                
                # Look for quoted strings that could be questions
                question_pattern = r'"([^"]*\?[^"]*)"'
                matches = re.findall(question_pattern, cleaned_content)
                if matches:
                    questions = matches
                    logger.info(f"Manual extraction found: {questions}")
                else:
                    # Look for any quoted strings as potential questions
                    general_pattern = r'"([^"]+)"'
                    matches = re.findall(general_pattern, cleaned_content)
                    # Filter for question-like content
                    questions = [q for q in matches if any(word in q.lower() for word in ['what', 'how', 'when', 'where', 'why', 'can', 'could', 'will', 'would', 'do', 'does', 'is', 'are']) or q.strip().endswith('?')]
                    logger.info(f"General extraction found: {questions}")
            
            # Validation and cleanup
            if not isinstance(questions, list):
                logger.error(f"Final result is not a list: {type(questions)}")
                return []
            
            # Clean up questions and validate
            cleaned_questions = []
            for q in questions:
                if isinstance(q, str) and len(q.strip()) > 3:
                    cleaned_questions.append(q.strip())
            
            # Limit number of questions
            if len(cleaned_questions) > 5:
                logger.warning(f"Too many questions ({len(cleaned_questions)}), limiting to 5")
                cleaned_questions = cleaned_questions[:5]
            
            logger.info(f"Final extracted questions: {cleaned_questions}")
            return cleaned_questions
            
        except Exception as e:
            logger.error(f"Critical error in question extraction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def batch_generate_responses(self, sender_name: str, email_content:str, questions_batch: List[str], tenant_id: str, language: str, language_code: str, query_ai_mode: str = None,template: str = None,query_regards: str= None) -> Dict[str, Any]:
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
                language_code=language_code
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
            logger.info(f"Generating query response for {sender_name} in {language}")
            
            # Base absolute rules
            base_rules = f"""You are responding to a customer email in {language}. This is the ACTUAL REPLY email.

    ABSOLUTE RULES - VIOLATION IS FORBIDDEN:
    1. NEVER generate phone numbers, email addresses, URLs, website links, business hours, or any other contact details under any circumstance.
    2. NEVER invent or infer information not explicitly present in the DOCUMENT CONTENT provided.
    3. If the DOCUMENT CONTENT does not explicitly contain an answer, always respond with EXACTLY: "I don't have that information available right now".
    4. Format responses strictly as point-by-point Q&A structure.
    5. Use only first person ("I", "we", "our team")."""

            # Closing instructions
            template_closing = template if template else ''
            default_closing = 'Best regards,\nCustomer Service Team' if not query_regards else ''
            # Closing instructions
            if query_regards:
                closing_instruction = "DO NOT add any closing/regards/signature — they will be added separately"
            else:
                closing_instruction = "End with a short, professional closing (e.g., 'Best regards, Customer Service Team')"

            # Build system prompt
            if template:
                system_prompt = f"""{base_rules}
    6. Follow this email template structure: {template_closing}
    7. {closing_instruction}

    Response format must be:
    Thank you for contacting us.

    1. [Question]: [Answer from document OR "I don't have that information available right now"]
    2. [Question]: [Answer from document OR "I don't have that information available right now"]
    """
            else:
                system_prompt = f"""{base_rules}
    6. {closing_instruction}

    Response format must be:
    Dear {sender_name},

    Thank you for reaching out to us.

    1. [Question]: [Answer from document OR "I don't have that information available right now"]
    2. [Question]: [Answer from document OR "I don't have that information available right now"]

    {default_closing}"""

            # Build user prompt
            user_prompt = f"""CUSTOMER EMAIL FROM: {sender_name}
    EMAIL CONTENT: {email_content}

    QUESTIONS WITH AVAILABLE DOCUMENTS:
    (Use ONLY the content provided below. DO NOT invent or infer information.)
    """
            # Append questions
            for idx, (question, docs) in enumerate(question_documents.items(), start=1):
                user_prompt += f"\nQUESTION {idx}: {question}\nDOCUMENT CONTENT: "
                if docs and len(docs) > 0 and docs[0].get('content'):
                    # Increased slice to 1000 chars
                    doc_content = docs[0]['content'][:1000].strip()
                    user_prompt += f'"{doc_content}"\n'
                else:
                    user_prompt += "NO INFORMATION AVAILABLE\n"
                user_prompt += "---\n"

            user_prompt += f"""
    CRITICAL INSTRUCTIONS:
    - Write complete email reply in {language}
    - Answer each question using ONLY the document content provided above
    - If document shows "NO INFORMATION AVAILABLE", respond with "I don't have that information available right now"
    - NEVER create phone numbers, emails, URLs, or any contact details
    - Format as numbered Q&A list
    - Be professional and helpful within strict document constraints"""

            # Build API payload
            try:
                data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=700)
                data["model"] = MISTRAL_CONFIG['model']
                data["temperature"] = 0.0
                data["top_p"] = 0.1
                data["repetition_penalty"] = 1.1
                logger.info("API payload created successfully")
            except Exception as payload_error:
                logger.error(f"Payload creation failed: {str(payload_error)}")
                return {"error": "Failed to create API request"}

            # Call API
            try:
                logger.info(f"Calling Mistral API : {data}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        MISTRAL_CONFIG['service_url'],
                        headers={"Content-Type": "application/json"},
                        json=data,
                        timeout=MISTRAL_CONFIG['timeout']  # increased to handle larger content
                    )
                logger.info(f"API response status: {response.status_code}")
            except httpx.TimeoutException:
                logger.error("API request timeout")
                return {"error": "Request timeout - please try again"}
            except Exception as api_error:
                logger.error(f"API call failed: {str(api_error)}")
                return {"error": "API communication failed"}

            # Parse response
            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text}")
                return {"error": f"API returned error: {response.status_code}"}

            try:
                response_data = response.json()
                content = response_data.get('message', {}).get('content')
                if not content:
                    logger.error("Empty response from API")
                    return {"error": "Empty response received"}
            except Exception as parse_error:
                logger.error(f"Response parsing failed: {str(parse_error)}")
                return {"error": "Failed to parse API response"}

            # Add regards if needed (exact phrase matching)
            if query_regards:
                content_lower = content.lower()
                if query_regards.lower() not in content_lower:
                    content = content.rstrip() + f"\n\n{query_regards}"
                    response_data['message']['content'] = content
                    logger.info("Custom regards added successfully")

            logger.info("Query response generated successfully")
            return response_data

        except Exception as e:
            import traceback
            logger.error(f"Critical error in query response: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"System error: {str(e)}"}


    async def search_knowledge_base(self, question: str, tenant_id: str, language_code: str) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch for relevant documents using vector search
        
        Args:
            question (str): The question to search for
            tenant_id (str): The tenant ID (compulsory field)
            language_code (str, optional): The language of the question for potential language-specific handling
            
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
            if language_code and language_code != "unknown":
                metadata_filters["language"] = language_code
            
            
            
            # Step 3: Perform vector search
            search_results = await self.search_elasticsearch_with_enhanced_chunking(
                embedding=embeddings,  # Get the first embedding
                tenant_id=tenant_id,
                top_k=3,  # Get top 3 results as in original function
                threshold=0.55,  # Cosine similarity threshold
                metadata_filters=metadata_filters,include_context= True
            )
        
            return search_results
        except Exception as e:
            logger.error(f"Error searching knowledge base with vector search: {str(e)}", exc_info=True)
            return []


    async def search_elasticsearch_with_enhanced_chunking(
        self, 
        embedding: List[float], 
        tenant_id: str, 
        top_k: int = 5, 
        threshold: float = 0.55,
        metadata_filters: Optional[Dict[str, Any]] = None,
        include_context: bool = True,
        original_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search that works with improved chunking but preserves contact information
        """
        try:
            # Build basic filter conditions
            filter_conditions = [{"term": {"tenantId": tenant_id}}]
            language_code = metadata_filters["language"]
            # Add metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    if isinstance(value, list):
                        filter_conditions.append({"terms": {f"metadata.{key}": value}})
                    else:
                        filter_conditions.append({"term": {f"metadata.{key}": value}})
            
            question = LangUtil._is_question(original_query,language_code)

            # Build query - hybrid if we have original text, semantic-only otherwise
            if original_query:
                # Hybrid search: semantic + keyword (simple version)
                query = {
                    "query": {
                        "bool": {
                            "filter": filter_conditions,
                            "should": [
                                # Semantic search with normalized scoring
                                {
                                    "script_score": {
                                        "query": {"match_all": {}},
                                        "script": {
                                            "source": "Math.max(0, (cosineSimilarity(params.query_vector, 'contentVector') + 1.0) / 2.0)",
                                            "params": {"query_vector": embedding}
                                        },
                                        "boost": 2.0
                                    }
                                },
                                # Simple keyword search
                                {
                                    "multi_match": {
                                        "query": original_query,
                                        "fields": ["content^2", "keywords^1.5", "sectionTitle"],
                                        "type": "best_fields",
                                        "boost": 1.0
                                    }
                                },
                                # Boost FAQ content for question-like queries
                                {
                                    "bool": {
                                        "must": [
                                            {"wildcard": {"chunkType": "*faq*"}},
                                            {"match": {"content": original_query}}
                                        ],
                                        "boost": 1.5 if question else 1.0
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "_source": ["content", "documentId", "chunkPosition", "totalChunks", "chunkType", "url"]
                }
            else:
                # Semantic-only search
                query = {
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "filter": filter_conditions
                                }
                            },
                            "script": {
                                "source": "Math.max(0, (cosineSimilarity(params.query_vector, 'contentVector') + 1.0) / 2.0)",
                                "params": {"query_vector": embedding}
                            }
                        }
                    },
                    "_source": ["content", "documentId", "chunkPosition", "totalChunks", "chunkType", "url"]
                }

            logger.info(f"Enhanced search with original_query: {bool(original_query)}")
            
            # Execute search
            response = await self.es_client.search(
                index=ES_CONFIG['tenant_document_index_name'],
                body=query,
                size=top_k
            )
            
            # Process results - preserve contact information
            results = []
            
            if include_context:
                # Get chunks with context but preserve contact info
                for hit in response['hits']['hits']:
                    score = hit['_score']
                    if score >= threshold:
                        enhanced_content = await self._get_chunk_with_adjacent_context_preserve_contacts(
                            hit['_source'], 
                            tenant_id
                        )
                        content = enhanced_content.get('content') or ""
                        # Don't strip newlines aggressively to preserve contact formatting
                        content = content.strip()
                    
                        results.append({
                            "content": content,
                            "url": hit['_source'].get('url', None)
                        })
            else:
                # Simple content only but preserve formatting
                for hit in response['hits']['hits']:
                    score = hit['_score']
                    if score >= threshold:
                        content = hit['_source'].get('content') or ""
                        # Preserve formatting for contact information
                        content = content.strip()

                        results.append({
                            "content": content,
                            "url": hit['_source'].get('url', None)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {str(e)}", exc_info=True)
            raise


    async def _get_chunk_with_adjacent_context_preserve_contacts(
        self, 
        chunk_source: Dict[str, Any], 
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Get chunk content with adjacent context - preserves contact information formatting
        """
        try:
            document_id = chunk_source.get('documentId')
            current_position = chunk_source.get('chunkPosition', 0)
            total_chunks = chunk_source.get('totalChunks', 1)
            
            base_content = {
                'content': chunk_source['content'],
            }
            
            # Only get adjacent context if we have multiple chunks and it's not already consolidated
            chunk_type = chunk_source.get('chunkType', '')
            if total_chunks > 1 and not chunk_type.endswith('_consolidated'):
                # Get previous and next chunk for context
                adjacent_positions = []
                if current_position > 0:
                    adjacent_positions.append(current_position - 1)
                if current_position < total_chunks - 1:
                    adjacent_positions.append(current_position + 1)
                
                if adjacent_positions:
                    adjacent_query = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"tenantId": tenant_id}},
                                    {"term": {"documentId": document_id}},
                                    {"terms": {"chunkPosition": adjacent_positions}}
                                ]
                            }
                        },
                        "_source": ["content", "chunkPosition"],
                        "sort": [{"chunkPosition": {"order": "asc"}}],
                        "size": 2
                    }
                    
                    try:
                        adjacent_response = await self.es_client.search(
                            index=ES_CONFIG['tenant_document_index_name'],
                            body=adjacent_query
                        )
                        
                        # Smarter context addition that preserves contact information
                        context_parts = [base_content['content']]
                        
                        for hit in adjacent_response['hits']['hits']:
                            pos = hit['_source']['chunkPosition']
                            content = hit['_source']['content']
                            
                            # Check if this chunk contains contact information patterns
                            contact_patterns = [
                                r'@[\w.-]+\.[a-zA-Z]{2,}',  # Email patterns
                                r'\+?\d{1,3}[-.\s]?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,6}',  # Phone patterns
                                r'contact|email|phone|call|reach|support',  # Contact keywords
                                r'sales@|support@|info@|help@',  # Common email prefixes
                            ]
                            
                            has_contact_info = any(re.search(pattern, content, re.IGNORECASE) for pattern in contact_patterns)
                            
                            if pos < current_position:
                                # Previous context - take more if it has contact info
                                context_length = 500 if has_contact_info else 200
                                context_parts.insert(0, content[-context_length:])
                            elif pos > current_position:
                                # Next context - take more if it has contact info
                                context_length = 500 if has_contact_info else 200
                                context_parts.append(content[:context_length])
                        
                        # Combine with separators that preserve readability
                        if len(context_parts) > 1:
                            base_content['content'] = ' ... '.join(context_parts)
                            
                    except Exception as e:
                        logger.warning(f"Could not fetch adjacent context: {str(e)}")
            
            return base_content
            
        except Exception as e:
            logger.error(f"Error getting chunk with context: {str(e)}")
            return {
                'content': chunk_source.get('content', '')
            }

    async def generate_embeddings(self, query: str) -> List[float]:
        """Generate embeddings for a list of queries using a thread pool"""
        try:
            start_time = datetime.datetime.now()
            
            # Move the embedding generation to a separate thread 
            # since SentenceTransformer is not async-compatible
            embeddings = await asyncio.to_thread(self._generate_embeddings_sync_optimized, query)
            
            end_time = datetime.datetime.now()
            logger.info(f"Generated {len(query)} embeddings in {(end_time - start_time).total_seconds()} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def _generate_embeddings_sync_optimized(self, query: str) -> List[float]:
        """Optimized synchronous embedding generation with proper device handling"""
        try:
            # Ensure model is on correct device
            if hasattr(self.embedding_model, 'to') and self.embedding_model.device != self.device:
                self.embedding_model = self.embedding_model.to(self.device)
            
            # Performance optimizations
            with torch.no_grad():  # Disable gradient computation
                # Set torch device for consistent tensor operations
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear cache before processing
                
                embeddings = self.embedding_model.encode(
                    query,
                    show_progress_bar=False,  # Disable progress bar for single queries
                    convert_to_numpy=True,    # Direct numpy conversion
                    normalize_embeddings=True,  # Normalize for cosine similarity
                    batch_size=1,            # Single query batch
                    device=self.device       # Explicit device specification
                )
            
            # Convert to list efficiently
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            else:
                return embeddings
                
        except Exception as e:
            logger.error(f"❌ Error in sync embedding generation: {str(e)}")
            raise
        finally:
            # Clean up GPU memory if using CUDA
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()



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