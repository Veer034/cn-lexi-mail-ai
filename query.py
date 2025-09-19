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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

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
        system_prompt = """You are a multilingual email analysis assistant specialized in identifying questions with precision and consistency across all languages.

    Your task is to find and extract questions from emails - including direct questions, information requests, and polite inquiries that seek specific information or action.

    WHAT TO EXTRACT:
    1. Direct questions using interrogative words in any language
    2. Sentences with question punctuation (varies by language)
    3. Polite requests that ask for specific information or action
    4. Information-seeking statements that clearly request answers

    WHAT NOT TO EXTRACT:
    1. General statements of need or desire that don't ask questions
    2. Statements of preference that don't seek information
    3. Greetings, thanks, or social pleasantries
    4. Confirmations or acknowledgments

    IMPORTANT: Analyze BOTH the subject line AND the email body - questions can appear in either location.

    Work with any language and writing system. Be consistent across all languages."""
        
        context_description = f"{type}" + (f", subtype: {subtype}" if subtype else "")
        
        user_prompt = f"""
    EMAIL TO ANALYZE:
    Subject: {email_content.split('Body:')[0].replace('Subject:', '').strip() if 'Subject:' in email_content else 'No subject'}
    Body: {email_content.split('Body:')[1].strip() if 'Body:' in email_content else email_content}

    TASK:
    Analyze BOTH the subject line AND body of this {context_description} email in {language} language.

    Extract questions that ask for information, action, or clarification from BOTH subject and body:
    - Direct questions (What plans are available?)
    - Information requests (Please provide contact details)
    - Polite inquiries seeking specific answers

    DO NOT extract general statements like "I need..." or "I am looking for..." unless they're phrased as actual questions.

    Return a clean JSON array:
    ["question from subject or body", "another question"]

    If no questions found, return: []"""
        
        # Call Mistral API
        data = {
            "model": MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 400,  # Reduced to encourage conciseness
            "temperature": 0.05  # Even lower for maximum consistency
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
            
            # Additional validation - limit to reasonable number of questions
            if len(questions) > 10:  # Sanity check
                logger.warning(f"Unusually high number of questions extracted: {len(questions)}, truncating to first 5")
                questions = questions[:5]
                
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
                        # Apply same validation
                        if len(questions) > 10:
                            questions = questions[:5]
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
            # Build system prompt based on template and regards availability
            if template:
                if query_regards:
                    system_prompt = f"""You are a customer service representative responding in {language}.

    Follow this template:
    {template}

    Rules:
    - Write as a helpful customer service agent speaking directly to the customer
    - Use information from provided documents when available
    - When information is not available, politely explain that you don't have that specific information at hand
    - Speak in first person ("I", "we", "our company") rather than third person
    - Be conversational and professional, not robotic
    - DO NOT add any closing text, signatures, regards, or contact information
    - DO NOT add any closing phrases or greetings at the end
    - Stop immediately after answering the last question

    Respond in {language}."""
                else:
                    system_prompt = f"""You are a customer service representative responding in {language}.

    Follow this template:
    {template}

    Rules:
    - Write as a helpful customer service agent speaking directly to the customer
    - Use information from provided documents when available
    - When information is not available, politely explain that you don't have that specific information at hand
    - Speak in first person ("I", "we", "our company") rather than third person
    - Be conversational and professional, not robotic
    - End with appropriate professional closing

    Respond in {language}."""
            else:
                if query_regards:
                    system_prompt = f"""You are a customer service representative responding in {language}.

    Write a response that:
    1. Thanks {sender_name} for reaching out
    2. Answers their questions using available information
    3. For missing information, politely explains you don't have those details readily available
    4. Speaks naturally in first person as a real person would

    Style guidelines:
    - Write as "I" or "we" (our team/company), not "the documentation says"
    - Be warm, helpful, and conversational
    - When you don't know something, say "I don't have that information available right now" instead of referring to documentation
    - DO NOT add any closing text, signatures, regards, or contact information
    - DO NOT add any closing phrases or greetings at the end
    - Stop immediately after the last answer

    Respond in {language}."""
                else:
                    system_prompt = f"""You are a customer service representative responding in {language}.

    Write a response that:
    1. Thanks {sender_name} for reaching out
    2. Answers their questions using available information  
    3. For missing information, politely explains you don't have those details readily available
    4. Offers to help further or connect them with someone who can assist
    5. Speaks naturally in first person as a real person would

    Style guidelines:
    - Write as "I" or "we" (our team/company), not "the documentation says"
    - Be warm, helpful, and conversational
    - When you don't know something, say "I don't have that information available right now" instead of referring to documentation
    - Sound like a real person helping another person
    - End with appropriate professional closing

    Respond in {language}."""

            # Build user prompt with better context
            user_prompt = f"""Customer: {sender_name}
    Customer's message: {email_content}

    Here are their questions with available information:
    """
            
            for i, (question, docs) in enumerate(question_documents.items(), 1):
                user_prompt += f"\nQuestion {i}: {question}\n"
                if docs and docs[0].get('content'):
                    user_prompt += f"Available information: {docs[0]['content'][:400]}\n"
                    if docs[0].get('url'):
                        user_prompt += f"Reference: {docs[0]['url']}\n"
                else:
                    user_prompt += "No specific information available for this question.\n"

            user_prompt += f"\nRespond naturally in {language} as a helpful customer service representative. Address each question conversationally, not in a formal Q&A format. When you don't have information, explain it naturally without mentioning 'documentation'."
            
            # Add instruction when custom regards are provided
            if query_regards:
                user_prompt += f"\n\nIMPORTANT: Do not add any closing text, regards, signatures, or closing phrases in {language}. Stop after answering all questions."

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=800)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.0  # Set to 0 for consistent responses
            
            logger.info(f"question request : {data}")

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
            logger.info(f"query response : {response_data}")
            
            # Add custom regards if provided
            if query_regards:
                logger.info(f"query_regards : {query_regards}")
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
        Enhanced search that works with improved chunking but keeps interface simple
        """
        try:
            # Build basic filter conditions
            filter_conditions = [{"term": {"tenantId": tenant_id}}]
            
            # Add metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    if isinstance(value, list):
                        filter_conditions.append({"terms": {f"metadata.{key}": value}})
                    else:
                        filter_conditions.append({"term": {f"metadata.{key}": value}})
            
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
                                        "boost": 1.5 if self._is_question(original_query) else 1.0
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "_source": ["content", "documentId", "chunkPosition", "totalChunks", "chunkType"]
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
                    "_source": ["content", "documentId", "chunkPosition", "totalChunks", "chunkType"]
                }

            logger.info(f"Enhanced search with original_query: {bool(original_query)}")
            
            # Execute search
            response = await self.es_client.search(
                index=ES_CONFIG['tenant_document_index_name'],
                body=query,
                size=top_k
            )
            
            # Process results - keep it simple
            results = []
            
            if include_context:
                # Get chunks with basic adjacent context (existing logic)
                for hit in response['hits']['hits']:
                    score = hit['_score']
                    if score >= threshold:
                        enhanced_content = await self._get_chunk_with_adjacent_context(
                            hit['_source'], 
                            tenant_id
                        )
                        content = enhanced_content.get('content') or ""
                        content = content.replace("\n", " ").strip()
                       
                        results.append({
                            "content": content,
                            "url": hit['_source'].get('url', None)
                        })
            else:
                # Simple content only
                for hit in response['hits']['hits']:
                    score = hit['_score']
                    if score >= threshold:
                        content = hit['_source'].get('content') or ""
                        content = content.replace("\n", " ").strip()

                        results.append({
                            "content": content,
                            "url": hit['_source'].get('url', None)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {str(e)}", exc_info=True)
            raise

    async def _get_chunk_with_adjacent_context(
            self, 
            chunk_source: Dict[str, Any], 
            tenant_id: str
        ) -> Dict[str, Any]:
        """
        Get chunk content with adjacent context - simplified version
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
                        
                        # Simple context addition
                        context_parts = [base_content['content']]
                        
                        for hit in adjacent_response['hits']['hits']:
                            pos = hit['_source']['chunkPosition']
                            content = hit['_source']['content']
                            
                            if pos < current_position:
                                context_parts.insert(0, content[-200:])  # Previous context
                            elif pos > current_position:
                                context_parts.append(content[:200])     # Next context
                        
                        # Combine with simple separators
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
        """Optimized synchronous embedding generation"""
        try:
            # Performance optimizations
            with torch.no_grad():  # Disable gradient computation
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