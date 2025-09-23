import json
import httpx
import re
import torch
import numpy as np
import datetime
import asyncio
from typing import List, Dict, Any, Optional
from response import TicketData
from config import  ES_CONFIG, MISTRAL_CONFIG
from lang_utils import LangUtil


# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

class ComplaintProcessor:
    def __init__(self, es_client=None, embedding_model=None):
        """
        Initialize the ComplaintProcessor with required clients and models.
        
        Args:
            es_client: Elasticsearch client
            embedding_model: Model for generating embeddings
        """
        self.es_client = es_client
        # Optimize device detection
        self.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
        logger.info(f"ComplaintProcessor using device: {self.device}")

        # Set model to device if provided during init
        if embedding_model and hasattr(embedding_model, 'to'):
            self.embedding_model = embedding_model.to(self.device)
        else:
            self.embedding_model = embedding_model
        

    async def extract_complaints(self, tenant_id: str, email_content: str, language: str, language_code:str) -> Dict:
        """
        Extract complaints and related information from an email
        
        Args:
            tenant_id: Tenant identifier for vector search context
            email_content: Content of the email to analyze
            language: Detected language of the email
            
        Returns:
            Dict: Dictionary with complaints and advices
        """
        try:
            # Extract complaints and related queries using Mistral
            extraction_result = await self._extract_complaints_with_mistral(email_content, language)
            
            complaints_list = extraction_result.get("complaints", [])
            related_queries = extraction_result.get("related_queries", [])

            if not complaints_list:
                logger.info("No complaints found in the email")
                return {"complaints": [], "advices": []}
            
            # Perform vector search for each complaint
            advice_list = []
            
            # Search using related queries for additional context
            for query in related_queries:
                
                # Use the vector_search method
                query_results = await self.search_knowledge_base(
                    question=query,
                    tenant_id=tenant_id,
                    language_code=language_code
                    )
                
                logger.info(f"Query results: {query_results}")
            
                for item in query_results:
                    advice_content = item.get("content", "")
                    advice_url = item.get("url", "#") if item.get("url") is not None else "#"
                    score = item.get("score", 0.5)
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in advice_list:
                        if existing["query"] == query and existing["advice"] == advice_content:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        advice_content = await self._generate_advice_with_mistral(query, advice_content, language)
                        
                        advice_list.append({
                            "query": query,
                            "advice": advice_content,
                            "url": advice_url,
                            "score": score
                        })

            # Sort by score and take top results
            sorted_advice = sorted(advice_list, key=lambda x: x.get("score", 0), reverse=True)
            top_advice = sorted_advice[:3]
            
            return {
                "complaints": complaints_list,
                "advices": top_advice
            }
        
        except Exception as e:
            logger.error(f"Error extracting complaints: {e}", exc_info=True)
            return {"complaints": [], "advices": []}

    
    async def _extract_complaints_with_mistral(self, email_content: str, language: str) -> Dict:
        """
        Use Mistral AI to extract complaints and relevant information from email content.
        """
        # Truncate email content if too long to avoid token issues
        if len(email_content) > 1000:
            email_content = email_content[:1000] + "..."
        
        system_prompt = f"""Extract customer complaints and related queries from email in {language}.

    RULES:
    1. Extract complaints in third-person format using ACTUAL information from the email
    2. Include any real order numbers, subscription IDs, transaction IDs, account numbers, or reference codes mentioned in the email
    3. Generate 3-5 related search queries for knowledge base lookup
    4. Return VALID JSON only: {{"complaints": [...], "related_queries": [...]}}
    5. If no complaints, return empty arrays
    6. Do NOT use backslashes or escape characters in JSON keys
    7. Ensure JSON is properly formatted

    IMPORTANT: Extract ACTUAL identifiers from the email content, not placeholder examples."""
        
        user_prompt = f"""EMAIL CONTENT TO ANALYZE:
    {email_content}

    EXTRACTION INSTRUCTIONS:
    1. Look for and include any actual order numbers, subscription IDs, transaction codes, account numbers, or reference IDs mentioned in the email
    2. Extract complaints in third-person format using the real information from this email
    3. Generate related search queries that would help find solutions to these specific complaints

    CRITICAL: Use only the ACTUAL numbers, IDs, and information from the email content above. Do not use example placeholders like "#123".

    Return only valid JSON in this format:
    {{"complaints": ["Customer complaint with actual ID/number if present"], "related_queries": ["search term 1", "search term 2"]}}"""
        
        # Check token usage
        estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
        logger.info(f"Complaint extraction tokens: {estimated_tokens}")
        
        # Create API payload
        data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=250)
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
            return {"complaints": [], "related_queries": []}
        
        response_data = response.json()
        logger.info(f"Extraction response: {response_data}")
        
        try:
            content = response_data['message']['content']
            logger.info(f"Raw content: {content}")
            
            # Clean up the content to fix common JSON issues
            cleaned_content = self._clean_json_content(content)
            logger.info(f"Cleaned content: {cleaned_content}")
            
            complaints = json.loads(cleaned_content)
            
            # Validate the structure
            if not isinstance(complaints, dict):
                logger.warning("Response is not a dictionary, returning empty result")
                return {"complaints": [], "related_queries": []}
            
            # Ensure required keys exist
            if "complaints" not in complaints:
                complaints["complaints"] = []
            if "related_queries" not in complaints:
                complaints["related_queries"] = []
                
            return complaints
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed content: {content}")
            return {"complaints": [], "related_queries": []}
        except Exception as e:
            logger.error(f"Error processing extraction response: {e}")
            return {"complaints": [], "related_queries": []}
    
    def _clean_json_content(self, content: str) -> str:
        """
        Clean up common JSON formatting issues from AI responses.
        """
        # Remove any markdown formatting
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Fix escaped underscores in JSON keys (common AI mistake)
        content = re.sub(r'\\_', '_', content)
        
        # Remove extra whitespace
        content = content.strip()
        
        # Extract JSON object if there's extra text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        return content

    async def _generate_advice_with_mistral(self, query: str, document_content: str, language: str) -> str:
        """
        Generate advice using document content with token optimization
        """
        # Optimize document content for token limits
        if len(document_content) > 800:
            document_content = document_content[:800] + "..."
        
        system_prompt = f"""Answer query using ONLY document content in {language}.

RULES:
1. Use only facts from document
2. If no relevant info, return empty string
3. Be concise and accurate"""
        
        user_prompt = f"""DOCUMENT: {document_content}

QUERY: {query}

Answer using document info only."""
        
        # Check token usage
        estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
        if estimated_tokens > 1500:
            # Further reduce document content
            document_content = document_content[:400] + "..."
            user_prompt = f"""DOCUMENT: {document_content}
QUERY: {query}
Answer using document info only."""
        
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
            logger.error(f"Mistral API error for advice generation: {response.status_code}")
            return document_content
        
        response_data = response.json()
        
        try:
            answer = response_data['message']['content']
            if not answer or answer.strip() == "":
                return document_content
            return answer
        except Exception as e:
            logger.error(f"Error processing advice response: {e}")
            return document_content


    async def generate_complaint_response(self, sender_name: str, email_content: str, 
                                documents: List[Dict], language: str, 
                                template: str = None, 
                                complaint_regards: str = None) -> str:
        """Generate complaint response with template and custom regards support"""
        try:
            # Base absolute rules
            base_rules = f"""You are responding to a customer complaint in {language}. This is the ACTUAL REPLY email.

    ABSOLUTE RULES - VIOLATION IS FORBIDDEN:
    1. NEVER ask customer to contact support team or any other department - YOU ARE THE SUPPORT TEAM
    2. NEVER generate phone numbers, email addresses, URLs, website links, business hours, or any other contact details
    3. NEVER invent or infer information not explicitly present in the DOCUMENT CONTENT provided
    4. If the DOCUMENT CONTENT does not contain solution for a complaint, respond with: "I understand your concern about this issue. Let me escalate this internally for resolution."
    5. Show genuine empathy and take responsibility where appropriate
    6. Provide specific solutions from documents when available
    7. Use only first person ("I", "we", "our team")"""

            # Closing instructions
            template_closing = template if template else ''
            default_closing = 'Best regards,\nCustomer Service Team' if not complaint_regards else ''
            
            if complaint_regards:
                closing_instruction = "DO NOT add any closing/regards/signature — they will be added separately"
            else:
                closing_instruction = "End with a short, professional closing (e.g., 'Best regards, Customer Service Team')"

            # Build system prompt based on template and regards availability
            if template:
                system_prompt = f"""{base_rules}
    8. Follow this email template structure: {template_closing}
    9. {closing_instruction}

    Response format must be:
    I sincerely apologize for the inconvenience you've experienced.

    Issues and Solutions:
    1. [Issue]: [Solution from document OR "I understand your concern about this issue. Let me escalate this internally for resolution."]
    2. [Issue]: [Solution from document OR "I understand your concern about this issue. Let me escalate this internally for resolution."]

    We appreciate your patience."""
            else:
                system_prompt = f"""{base_rules}
    8. {closing_instruction}

    Response format must be:
    Dear {sender_name},

    I sincerely apologize for the inconvenience you've experienced.

    Issues and Solutions:
    1. [Issue]: [Solution from document OR "I understand your concern about this issue. Let me escalate this internally for resolution."]
    2. [Issue]: [Solution from document OR "I understand your concern about this issue. Let me escalate this internally for resolution."]

    We are committed to resolving your concerns.

    {default_closing}"""

            # Build user prompt with document content
            user_prompt = f"""CUSTOMER COMPLAINT FROM: {sender_name}
    EMAIL CONTENT: {email_content}

    AVAILABLE RESOLUTION DOCUMENTS:
    (Use ONLY the content provided below. DO NOT invent or infer solutions.)
    """
            
            if documents:
                for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents
                    user_prompt += f"\nDOCUMENT {i}: "
                    if doc.get('content'):
                        # Increased slice to 800 chars for better solution context
                        doc_content = doc['content'][:800].strip()
                        user_prompt += f'"{doc_content}"\n'
                    else:
                        user_prompt += "NO INFORMATION AVAILABLE\n"
                    if doc.get('url'):
                        user_prompt += f"Source: {doc['url']}\n"
                    user_prompt += "---\n"
            else:
                user_prompt += "NO RESOLUTION DOCUMENTS AVAILABLE\n"

            user_prompt += f"""
    CRITICAL INSTRUCTIONS:
    - Write complete apology email reply in {language}
    - Address each complaint issue using ONLY the document solutions provided above
    - If documents show "NO INFORMATION AVAILABLE" or no relevant solution, respond with "I understand your concern about this issue. Let me escalate this internally for resolution."
    - Show empathy and take responsibility
    - NEVER ask customer to contact support - YOU ARE THE SUPPORT
    - NEVER create phone numbers, emails, URLs, or any contact details
    - Format as numbered complaint issues and solutions
    - Be professional and solution-focused within strict document constraints"""

            # Check token usage
            estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
            if estimated_tokens > 7000:
                user_prompt = user_prompt[:4000] + "..."

            # Create API payload
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=800)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.0
            data["top_p"] = 0.1
            data["repetition_penalty"] = 1.1
            
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
                return f"Dear {sender_name}, I sincerely apologize for the inconvenience you've experienced. We are looking into your concerns and will resolve them promptly."
            
            response_data = response.json()
            content = response_data['message']['content'].strip()
            
            # Add regards if needed (exact phrase matching)
            if complaint_regards:
                content_lower = content.lower()
                if complaint_regards.lower() not in content_lower:
                    content = content.rstrip() + f"\n\n{complaint_regards}"
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating complaint response: {str(e)}", exc_info=True)
            return f"Dear {sender_name}, I sincerely apologize for the inconvenience you've experienced. We are looking into your concerns and will resolve them promptly."


    async def generate_ticket_data(self, sender_name: str, complaints: List[str], 
                             suggestions: List[Dict], language: str) -> Optional[TicketData]:
        """Generate structured ticket data using AI for title, description, and priority"""
        try:
            # Limit and truncate data for token efficiency
            limited_complaints = complaints[:5]  # Max 5 complaints
            limited_suggestions = suggestions[:3]  # Max 3 suggestions
            
            # Truncate long items
            short_complaints = [
                c[:150] + "..." if len(c) > 150 else c 
                for c in limited_complaints
            ]
            
            # Create structured prompt for JSON response
            system_prompt = f"""Generate structured ticket data in {language}. 

    PRIORITY LEVELS: Low, Medium, High, Urgent

    Return ONLY valid JSON with this exact structure:
    {{
    "title": "Brief ticket title (max 100 chars)",
    "description": "Detailed description of issues and context",
    "priority": "One of: Low, Medium, High, Urgent"
    }}

    PRIORITY GUIDELINES:
    - Low: Minor issues, cosmetic problems
    - Medium: Functional issues that don't block core features
    - High: Core functionality affected, business impact
    - Urgent: Critical system failures, data loss, security issues"""
            
            complaints_text = "\n".join([f"- {c}" for c in short_complaints])
            suggestions_text = "\n".join([
                f"- {s.get('suggestion', s.get('advice', ''))[:100]}" 
                for s in limited_suggestions
            ])
            
            user_prompt = f"""COMPLAINANT: {sender_name}

    ISSUES REPORTED:
    {complaints_text}

    SUGGESTED SOLUTIONS:
    {suggestions_text}

    Generate structured ticket data as JSON only. No extra text."""
            
            # Check tokens and create payload using utils
            estimated_tokens = self.estimate_tokens(system_prompt + user_prompt)
            logger.info(f"Ticket generation tokens: {estimated_tokens}")
            
            data = self.create_mistral_payload(system_prompt, user_prompt, max_tokens=300)
            data["model"] = MISTRAL_CONFIG['model']
            data["temperature"] = 0.2
            
            # Call API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    MISTRAL_CONFIG['service_url'],
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=MISTRAL_CONFIG['timeout']
                )
            
            if response.status_code != 200:
                logger.error(f"Ticket generation API error: {response.status_code}")
                return self._create_fallback_ticket(sender_name, short_complaints)
            
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "")
            
            if not content:
                return self._create_fallback_ticket(sender_name, short_complaints)
            
            # Parse and validate the AI response
            ticket_data = self._parse_ticket_response(content, sender_name, short_complaints)
            return ticket_data
            
        except Exception as e:
            logger.error(f"Error generating ticket: {e}", exc_info=True)
            return self._create_fallback_ticket(sender_name, short_complaints or complaints)

    def _parse_ticket_response(self, content: str, sender_name: str, complaints: List[str]) -> Optional[TicketData]:
        """Parse AI response and create TicketData object"""
        try:
            # Clean content similar to complaint extraction
            cleaned_content = self._clean_json_content(content)
            logger.info(f"Cleaned ticket content: {cleaned_content}")
            
            ticket_json = json.loads(cleaned_content)
            
            # Validate required fields
            if not all(key in ticket_json for key in ["title", "description", "priority"]):
                logger.warning("Missing required fields in AI response")
                return self._create_fallback_ticket(sender_name, complaints)
            
            # Validate priority level
            priority = ticket_json["priority"]
            if priority not in ["Low", "Medium", "High", "Urgent"]:
                logger.warning(f"Invalid priority: {priority}, defaulting to Medium")
                priority = "Medium"
            
            # Create TicketData object
            ticket_data = TicketData(
                title=ticket_json["title"][:100],  # Ensure max length
                description=ticket_json["description"],
                priority=priority
            )
            
            return ticket_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in ticket parsing: {e}")
            return self._create_fallback_ticket(sender_name, complaints)
        except Exception as e:
            logger.error(f"Error parsing ticket response: {e}")
            return self._create_fallback_ticket(sender_name, complaints)

    def _create_fallback_ticket(self, sender_name: str, complaints: List[str]) -> TicketData:
        """Create fallback ticket when AI generation fails"""
        issues_count = len(complaints) if complaints else 0
        
        return TicketData(
            title=f"Customer Issue Report - {sender_name}",
            description=f"Customer {sender_name} reported {issues_count} issue(s). Manual review required.\n\nIssues:\n" + 
                    "\n".join([f"- {c}" for c in complaints[:3]]) if complaints else "No specific details available.",
            priority= "Medium"
        )

    def _clean_json_content(self, content: str) -> str:
        """Clean up common JSON formatting issues from AI responses"""
        # Remove any markdown formatting
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Fix escaped underscores in JSON keys
        content = re.sub(r'\\_', '_', content)
        
        # Remove extra whitespace
        content = content.strip()
        
        # Extract JSON object if there's extra text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        return content


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
            
            # logger.info(f" query {query}   , response:{response}")

            # Process results
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                if score >= threshold:
                    results.append({
                         "content": hit['_source']['content'],
                         "url": hit['_source'].get('url', "#"),
                         "score": score
                    })
            
            logger.info(f"Vector search returned {len(results)} results above threshold {threshold}")
            
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            return []


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
                metadata_filters=metadata_filters,include_context= True,
                language_code=language_code
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
        language_code: str = "en",
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
                    "_source": ["content", "documentId", "chunkPosition", "totalChunks",  "chunkType"]
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
                            "url": hit['_source'].get('url', None),
                            "score": score
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
                            "url": hit['_source'].get('url', None),
                            "score": score
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


    def estimate_tokens(self,text: str) -> int:
        """Estimate token count for text (1 token ≈ 4 characters + 20% buffer)"""
        return int((len(text) / 4) * 1.2)

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

