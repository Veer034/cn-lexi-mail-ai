import json
import httpx
import logging
import datetime
import asyncio
from typing import List, Dict, Any, Optional
from config import  ES_CONFIG, MISTRAL_CONFIG

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

class ComplaintExtractor:
    def __init__(self, es_client=None, embedding_model=None):
        """
        Initialize the ComplaintExtractor with required clients and models.
        
        Args:
            es_client: Elasticsearch client
            embedding_model: Model for generating embeddings
        """
        self.es_client = es_client
        self.embedding_model = embedding_model
        

    async def extract_complaints(self, tenant_id: str, email_content: str, language: str) -> List[Dict]:
        """
        Extract complaints and related information from an email
        
        Args:
            tenant_id: Tenant identifier for vector search context
            email_content: Content of the email to analyze
            language: Detected language of the email
            
        Returns:
            List[Dict]: List of complaints with related advice
        """
        try:
            # Extract complaints and related queries using Mistral
            extraction_result = await self._extract_complaints_with_mistral(email_content, language)
            
            complaints_list = extraction_result.get("complaints", [])
            related_queries = extraction_result.get("related_queries", [])

            if not complaints_list:
                logger.info("No complaints found in the email")
                return []
            
            result = {}
            

            # Perform vector search for each complaint
            advice_list = []
            
            
            metadata_filters = {}
            if language and language != "unknown":
                metadata_filters["language"] = language
            
            # Then search using related queries for additional context
            for query in related_queries:
                query_embedding = await self.generate_embeddings(query)
                
                # Use the vector_search method with the correct parameters
                query_results = await self.vector_search(
                    embedding=query_embedding,
                    tenant_id=tenant_id,
                    top_k=2,
                    threshold=0.65,
                    metadata_filters=metadata_filters
                )
                
                logger.info(f" query_results : {query_results}")

            
                for item in query_results:
                    advice_content = item.get("content", "")
                    advice_url = item.get("url", "#") if item.get("url") is not None else "#"
                    score = item.get("score", 0.5)  # Get the score from the vector search result
                    
                    # Check for duplicates based on content for the same query
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
                            "score": score  # Use the actual score from vector search
                        })

            # Sort by score (priority) and take top results
            sorted_advice = sorted(advice_list, key=lambda x: x.get("score", 0), reverse=True)
            top_advice = sorted_advice[:3]  # Limit to top 3 pieces of advice
            
            # Format the final result
            result = {
                "complaints": complaints_list,
                "advices": top_advice
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error extracting complaints: {e}", exc_info=True)
            return []

    async def _extract_complaints_with_mistral(self, email_content: str, language: str) -> Dict:
        """
        Use Mistral AI to extract complaints and relevant information from email content.
        
        Args:
            email_content: The content of the email to analyze
            language: The language of the email content
        
        Returns:
            Dict: Contains 'complaints' with extracted issues and 'related_queries' with common questions
        """
        system_prompt = """You are a multilingual email analysis assistant specialized in identifying customer issues and relevant information.
    Your task is to carefully analyze an email and extract two types of data:

    1. COMPLAINTS: Actual expressions of dissatisfaction or problems reported by the customer. Extract the complaint in third-person format with relevant details (like order numbers, dates, product names).

    2. RELATED QUERIES: Common questions or topics that this customer might need help with, based on their complaint. These should be worded as search queries that could be used to find relevant information in a knowledge base.

    Be precise and focused on the actual content provided in the email."""
        
        user_prompt = f"""
    ORIGINAL EMAIL:
    {email_content}

    TASK:
    Analyze the above email in {language} language. 

    STEP 1: Extract ALL specific complaints being expressed by the customer, rewritten in third-person format, including any relevant identifiers like order numbers or dates. For example, if the email says "I received damaged items from order #12345", extract "Customer received damaged items from order #12345".

    STEP 2: Based on the complaints, generate 3-5 related search queries that would be useful to resolve the customer's issue. These should be short phrases someone might search for in a knowledge base.

    Return your response as a JSON object with two arrays:
    {{
    "complaints": ["complaint1", "complaint2", ...],
    "related_queries": ["how to resolve issue ", "steps to fix issue","damaged product policy", ...]
    }}

    If no complaints are found, return empty arrays for both fields.
    """
        
        # Call Mistral API
        data = {
            "model": MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 200,
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
            return {"complaints": [], "related_queries": []}
        
        response_data = response.json()
        logger.info(f"response_data {response_data}")
        
        try:
            content = response_data['message']['content']
            complaints = json.loads(content)
            return complaints
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return {"complaints": [], "related_queries": []}


    async def _generate_advice_with_mistral(self, query: str, document_content: str, language: str) -> str:
        """
        Use Mistral AI to generate an answer for the query using only information present in the document content.
        
        Args:
            query: The user's question
            document_content: The content of the document to use as context
            language: The language to generate the answer in
        
        Returns:
            str: The answer generated from document content or the original document content if no answer could be found
        """
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided document content.
    Your task is to:
    1. Read the document content carefully
    2. Look for information relevant to the query
    3. If relevant information exists, provide a clear answer using ONLY facts from the document
    4. Do NOT add any information not present in the document
    5. If you cannot find relevant information to answer the query, return an empty string
    """
        
        user_prompt = f"""
    DOCUMENT CONTENT:
    {document_content}

    QUERY:
    {query}

    TASK:
    Answer the above query based ONLY on information present in the document content.
    Generate your answer in {language} language.
    Do not add any information that is not explicitly stated in the document.
    If you cannot find enough information to answer the query, return an empty string.
    """
        
        # Call Mistral API
        data = {
            "model": MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 300,
            "temperature": 0.1  # Lower temperature for more focused output
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
            return document_content
        
        response_data = response.json()
        logger.info(f"response_data {response_data}")
        
        try:
            answer = response_data['message']['content']
            # If answer is empty, return original document
            if not answer or answer.strip() == "":
                return document_content
            return answer
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return document_content


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
                index=ES_CONFIG['index_name'],
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


    async def generate_embeddings(self, query: str) -> List[List[float]]:
        """Generate embeddings for a query using a thread pool"""
        try:
            start_time = datetime.datetime.now()
            
            # Move the embedding generation to a separate thread 
            # since SentenceTransformer is not async-compatible
            embeddings = await asyncio.to_thread(self._generate_embeddings_sync, query)
            
            end_time = datetime.datetime.now()
            logger.info(f"Generated {len(query)} complaint embeddings in {(end_time - start_time).total_seconds()} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating complaint embeddings: {str(e)}", exc_info=True)
            raise
    
    def _generate_embeddings_sync(self, query: str) -> List[List[float]]:
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