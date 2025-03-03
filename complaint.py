import json
import httpx
import logging
import datetime
import asyncio
from typing import List, Dict, Any, Optional
from config import  ES_CONFIG, MISTRAL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

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
        

    async def extract_complaints(self, tenant_id: str, email_content: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Extract complaints from emails and search for matching solutions in Elasticsearch using Mistral AI.

        Args:
            email_content: The email content to analyze
            tenant_id: The tenant identifier to filter results
            language: The language of the email

        Returns:
            List of complaints, where each complaint is a dictionary with:
                - "question": The extracted complaint
                - "advices": List of advice objects containing "advice" and "url"
        """
        complaints_list = []

        # Extract complaints from the email using Mistral AI
        complaints = await self._extract_complaints_with_mistral(email_content, language)

        # If no complaints found, return an empty list
        if not complaints:
            return complaints_list

        # Search for matching Q&As for each complaint
        for complaint in complaints:
            logger.info(f"Processing complaint: {complaint}")
            embedding = await self.generate_embeddings(complaint)

            if not embedding:
                continue

            advices = await self.vector_search(
                embedding=embedding,
                tenant_id=tenant_id,
                threshold=0.55
            )

            # Append question and its corresponding advices to the list
            complaints_list.append({
                "question": complaint,
                "advices": advices if advices else []
            })

        return complaints_list



    async def _extract_complaints_with_mistral(self, email_content: str, language: str) -> List[str]:
        """
        Use Mistral AI to extract complaints from email content.
        
        Args:
            email_content: The content of the email to analyze
            language: The language of the email content
        
        Returns:
            List[str]: A list of extracted complaints
        """
        system_prompt = """You are a multilingual email analysis assistant specialized in identifying customer complaints.
    Your task is to find and extract ONLY the explicit complaints, issues, or expressions of dissatisfaction in an email.

    Be accurate and precise when identifying complaints. Focus on actual expressions of dissatisfaction, reported problems, 
    or negative experiences."""
        
        user_prompt = f"""
    ORIGINAL EMAIL:
    {email_content}

    TASK:
    Analyze the above email in {language} language. Extract ALL complaints being expressed.
    Focus on:
    - Issues described by the customer
    - Problems with products or services
    - Expressions of dissatisfaction
    - Reported malfunctions or errors
    - Any negative experiences mentioned

    Return your response as a JSON array of strings containing ONLY the complaints found in the ORIGINAL EMAIL:
    []

    If no complaints are found, return an empty array.
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
        
        logger.info(f"data mistral {data}")

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
        logger.info(f"response_data {response_data}")
        
        try:
            content = response_data['message']['content']
            complaints = json.loads(content)
            return complaints
        except Exception as e:
            logger.error(f"Error processing complaints: {e}")
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
            filter_conditions = [{"term": {"tenant_id": tenant_id}}]
            
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
                            "source": "cosineSimilarity(params.query_vector, 'content_vector')",
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