import datetime
import json
import os
import re
import uuid
import nltk
import httpx
import logging
import asyncio
import signal
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from confluent_kafka import Consumer, Producer, KafkaError, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from config import KAFKA_CONFIG, ES_CONFIG, MISTRAL_CONFIG
from deberta import EmailClassifier  # Import the function
from query import QueryProcessor
from complaint import ComplaintProcessor
from suggestion import SuggestionProcessor
from response import EmailClassificationDto, Advice,Complaint,TicketData
from contextvars import ContextVar

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

tracking_id_var = ContextVar("X-Tracking-ID", default="NA")


def detect_language(content):
    """
    Language detection that works with multiple languages.
    Falls back to basic detection if langdetect is not available.
    """
    try:
        from langdetect import detect
        return detect(content)
    except (ImportError, Exception) as e:
        logger.warning(f"Error using langdetect: {str(e)}. Falling back to basic detection.")
        
        # More sophisticated fallback than just checking for English
        # Note: This is a simplified approach - production systems should use a proper language detection library
        
        # Check for character sets that are distinctive to certain languages
        # Chinese/Japanese/Korean characters
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', content):
            # Distinguish between Chinese, Japanese and Korean
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):
                return 'ja'  # Japanese
            elif re.search(r'[\uac00-\ud7af]', content):
                return 'ko'  # Korean
            else:
                return 'zh'  # Chinese
        
        # Cyrillic (Russian, etc.)
        elif re.search(r'[\u0400-\u04FF]', content):
            return 'ru'  # Russian as default for Cyrillic
        
        # Arabic
        elif re.search(r'[\u0600-\u06FF]', content):
            return 'ar'
        
        # Greek
        elif re.search(r'[\u0370-\u03FF]', content):
            return 'el'
        
        # Latin-based languages - check for distinctive accented characters
        elif re.search(r'[áàâäãåāăąèéêëēėęíìîïīįıóòôöõøōőúùûüūųýÿźžż]', content):
            # Further refine detection using common words
            if re.search(r'\b(el|la|de|una|por|para|con)\b', content, re.IGNORECASE):
                return 'es'  # Spanish
            elif re.search(r'\b(le|la|et|pour|avec|vous)\b', content, re.IGNORECASE):
                return 'fr'  # French
            elif re.search(r'\b(und|der|die|das|mit|nicht)\b', content, re.IGNORECASE):
                return 'de'  # German
            elif re.search(r'\b(e|o|um|para|com|que)\b', content, re.IGNORECASE):
                return 'pt'  # Portuguese
            else:
                return 'und'  # Undefined Latin script language
        
        # Default to English for primarily ASCII text
        else:
            return 'en'



class MultilingualMessageProcessor:

    
    
    def __init__(self, model_path=None):
        logger.info("=" * 60)
        logger.info("INITIALIZING MULTILINGUAL MESSAGE PROCESSOR")
        logger.info("=" * 60)
        
        # Log system information
        logger.info(f"Server startup time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Process ID: {os.getpid()}")
       
    
        try:
            # Kafka configuration
            logger.info("Loading Kafka configuration...")
            self.consumer_config = {
                'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
                'group.id': KAFKA_CONFIG['group_id'],
                'auto.offset.reset': KAFKA_CONFIG.get('auto_offset_reset', 'earliest'),
                'enable.auto.commit': True,
                'session.timeout.ms': 45000,
                'heartbeat.interval.ms': 15000,
                'request.timeout.ms': 65000
            }
            
            self.producer_config = {
                'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers']
            }
            
            # Kafka topic
            self.topic = KAFKA_CONFIG['classification_request_topic']
            logger.info("✓ Kafka configuration loaded successfully")

            # Initialize HTTP client
            logger.info("Initializing HTTP client...")
            self.http_client = httpx.AsyncClient(timeout=30.0)
            logger.info("✓ HTTP client initialized successfully")
                

        
            # Initialize async Elasticsearch client
            logger.info("Initializing Elasticsearch client...")

            
            self.es_client = AsyncElasticsearch(
                ES_CONFIG['hosts'],
                basic_auth=(ES_CONFIG['username'], ES_CONFIG['password']),
                verify_certs=ES_CONFIG.get('verify_certs', True),
                ssl_show_warn=ES_CONFIG.get('ssl_show_warn', True),
                # ca_certs=ES_CONFIG.get('ca_certs'),  # Add this line
                retry_on_timeout=True,
                max_retries=3
            )
            logger.info("✓ Elasticsearch client initialized successfully")
                
            # Initialize SentenceTransformer with multilingual model
            model_name = 'paraphrase-multilingual-mpnet-base-v2'
            # model_path = models_path or os.path.join(os.getcwd(), 'models', 'sentence_transformer')
        
            try:
                if model_path:
                    logger.info(f"Loading model from local path: {model_path}")
                    self.st_model = SentenceTransformer(model_path)
                else:
                    logger.info(f"Loading model {model_name} from Hugging Face")
                    self.st_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {e}")
                raise
            

            self.categories = self._load_categories()
            

            # Create EmailProcessor instance
            self.query_processor = QueryProcessor(
                es_client=self.es_client,
                embedding_model=self.st_model
            )

            self.complaint_processor = ComplaintProcessor(
                es_client=self.es_client,
                embedding_model=self.st_model
            )

            self.suggestion_processor = SuggestionProcessor(
            )

            self.email_classifier = EmailClassifier()

            
            # Try to download nltk data for multiple languages
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK punkt: {str(e)}")
            
            
            # Initialize shutdown flag
            self.shutdown_requested = False
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("✓ Signal handlers configured for graceful shutdown")
            
            logger.info("=" * 60)
            logger.info("✓ MULTILINGUAL MESSAGE PROCESSOR INITIALIZED SUCCESSFULLY")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error("✗ FAILED TO INITIALIZE MULTILINGUAL MESSAGE PROCESSOR")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            raise


    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        self.shutdown_requested = True


    def _load_categories(self):
        try:
            with open('classification.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading categories: {str(e)}")
            return []
        


    async def publish_classification_to_kafka(self, tenant_id:str, topic: str, classification: EmailClassificationDto):
        """
        Publishes a document to a Kafka topic, preserving existing metadata.
        
        Args:
            tenant_id: Tenant identifier
            topic: Kafka topic to publish to
            classification: Email classification
            
        
        Returns:
            Future for the message delivery
        """
        

        # Serialize key and value
        serialized_key = str(tenant_id).encode("utf-8")  # Convert tenant_id to bytes
        # Convert Pydantic model to dict first, then to JSON string
        # Use model_dump() instead of dict() for Pydantic v2 compatibility
        try:
            # Try the new Pydantic v2 method first
            data = classification.model_dump()
        except AttributeError:
            # Fall back to the old method for Pydantic v1
            data = classification.dict()
        
        serialized_classification = json.dumps(data).encode("utf-8")

        # Create an asyncio Future to wait for delivery report
        future = asyncio.Future()
        
        def delivery_callback(err, msg):
            if err:
                future.set_exception(Exception(f"Message delivery failed: {err}"))
            else:
                future.set_result(msg)
        
        # Get current trackingId and ensure it's a string
        tracking_id = tracking_id_var.get() or "NA"
        if isinstance(tracking_id, bytes):
            tracking_id_str = tracking_id.decode("utf-8")
        else:
            tracking_id_str = str(tracking_id)

        self.producer.produce(topic, key=serialized_key, value=serialized_classification, callback=delivery_callback,headers=[("X-Tracking-ID", tracking_id_str)])
        self.producer.poll(1)  # Trigger delivery callbacks
        self.producer.flush()   # Ensure delivery
        
        return await future

    async def categorize_email_using_mistral(self, email_content: str, language: str,  department: Optional[str] = None) -> Dict[str, str]:
        """
        Process an email with a balanced approach:
        1. First classify the email type/subtype
        2. If it's a query, extract the question in a separate call
        3. Retrieve relevant documents and generate response if needed
        """
        try:
            # Step 1: Classify the email
            classification = await self._classify_email(email_content,language, department)
            
            email_type = classification.get("type")
            email_subtype = classification.get("subtype", "")
        
            return  email_type, email_subtype
            
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def _classify_email(self, email_content: str, detected_language: str, department: Optional[str] = None):
        """
        Classify email into type and subtype using Mistral
        This function handles emails in multiple languages including Japanese, Chinese,
        Spanish, French, Russian, Arabic, and others
        """
        department_str = ""
        if department:
            department_data = next((item for item in self.categories if item["sector"] == department), None)
            if department_data:
                department_str = f"For the {department} department with these types:\n{json.dumps(department_data['types'], indent=2)}\n\n"
        
       
        system_prompt = """You are a multilingual email classification assistant. Classify the email by type and subtype.
    You can handle emails in any language including English, Japanese, Chinese, Spanish, French, Russian, Arabic, and others.
    Identify the main intent regardless of the language used."""
        
        user_prompt = f"""Email content: {email_content}
        
        Language detected: {detected_language}
        
        {department_str}Classify this email as one of these types: "complaint", "query", "suggestion", or "spam".
        If a specific subtype applies, include it. Otherwise, set subtype to "".
        
        Understand the context and intent regardless of language.
        For non-Latin script languages (Japanese, Chinese, Arabic, Russian, etc.), analyze the message structure and key phrases.

        Respond ONLY with a JSON object like:
        {{
        "type": "selected_type",
        "subtype": "selected_subtype"
        }}"""
        
        # Call Mistral API with increased max_tokens to allow for better multilingual processing
        data = {
            "model": MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 200,  # Increased to give model more room for processing non-Latin languages
            "temperature": 0.2  # Lower temperature for more consistent classification results
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MISTRAL_CONFIG['service_url'],
                headers={
                    "Content-Type": "application/json",
                    "Accept-Charset": "UTF-8"  # Ensure proper character encoding for non-Latin scripts
                },
                json=data,
                timeout=MISTRAL_CONFIG['timeout']
            )
        
        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.status_code} - {response.text}")
            return {"type": "error", "subtype": ""}
        
        response_data = response.json()
    
        
        try:
            content = response_data['message']['content']
            clasification = json.loads(content)
            return clasification
        except Exception as e:
            logger.error(f"Error processing complaints: {e}")
            return []


    
    async def generate_embeddings(self, query: str) -> List[List[float]]:
        """Generate embeddings for a query using a thread pool"""
        try:
            start_time = datetime.datetime.now()
            
            # Move the embedding generation to a separate thread 
            # since SentenceTransformer is not async-compatible
            embeddings = await asyncio.to_thread(self._generate_embeddings_sync, query)
            
            end_time = datetime.datetime.now()
            logger.info(f"Generated {len(query)} embeddings in {(end_time - start_time).total_seconds()} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
    
    def _generate_embeddings_sync(self, query: str) -> List[List[float]]:
        """Synchronous method to generate embeddings (runs in a thread)"""
        embeddings = self.st_model.encode(query)
        return embeddings.tolist()



    def extract_sender_name_multilingual(email_data, language):
        """
        Extract sender name from email supporting multiple languages.
        
        Args:
            email_data: Raw email or email body text
            language:  language code to assist in extraction
        
        Returns:
            str: Extracted sender name or None if not found
        """
   

        
        # 1. Try signature patterns based on common formats across languages
        patterns = {
            'en': [
                r'(?:Best|Kind|Warm)?\s*(?:regards|wishes),\s*([^\n\r,\.]{2,50})',
                r'(?:Sincerely|Yours truly),\s*([^\n\r,\.]{2,50})',
                r'Thanks(?:,|\.| you,)\s*([^\n\r,\.]{2,50})',
                r'Cheers(?:,|\.)\s*([^\n\r,\.]{2,50})',
                r'From:\s*([^\n\r,\.]{2,50})'
            ],
            'es': [
                r'(?:Saludos|Atentamente|Cordialmente|Afectuosamente),\s*([^\n\r,\.]{2,50})',
                r'Gracias(?:,|\.)\s*([^\n\r,\.]{2,50})'
            ],
            'fr': [
                r'(?:Cordialement|Bien à vous|Salutations),\s*([^\n\r,\.]{2,50})',
                r'Merci(?:,|\.)\s*([^\n\r,\.]{2,50})'
            ],
            'de': [
                r'(?:Mit freundlichen Grüßen|Viele Grüße|Beste Grüße),\s*([^\n\r,\.]{2,50})',
                r'Danke(?:,|\.)\s*([^\n\r,\.]{2,50})'
            ],
            'ja': [
                r'(?:よろしくお願いします|敬具),\s*([^\n\r,\.]{2,50})',
                r'([^\n\r,\.]{2,30})(?:より|から)'
            ],
            'zh': [
                r'(?:此致|敬礼|祝好|顺祝|谢谢),\s*([^\n\r,\.]{2,30})',
                r'([^\n\r,\.]{2,30})(?:敬上|拜上)'
            ],
            'ru': [
                r'(?:С уважением|Искренне Ваш|С наилучшими пожеланиями),\s*([^\n\r,\.]{2,50})',
                r'(?:от|из|с),\s*([^\n\r,\.]{2,50})'
            ],
            'ar': [
                r'(?:مع تحياتي|مع أطيب التحيات|تفضلوا بقبول فائق الاحترام),\s*([^\n\r,\.]{2,50})',
                r'([^\n\r,\.]{2,50})(?:المرسل|من)'
            ],
            # Universal patterns that might work across languages
            'universal': [
                r'--+\s*([^\n\r<>@:;,\.]{2,50})',  # Name after a signature line
                r'\*\s*([^\n\r<>@:;,\.]{2,50})\s*\*',  # Name between asterisks
                r'[|]\s*([^\n\r<>@:;,\.]{2,50})'  # Name after a vertical bar
            ]
        }
        
        # Add language-specific patterns first if language is specified
        if language and language in patterns:
            specific_patterns = patterns[language]
        else:
            specific_patterns = []
        
        # Try all patterns - prioritize specific language if known, then try English, then universal
        all_patterns = specific_patterns + \
                    ([] if language == 'en' else patterns['en']) + \
                    patterns['universal']
        
        for pattern in all_patterns:
            try:
                match = re.search(pattern, email_data, re.IGNORECASE | re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    # Validate name - at least 2 chars, no email addresses
                    if len(name) >= 2 and '@' not in name:
                        return name
            except Exception as e:
                print(f"Error with pattern {pattern}: {e}")
        
        # 4. Try to use NLP for language-specific name extraction if available
        try:
            import spacy
            
            # Map language codes to available spaCy models
            lang_models = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'fr': 'fr_core_news_sm',
                'es': 'es_core_news_sm',
                'ja': 'ja_core_news_sm',
                'zh': 'zh_core_web_sm',
                'ru': 'ru_core_news_sm'
            }
            
            # Use language-specific model if available
            if language in lang_models:
                nlp = spacy.load(lang_models[language])
                
                # Process the text
                doc = nlp(email_data)
                
                # Look for person names in the last few sentences (likely signature area)
                sentences = list(doc.sents)
                potential_signature = " ".join([str(sent) for sent in sentences[-3:]])
                
                # Find person entities in the potential signature
                signature_doc = nlp(potential_signature)
                person_entities = [ent.text for ent in signature_doc.ents if ent.label_ == "PERSON"]
                
                if person_entities:
                    return person_entities[0]  # Return the first person entity found
        except ImportError:
            print("spaCy not available for NLP-based extraction")
        except Exception as e:
            print(f"Error in NLP extraction: {e}")
        
        return None
   

    async def process_email_message(self, message):
        """Process individual message and store in Elasticsearch with vectors"""
        logger.info(f"Message complete: {message}")

        tenant_id = message.get('tenantId')
        thread_id = message.get('threadId')
        message_id = message.get('messageId')
        department: Optional[str] = message.get('department')
        sender_name = message.get('senderName')
        
        # Extract text content from message
        content = message.get('emailBody') or ""
        subject = message.get('subject') or ""
        # After cleaning (empty strings remain empty)
        content = content.replace('\n', ' ').replace('\r', ' ')  # "" stays ""
        subject = subject.replace('\n', ' ').replace('\r', ' ')  # "" stays ""
        
        query_ai_mode = message.get('queryAIMode') or ""
        query_reply_template = message.get('queryReplyTemplate') or ""
        query_regards = message.get('queryRegards') or ""
        
        complaint_ai_mode = message.get('complaintAIMode') or ""
        complaint_reply_template = message.get('complaintReplyTemplate') or ""
        auto_complaint_ticket_generation = message.get('autoComplaintTicketGeneration') or False
        complaint_regards = message.get('complaintRegards') or ""
        
        suggestion_ai_mode = message.get('suggestionAIMode') or ""
        suggestion_reply_template = message.get('suggestionReplyTemplate') or ""
        suggestion_regards = message.get('suggestionRegards') or ""

        # Build complete_content intelligently
        if subject and content:
            complete_content = f"Subject: {subject}, Body: {content}"
        elif subject:
            complete_content = f"Subject: {subject}"
        elif content:
            complete_content = content
        else:
            complete_content = ''

        complete_content = complete_content.replace('\n', ' ').replace('\r', ' ')

        if not complete_content:
            logger.warning(f"Email Message has no content, tenantId: {tenant_id}, threadId: {thread_id}")
            return

        # Step 1: Basic classification - this must succeed or we go to DLQ
        try:
            language = detect_language(content)
            
            if not sender_name:
                sender_name = self.extract_sender_name_multilingual(content, language)

            logging.info(f"language: {language} department: {department}, complete_content {complete_content}")

            # Perform classification
            if language == 'en' and department:
                type, subtype = await self.email_classifier.process_emails(content, department)
            else:
                type, subtype = await self.categorize_email_using_mistral(complete_content, language, department)

            logger.info(f"Email Type: {type}, SubType: {subtype}")
            
        except Exception as e:
            # Classification failed - this should go to DLQ
            logger.error(f"Classification failed for message - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
            raise  # Re-raise to trigger DLQ handling
        
        # Step 2: Create basic classification object (fallback in case of processing errors)
        basic_classification = EmailClassificationDto(
            tenantId=tenant_id, 
            threadId=thread_id, 
            messageId=message_id, 
            senderName=sender_name,
            type=type, 
            subType=subtype
        )
        
        # Step 3: Enhanced processing based on type - if this fails, we'll use basic classification
        enhanced_classification = None
        processing_error = None
        
        try:
            if type == "query":
                enhanced_classification = await self._process_query_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, language, query_ai_mode, query_reply_template, query_regards
                )
            elif type == "complaint":
                enhanced_classification = await self._process_complaint_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, subject, content, language, complaint_ai_mode, 
                    complaint_reply_template, auto_complaint_ticket_generation, complaint_regards
                )
            elif type == "suggestion":
                enhanced_classification = await self._process_suggestion_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, language, suggestion_ai_mode, 
                    suggestion_reply_template, suggestion_regards
                )
            else:
                # For spam or other types, use basic classification
                enhanced_classification = basic_classification
                
        except Exception as e:
            processing_error = e
            logger.error(f"Enhanced processing failed for {type} - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
        
        # Step 4: Use enhanced classification if successful, otherwise use basic
        final_classification = enhanced_classification if enhanced_classification else basic_classification
        
        # Step 5: Publish the classification (always succeeds with at least basic classification)
        try:
            await self.publish_classification_to_kafka(tenant_id, KAFKA_CONFIG['classification_response_topic'], final_classification)
            
            if processing_error:
                logger.info(f"Classification published with basic data due to processing error - tenantId: {tenant_id}, threadId: {thread_id}")
            else:
                logger.info(f"Classification published successfully - tenantId: {tenant_id}, threadId: {thread_id}")
                
        except Exception as e:
            # Even publishing failed - this is a system error
            logger.error(f"Failed to publish classification - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
            raise  # Re-raise to trigger DLQ handling

    async def _process_query_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                 complete_content, language, query_ai_mode, query_reply_template, query_regards):
        """Process query type emails with enhanced data extraction"""
        
        # If query reply generation is not required
        if query_ai_mode in ["no_reply", "template_only"]:
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype
            )
        
        # Extract all questions from the email and generate responses
        query_response = await self.query_processor.extract_query_generate_responses(
            tenant_id, 
            thread_id, 
            sender_name, 
            complete_content, 
            language, 
            type, 
            subtype, 
            query_ai_mode, 
            query_reply_template,
            query_regards
        )
        
        # Determine response content
        response_content = None
        if query_response:
            response_content = query_response.get("message", {}).get("content")
            # Fallback if content is empty or None
            if not response_content:
                response_content = "No answer found"
        
        # Create classification with or without response content
        if response_content and response_content != "No answer found":
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype, 
                queryResponse=response_content
            )
        else:
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype
            )

    async def _process_complaint_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                    complete_content, subject, content, language, complaint_ai_mode, 
                                    complaint_reply_template, auto_complaint_ticket_generation, complaint_regards):
        """Process complaint type emails with enhanced data extraction"""
        
        # If complaint reply generation is not required
        if complaint_ai_mode in ["no_reply"] or complaint_ai_mode in ["template_only"] and not auto_complaint_ticket_generation:
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype
            )
        
        # Step 1: Extract complaints and get relevant documents
        complaint_list = await self.complaint_processor.extract_complaints(tenant_id, complete_content, language)
        logger.info(f"complaint_list: {complaint_list}")
        
        complaint_response_content = None
        
        if complaint_list:
            complaints = complaint_list.get('complaints', [])
            advices_data = complaint_list.get('advices', [])

            # Create advice objects from the existing advices
            advices_list = [
                Advice(
                    query=advice.get("query", ""), 
                    advice=advice.get("advice", ""), 
                    url=advice.get("url", "#")
                ) 
                for advice in advices_data
            ]
            
            # Create the complaint object
            complaint_object = Complaint(
                complaints=complaints,  
                advises=advices_list
            )
            
            # Step 2: Generate complaint response if AI mode requires it
            if complaint_ai_mode not in ["no_reply", "template_only"]:
                # Convert advices_data to documents format for response generation
                documents_for_response = [
                    {
                        "content": advice.get("advice", ""),
                        "url": advice.get("url", "#")
                    }
                    for advice in advices_data
                ]
                
                complaint_response = await self.complaint_processor.generate_complaint_response(
                    sender_name=sender_name,
                    email_content=complete_content,
                    documents=documents_for_response,
                    language=language,
                    template=complaint_reply_template,
                    complaint_regards=complaint_regards
                )

                if complaint_response:
                    complaint_response_content = complaint_response  # It's already a string
                    if not complaint_response_content or complaint_response_content.strip() == "":
                        complaint_response_content = "No response generated"
                        
            # Step 3: Generate ticket body if auto ticket generation is enabled
            ticket_data = None
            if auto_complaint_ticket_generation:
                try:
                    ticket_data = await self.complaint_processor.generate_ticket_data(
                        sender_name=sender_name,
                        complaints=complaints,
                        suggestions=advices_data,
                        language=language
                    )
                    logger.info(f"Generated ticket data: {ticket_data}")
                except Exception as e:
                    logger.error(f"Error generating ticket data: {e}")
                    # Create fallback ticket
                    ticket_data = TicketData(
                        title=f"{subject}",
                        description=f"{content}",
                        priority="Medium"
                    )
            
            # Step 4: Create classification with structured ticket data
            classification_data = {
                "tenantId": tenant_id,
                "threadId": thread_id,
                "messageId": message_id,
                "senderName": sender_name,
                "type": type,
                "subType": subtype,
                "complaint": complaint_object
            }
            
            # Add complaint response if generated
            if complaint_response_content and complaint_response_content != "No response generated":
                classification_data["complaintResponse"] = complaint_response_content
            
            # Add structured ticket data if generated
            if ticket_data:
                classification_data["ticketData"] = ticket_data

            return EmailClassificationDto(**classification_data)
            
        else:
            # No complaints found
            return EmailClassificationDto(
                tenantId=tenant_id,
                threadId=thread_id,
                messageId=message_id,
                senderName=sender_name,
                type=type,
                subType=subtype
            )

    async def _process_suggestion_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                     complete_content, language, suggestion_ai_mode, 
                                     suggestion_reply_template, suggestion_regards):
        """Process suggestion type emails with enhanced data extraction"""
        
        # If suggestion reply generation is not required
        if suggestion_ai_mode in ["no_reply", "template_only"]:
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype
            )
        
        # Step 1: Extract suggestions (no document search needed)
        suggestions_list = await self.suggestion_processor.extract_suggestions(
            email_content=complete_content,
            type=type,
            subType=subtype,
            language=language
        )
        logger.info(f"extracted suggestions: {suggestions_list}")
        
        suggestion_response_content = None
        
        if suggestions_list:
            # Step 2: Generate acknowledgment response if AI mode requires it
            if suggestion_ai_mode not in ["no_reply", "template_only"]:
                suggestion_response = await self.suggestion_processor.generate_suggestion_response(
                    sender_name=sender_name,
                    email_content=complete_content,
                    suggestions=suggestions_list,  # Pass the suggestions directly
                    language=language,
                    template=suggestion_reply_template,
                    suggestion_regards=suggestion_regards
                )
                
                if suggestion_response:
                    # The method now returns a string directly
                    suggestion_response_content = suggestion_response
                    
                    # Ensure we have valid content
                    if not suggestion_response_content or suggestion_response_content.strip() == "None":
                        suggestion_response_content = None
            
            # Step 3: Create classification with suggestions list and optional response
            classification_data = {
                "tenantId": tenant_id,
                "threadId": thread_id,
                "messageId": message_id,
                "senderName": sender_name,
                "type": type,
                "subType": subtype,
                "suggestions": suggestions_list  # Store as simple list
            }
            
            # Add suggestion response if generated and valid
            if suggestion_response_content and suggestion_response_content.strip():
                classification_data["suggestionResponse"] = suggestion_response_content
            
            return EmailClassificationDto(**classification_data)
            
        else:
            # No suggestions found
            return EmailClassificationDto(
                tenantId=tenant_id,
                threadId=thread_id,
                messageId=message_id,
                senderName=sender_name,
                type=type,
                subType=subtype
            )

    async def consume_messages(self):
        """Consume messages from Kafka"""
        consumer = Consumer(self.consumer_config)
        
        try:
            # Subscribe to topic
            consumer.subscribe([self.topic])
            logger.info(f"✓ Successfully subscribed to Kafka topic: {self.topic}")
            logger.info("🔄 Starting message consumption loop...")
            
            message_count = 0
            last_heartbeat = datetime.datetime.now()
            
            while not self.shutdown_requested:
                msg = consumer.poll(1.0)
                
                # Send periodic heartbeat logs
                now = datetime.datetime.now()
                if (now - last_heartbeat).seconds >= 30:  # Every 30 seconds
                    logger.info(f"💓 Server heartbeat - Status: RUNNING | Messages processed: {message_count}")
                    last_heartbeat = now
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"✗ Kafka consumer error: {msg.error()}")
                    continue
                
                # Process message
                value = None
                try:
                    value = msg.value()
                    if isinstance(value, bytes):
                        value = json.loads(value.decode('utf-8'))
                    elif isinstance(value, str):
                        value = json.loads(value)
                    
                    message_count += 1
                    tenant_id = value.get('tenantId', 'unknown')
                    thread_id = value.get('threadId', 'unknown')
                    
                    # --- Set the tracking ID from Kafka headers before logging ---
                    kafka_headers = dict(msg.headers() or [])
                    tracking_id = kafka_headers.get('X-Tracking-ID', b'NA')
                    tracking_id_var.set(tracking_id.decode('utf-8') if isinstance(tracking_id, bytes) else str(tracking_id))

                    logger.info(f"📨 Processing message #{message_count} | Tenant: {tenant_id} | Thread: {thread_id}")
                    
                    # Process the email message - this now handles graceful degradation internally
                    await self.process_email_message(value)
                    
                    logger.info(f"✅ Successfully processed message #{message_count} for tenant: {tenant_id}")
                    
                except Exception as e:
                    # Only critical failures (like classification failure or publishing failure) reach here
                    request_id = str(uuid.uuid4())
                    logger.error(f"✗ Critical error processing message #{message_count} | Request ID: {request_id} | Error: {str(e)}", exc_info=True)
                    await self.send_to_dead_letter_queue(request_id, value, str(e))
                    
        except KeyboardInterrupt:
            pass
        finally:
            # Close the consumer
            consumer.close()


    
    

    def safe_json_deserializer(self, x):
        """Safely deserialize JSON, return None if invalid"""
        try:
            return json.loads(x.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {x}. Error: {e}")
            return {"raw_message": x.decode("utf-8"), "error": str(e)}

    async def send_to_dead_letter_queue(self, request_id, message, error):
        """Send problematic messages to a dead letter topic"""
        error_message = {
            "original_message": message,
            "error": error,
            "request_id":request_id
        }
        
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Failed to send to dead letter queue: {err}")
            else:
                logger.info(f"Message sent to classification_request_dlq_topic")
        
        self.producer.produce(KAFKA_CONFIG['classification_request_dlq_topic'], json.dumps(error_message).encode('utf-8'), callback=delivery_callback)
        self.producer.poll(1)  # Trigger delivery callbacks
        self.producer.flush()   # Ensure delivery

    async def consume_messages(self):
        """Consume messages from Kafka"""
        consumer = Consumer(self.consumer_config)
        
        try:
            # Subscribe to topic
            consumer.subscribe([self.topic])
            logger.info(f"✓ Successfully subscribed to Kafka topic: {self.topic}")
            logger.info("🔄 Starting message consumption loop...")
            
            message_count = 0
            last_heartbeat = datetime.datetime.now()
            
            while not self.shutdown_requested:
                msg = consumer.poll(1.0)
                
                # Send periodic heartbeat logs
                now = datetime.datetime.now()
                if (now - last_heartbeat).seconds >= 30:  # Every 30 seconds
                    logger.info(f"💓 Server heartbeat - Status: RUNNING | Messages processed: {message_count}")
                    last_heartbeat = now
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"✗ Kafka consumer error: {msg.error()}")
                    continue
                
                # Process message
                try:
                    value = msg.value()
                    if isinstance(value, bytes):
                        value = json.loads(value.decode('utf-8'))
                    elif isinstance(value, str):
                        value = json.loads(value)
                    
                    message_count += 1
                    tenant_id = value.get('tenantId', 'unknown')
                    thread_id = value.get('threadId', 'unknown')
                    
                    # --- Set the tracking ID from Kafka headers before logging ---
                    kafka_headers = dict(msg.headers() or [])
                    tracking_id = kafka_headers.get('X-Tracking-ID', b'NA')
                    tracking_id_var.set(tracking_id.decode('utf-8') if isinstance(tracking_id, bytes) else str(tracking_id))

                    logger.info(f"📨 Processing message #{message_count} | Tenant: {tenant_id} | Thread: {thread_id}")
                    
                    await self.process_email_message(value)
                    
                    logger.info(f"✅ Successfully processed message #{message_count} for tenant: {tenant_id}")
                    
                except Exception as e:
                    request_id = str(uuid.uuid4())
                    logger.error(f"✗ Error processing message #{message_count} | Request ID: {request_id} | Error: {str(e)}", exc_info=True)
                    await self.send_to_dead_letter_queue(request_id, value, str(e))
                    
        except KeyboardInterrupt:
            pass
        finally:
            # Close the consumer
            consumer.close()

    async def run(self):
        """Main processing loop"""
        logger.info("=" * 60)
        logger.info("STARTING MULTILINGUAL MESSAGE PROCESSOR SERVER")
        logger.info("=" * 60)
        
        try:
            # Perform health check
            if not await self.health_check():
                logger.error("✗ Health check failed. Cannot start server.")
                return
            
            # Initialize producer
            logger.info("Initializing Kafka producer...")
            self.producer = Producer(self.producer_config)
            logger.info("✓ Kafka producer initialized successfully")
            
            logger.info("=" * 60)
            logger.info("🚀 SERVER STARTED SUCCESSFULLY!")
            logger.info(f"📧 Listening for messages on topic: {self.topic}")
            logger.info(f"👥 Consumer group: {self.consumer_config['group.id']}")
            logger.info(f"🏥 Server status: HEALTHY")
            logger.info(f"⏰ Server ready at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            
            # Start consuming messages
            await self.consume_messages()
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("✗ FATAL ERROR IN MAIN LOOP")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            raise
        finally:
            await self.shutdown()

    async def health_check(self):
        """Perform health check on all components"""
        logger.info("Performing health check...")
        
        try:
            # Check Elasticsearch connection
            logger.info("Checking Elasticsearch connection...")
            es_info = await self.es_client.info()
            logger.info(f"✓ Elasticsearch connection healthy - Version: {es_info['version']['number']}")
            
            # Check Kafka connection by creating a test consumer
            logger.info("Checking Kafka connection...")
            test_consumer = Consumer(self.consumer_config)
            topics = test_consumer.list_topics(timeout=5)
            test_consumer.close()
            logger.info(f"✓ Kafka connection healthy - Available topics: {len(topics.topics)}")
            
            logger.info("✓ All health checks passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Health check failed: {str(e)}")
            return False


    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("=" * 60)
        logger.info("🛑 INITIATING GRACEFUL SHUTDOWN")
        logger.info("=" * 60)
        
        try:
            # Close the HTTP client
            logger.info("Closing HTTP client...")
            await self.http_client.aclose()
            logger.info("✓ HTTP client closed successfully")
            
            # Close the Elasticsearch client
            logger.info("Closing Elasticsearch client...")
            await self.es_client.close()
            logger.info("✓ Elasticsearch client closed successfully")
            
            # Ensure all messages are delivered before shutting down producer
            logger.info("Flushing Kafka producer...")
            if hasattr(self, 'producer'):
                self.producer.flush()
                logger.info("✓ Kafka producer flushed successfully")
            
            logger.info("=" * 60)
            logger.info("✅ GRACEFUL SHUTDOWN COMPLETED")
            logger.info(f"🕐 Shutdown completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"✗ Error during shutdown: {str(e)}", exc_info=True)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🌟 MULTILINGUAL MESSAGE PROCESSOR - STARTING UP")
    logger.info("=" * 80)
    
    try:
        # Initialize processor
        processor = MultilingualMessageProcessor()
        
        # Run the processor in an asyncio event loop
        asyncio.run(processor.run())
        
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
    except Exception as e:
        logger.error(f"💥 Application crashed: {str(e)}", exc_info=True)
    finally:
        logger.info("🏁 Application shutdown complete")
        logger.info("=" * 80)