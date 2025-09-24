import datetime
import json
import os
import re
import uuid
import nltk
import httpx
import logging
import asyncio
from asyncio import Semaphore
import time
from datetime import datetime, timedelta
import signal
import psutil
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from confluent_kafka import Consumer, Producer, KafkaError, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from config import KAFKA_CONFIG, ES_CONFIG, MISTRAL_CONFIG
from deberta import EmailClassifier
from lang_utils import LangUtil
from query import QueryProcessor
from complaint import ComplaintProcessor
from suggestion import SuggestionProcessor
from response import EmailClassificationDto, Advice, Complaint, TicketData
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

tracking_id_var = ContextVar("X-Tracking-ID", default="NA")

@dataclass
class SystemConfig:
    """System configuration based on hardware capabilities"""
    cpu_cores: int
    has_gpu: bool
    thread_pool_size: int
    embedding_batch_size: int
    kafka_poll_timeout: float
    max_concurrent_messages: int = 3  # Fixed to 3 for simplicity

class HardwareDetector:
    """Detect system capabilities and configure accordingly"""
    
    @staticmethod
    def detect_system_config() -> SystemConfig:
        """Detect system capabilities and create optimal configuration"""
        cpu_cores = psutil.cpu_count(logical=False) or 4
        logical_cores = psutil.cpu_count(logical=True) or 8
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU Detection
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                logger.info(f"GPU detected: CUDA available")
        except ImportError:
            logger.info("No GPU libraries available")
        
        # Simple configuration - always use 3 parallel messages
        config = SystemConfig(
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
            max_concurrent_messages=3,  # Fixed to 3
            thread_pool_size=min(cpu_cores, 6),
            embedding_batch_size=16,
            kafka_poll_timeout=0.1
        )
        
        logger.info(f"System Config - CPU Cores: {cpu_cores}, GPU: {has_gpu}")
        logger.info(f"Max Concurrent Messages: 3 (fixed)")
        logger.info(f"Thread Pool: {config.thread_pool_size}")
        
        return config

def detect_language(content):
    """Language detection with fallback"""
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
        
        # Detect system capabilities
        self.system_config = HardwareDetector.detect_system_config()
        
        logger.info("=" * 60)
        logger.info("MULTILINGUAL MESSAGE PROCESSOR INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)

        # Log system information
        logger.info(f"Server startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Process ID: {os.getpid()}")
    
        try:
            # Thread pool for CPU-bound operations
            self.thread_pool = ThreadPoolExecutor(max_workers=self.system_config.thread_pool_size)
            logger.info(f"Thread pool initialized with {self.system_config.thread_pool_size} workers")
            
            # Simple semaphore for exactly 3 concurrent messages
            self.processing_semaphore = asyncio.Semaphore(3)
            
            # Track processing messages for graceful shutdown
            self.active_tasks = set()
            
            # Kafka configuration with optimized settings
            logger.info("Loading Kafka configuration...")
            self.consumer_config = {
                'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
                'group.id': KAFKA_CONFIG['group_id'],
                'auto.offset.reset': KAFKA_CONFIG.get('auto_offset_reset', 'earliest'),
                'enable.auto.commit': True,  # Let Kafka handle commits automatically
                'auto.commit.interval.ms': 5000,
                'session.timeout.ms': 45000,
                'heartbeat.interval.ms': 15000,
                'request.timeout.ms': 65000,
                'fetch.min.bytes': 1024,
                'fetch.wait.max.ms': 500
            }
            
            self.producer_config = {
                'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
                'linger.ms': 5,
                'compression.type': 'snappy',
                'batch.size': 16384
            }
            
            # Kafka topic
            self.topic = KAFKA_CONFIG['classification_request_topic']
            logger.info("Kafka configuration loaded successfully")

            # Initialize HTTP client with connection pooling
            logger.info("Initializing HTTP client...")
            self.http_client = httpx.AsyncClient(
                timeout=120.0,  # Increased timeout for Mistral responses
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            logger.info("HTTP client initialized successfully")

            # Initialize async Elasticsearch client
            logger.info("Initializing Elasticsearch client...")

            
            self.es_client = AsyncElasticsearch(
                ES_CONFIG['hosts'],
                basic_auth=(ES_CONFIG['username'], ES_CONFIG['password']),
                verify_certs=ES_CONFIG.get('verify_certs', True),
                ssl_show_warn=ES_CONFIG.get('ssl_show_warn', True),
                # ca_certs=ES_CONFIG.get('ca_certs'),
                retry_on_timeout=True,
                max_retries=3,
                http_compress=True
            )
            logger.info("Elasticsearch client initialized successfully")

            self.MISTRAL_CONFIG = MISTRAL_CONFIG    

            # Initialize SentenceTransformer with multilingual model
            model_name = 'paraphrase-multilingual-mpnet-base-v2'
            self.model_path = model_path
            self.model_name = model_name
            self.st_model = None  # Will be loaded asynchronously
            
            self.categories = self._load_categories()
            
            # Create processor instances
            self.query_processor = QueryProcessor(
                es_client=self.es_client,
                embedding_model=self.st_model
            )

            self.complaint_processor = ComplaintProcessor(
                es_client=self.es_client,
                embedding_model=self.st_model
            )

            self.suggestion_processor = SuggestionProcessor()
            self.email_classifier = EmailClassifier()

            # Try to download nltk data
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK punkt: {str(e)}")
            
            # Initialize shutdown flag
            self.shutdown_requested = False
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers configured for graceful shutdown")
            
            # Statistics
            self.processed_count = 0
            self.failed_count = 0
            self.concurrent_peak = 0
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("FAILED TO INITIALIZE MULTILINGUAL MESSAGE PROCESSOR")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            raise

    async def _initialize_models(self):
        """Initialize models asynchronously"""
        try:
            if self.model_path:
                logger.info(f"Loading model from local path: {self.model_path}")
                self.st_model = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, SentenceTransformer, self.model_path
                )
            else:
                logger.info(f"Loading model {self.model_name} from Hugging Face")
                self.st_model = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, SentenceTransformer, self.model_name
                )
            
            # Update processor instances with loaded model
            self.query_processor.embedding_model = self.st_model
            self.complaint_processor.embedding_model = self.st_model
            
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
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

    async def process_message_with_semaphore(self, msg_data: dict, msg_key: str, msg_headers: dict):
        """Process single message with concurrency control (max 3)"""
        async with self.processing_semaphore:
            try:
                # Set tracking ID for this task
                tracking_id = msg_headers.get('X-Tracking-ID', b'NA')
                if isinstance(tracking_id, bytes):
                    tracking_id = tracking_id.decode('utf-8')
                tracking_id_var.set(str(tracking_id))
                
                # Process the message
                await self.process_email_message(msg_data)
                self.processed_count += 1
                
            except Exception as e:
                self.failed_count += 1
                request_id = str(uuid.uuid4())
                logger.error(f"Error processing message | Request ID: {request_id} | Error: {str(e)}", exc_info=True)
                await self.send_to_dead_letter_queue(request_id, msg_data, str(e))

    async def publish_classification_to_kafka(self, tenant_id: str, topic: str, classification: EmailClassificationDto):
        """Publish classification to Kafka with async handling"""
        # Serialize key and value
        serialized_key = str(tenant_id).encode("utf-8")
        
        try:
            data = classification.model_dump()
        except AttributeError:
            data = classification.dict()
        
        serialized_classification = json.dumps(data).encode("utf-8")

        # Create future for async delivery
        future = asyncio.Future()
        
        def delivery_callback(err, msg):
            if err:
                future.set_exception(Exception(f"Message delivery failed: {err}"))
            else:
                future.set_result(msg)
        
        tracking_id = tracking_id_var.get() or "NA"
        tracking_id_str = str(tracking_id)

        self.producer.produce(
            topic, 
            key=serialized_key, 
            value=serialized_classification, 
            callback=delivery_callback,
            headers=[("X-Tracking-ID", tracking_id_str)]
        )
        self.producer.poll(0)
        
        return await future

    async def categorize_email_using_mistral(self, email_content: str, language: str, department: Optional[str] = None) -> tuple[str, str]:
        """Simple Mistral classification without rate limiting"""
        try:
            logger.debug(f"Making Mistral request for classification")
            
            classification = await self._classify_email_with_timeout(email_content, language, department)
            
            email_type = classification.get("type")
            email_subtype = classification.get("subtype", "")
            
            return email_type, email_subtype
            
        except asyncio.TimeoutError:
            logger.error("Mistral request timeout")
            return "error", ""
        except Exception as e:
            logger.error(f"Mistral request error: {str(e)}")
            return "error", ""
    
    async def _classify_email_with_timeout(self, email_content: str, detected_language: str, department: Optional[str] = None):
        """Classification with timeout for slow responses"""
        department_str = ""
        if department:
            department_data = next((item for item in self.categories if item["sector"] == department), None)
            if department_data:
                department_str = f"For the {department} department, these are the valid types and subtypes:\n{json.dumps(department_data['types'], indent=2)}\n\n"

        system_prompt = (
            "You are a multilingual email classification assistant. "
            "Classify the email into one of the following types: complaint, query, suggestion, spam. "
            "If a subtype is relevant based on the industry, include it. Otherwise, set subtype to an empty string."
        )

        user_prompt = f"""
Email content: {email_content}

Language detected: {detected_language}

{department_str}
Classify this email strictly into the format:

{{
"type": "selected_type",
"subtype": "selected_subtype"
}}

- `type` must be one of: "query", "complaint", "suggestion", "spam".
- `subtype` must be one of the predefined subtypes for the department, or "" if not applicable.
- Respond ONLY with a JSON object, nothing else.
- Do not add explanations or extra text.
"""

        data = {
            "model": self.MISTRAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "max_tokens": 250,
            "temperature": 0.2
        }
        
        # Use longer timeout for Mistral
        response = await self.http_client.post(
            self.MISTRAL_CONFIG['service_url'],
            headers={
                "Content-Type": "application/json",
                "Accept-Charset": "UTF-8"
            },
            json=data,
            timeout=self.MISTRAL_CONFIG['timeout']
        )

        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.status_code} - {response.text}")
            return {"type": "error", "subtype": ""}

        try:
            response_data = response.json()
            content = response_data['message']['content']

            # Ensure strict JSON parsing
            classification = json.loads(content)
            if classification.get("type") not in ["query", "complaint", "suggestion", "spam"]:
                classification["type"] = "spam"
            if "subtype" not in classification:
                classification["subtype"] = ""
            return classification
        except Exception as e:
            logger.error(f"Error parsing Mistral response: {e}")
            return {"type": "error", "subtype": ""}

    async def extract_sender_name_multilingual(self, email_data, language):
        """Extract sender name from email supporting multiple languages."""
        
        # Language-specific patterns
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
        
        specific_patterns = patterns.get(language, [])
        all_patterns = specific_patterns + patterns.get('en', []) + patterns['universal']
        
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
            
            lang_models = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'fr': 'fr_core_news_sm',
                'es': 'es_core_news_sm',
                'ja': 'ja_core_news_sm',
                'zh': 'zh_core_web_sm',
                'ru': 'ru_core_news_sm'
            }
            
            if language in lang_models:
                # Run spaCy processing in thread pool
                result = await asyncio.to_thread(self._process_with_spacy, email_data, lang_models[language])
                if result:
                    return result
        except ImportError:
            logger.debug("spaCy not available for NLP-based extraction")
        except Exception as e:
            logger.debug(f"Error in NLP extraction: {e}")
   
    def _process_with_spacy(self, email_data, model_name):
        """Process email data with spaCy in thread pool"""
        try:
            import spacy
            nlp = spacy.load(model_name)
            doc = nlp(email_data)
            sentences = list(doc.sents)
            potential_signature = " ".join([str(sent) for sent in sentences[-3:]])
            signature_doc = nlp(potential_signature)
            person_entities = [ent.text for ent in signature_doc.ents if ent.label_ == "PERSON"]
            return person_entities[0] if person_entities else None
        except Exception as e:
            logger.debug(f"spaCy processing error: {e}")
            return None

    async def process_email_message(self, message):
        """Process individual message and store in Elasticsearch with vectors"""
        logger.info(f"Processing message: {message.get('tenantId')}/{message.get('threadId')}")

        tenant_id = message.get('tenantId')
        thread_id = message.get('threadId')
        message_id = message.get('messageId')
        department: Optional[str] = message.get('department')
        sender_name = message.get('senderName')
        
        # Extract text content from message
        content = message.get('emailBody') or ""
        subject = message.get('subject') or ""
        content = content.replace('\n', ' ').replace('\r', ' ')
        subject = subject.replace('\n', ' ').replace('\r', ' ')
        
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

        # Build complete_content
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

        # Step 1: Basic classification - must succeed or goes to DLQ
        try:
            # Run language detection in thread pool
            language_code = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, detect_language, content
            )
            
            if not sender_name:
                sender_name = await self.extract_sender_name_multilingual(content, language_code)
                    
            language = LangUtil._get_language_by_code(language_code)

            # Perform classification
            if language_code == 'en' and department:
                type, subtype = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, 
                    lambda: asyncio.run(self.email_classifier.process_emails(content, department))
                )
            else:
                type, subtype = await self.categorize_email_using_mistral(complete_content, language, department)

            logger.info(f"Email classified as: {type}, SubType: {subtype}")
            
        except Exception as e:
            logger.error(f"Classification failed for message - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
            raise
        
        # Create basic classification object
        basic_classification = EmailClassificationDto(
            tenantId=tenant_id, 
            threadId=thread_id, 
            messageId=message_id, 
            senderName=sender_name,
            type='query', 
            subType=''
        )
        
        # Step 3: Enhanced processing based on type
        enhanced_classification = None
        processing_error = None
        
        try:
            if type == "query":
                enhanced_classification = await self._process_query_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, language, language_code, query_ai_mode, query_reply_template, query_regards
                )
            elif type == "complaint":
                enhanced_classification = await self._process_complaint_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, subject, content, language, language_code, complaint_ai_mode, 
                    complaint_reply_template, auto_complaint_ticket_generation, complaint_regards
                )
            elif type == "suggestion":
                enhanced_classification = await self._process_suggestion_type(
                    tenant_id, thread_id, message_id, sender_name, type, subtype,
                    complete_content, language, suggestion_ai_mode, 
                    suggestion_reply_template, suggestion_regards
                )
            else:
                logging.info(f"Unknown email classification type : {type}")
                # For spam or other types, use basic classification
                enhanced_classification = basic_classification
                
        except Exception as e:
            processing_error = e
            logger.error(f"Enhanced processing failed for {type} - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
        
        # Step 4: Use enhanced classification if successful, otherwise use basic
        final_classification = enhanced_classification if enhanced_classification else basic_classification
        
        # Step 5: Publish the classification
        try:
            await self.publish_classification_to_kafka(tenant_id, KAFKA_CONFIG['classification_response_topic'], final_classification)
            
            if processing_error:
                logger.info(f"Classification published with basic data due to processing error - tenantId: {tenant_id}, threadId: {thread_id}")
            else:
                logger.info(f"Classification published successfully - tenantId: {tenant_id}, threadId: {thread_id}")
                
        except Exception as e:
            logger.error(f"Failed to publish classification - tenantId: {tenant_id}, threadId: {thread_id}, error: {str(e)}", exc_info=True)
            raise

    # Keep all the existing _process_*_type methods unchanged
    async def _process_query_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                 complete_content, language, language_code, query_ai_mode, query_reply_template, query_regards):
        """Process query type emails"""
        if query_ai_mode in ["no_reply", "template_only"]:
            return EmailClassificationDto(
                tenantId=tenant_id, 
                threadId=thread_id, 
                messageId=message_id, 
                senderName=sender_name,
                type=type, 
                subType=subtype
            )
        
        query_response = await self.query_processor.extract_query_generate_responses(
            tenant_id, thread_id, sender_name, complete_content, language, language_code,
            type, subtype, query_ai_mode, query_reply_template, query_regards
        )
        
        response_content = None
        if query_response:
            response_content = query_response.get("message", {}).get("content")
            if not response_content:
                response_content = "No answer found"
        
        if response_content and response_content != "No answer found":
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id, 
                senderName=sender_name, type=type, subType=subtype, queryResponse=response_content
            )
        else:
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id, 
                senderName=sender_name, type=type, subType=subtype
            )

    async def _process_complaint_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                    complete_content, subject, content, language, language_code, complaint_ai_mode, 
                                    complaint_reply_template, auto_complaint_ticket_generation, complaint_regards):
        """Process complaint type emails"""
        if complaint_ai_mode in ["no_reply"] or complaint_ai_mode in ["template_only"] and not auto_complaint_ticket_generation:
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id, 
                senderName=sender_name, type=type, subType=subtype
            )
        
        complaint_list = await self.complaint_processor.extract_complaints(tenant_id, complete_content, language, language_code)
        
        complaint_response_content = None
        
        if complaint_list:
            complaints = complaint_list.get('complaints', [])
            advices_data = complaint_list.get('advices', [])

            advices_list = [
                Advice(query=advice.get("query", ""), advice=advice.get("advice", ""), url=advice.get("url", "#")) 
                for advice in advices_data
            ]
            
            complaint_object = Complaint(complaints=complaints, advises=advices_list)
            
            if complaint_ai_mode not in ["no_reply", "template_only"]:
                documents_for_response = [
                    {"content": advice.get("advice", ""), "url": advice.get("url", "#")}
                    for advice in advices_data
                ]
                
                complaint_response = await self.complaint_processor.generate_complaint_response(
                    sender_name=sender_name, email_content=complete_content, documents=documents_for_response,
                    language=language, template=complaint_reply_template, complaint_regards=complaint_regards
                )

                if complaint_response:
                    complaint_response_content = complaint_response
                    if not complaint_response_content or complaint_response_content.strip() == "":
                        complaint_response_content = "No response generated"
                        
            ticket_data = None
            if auto_complaint_ticket_generation:
                try:
                    ticket_data = await self.complaint_processor.generate_ticket_data(
                        sender_name=sender_name, complaints=complaints, suggestions=advices_data, language=language
                    )
                except Exception as e:
                    logger.error(f"Error generating ticket data: {e}")
                    ticket_data = TicketData(title=f"{subject}", description=f"{content}", priority="Medium")
            
            classification_data = {
                "tenantId": tenant_id, "threadId": thread_id, "messageId": message_id,
                "senderName": sender_name, "type": type, "subType": subtype, "complaint": complaint_object
            }
            
            if complaint_response_content and complaint_response_content != "No response generated":
                classification_data["complaintResponse"] = complaint_response_content
            
            if ticket_data:
                classification_data["ticketData"] = ticket_data

            return EmailClassificationDto(**classification_data)
            
        else:
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id,
                senderName=sender_name, type=type, subType=subtype
            )

    async def _process_suggestion_type(self, tenant_id, thread_id, message_id, sender_name, type, subtype,
                                     complete_content, language, suggestion_ai_mode, 
                                     suggestion_reply_template, suggestion_regards):
        """Process suggestion type emails"""
        if suggestion_ai_mode in ["no_reply", "template_only"]:
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id, 
                senderName=sender_name, type=type, subType=subtype
            )
        
        suggestions_list = await self.suggestion_processor.extract_suggestions(
            email_content=complete_content, type=type, subType=subtype, language=language
        )
        
        suggestion_response_content = None
        
        if suggestions_list:
            if suggestion_ai_mode not in ["no_reply", "template_only"]:
                suggestion_response = await self.suggestion_processor.generate_suggestion_response(
                    sender_name=sender_name, email_content=complete_content, suggestions=suggestions_list,
                    language=language, template=suggestion_reply_template, suggestion_regards=suggestion_regards
                )
                
                if suggestion_response:
                    suggestion_response_content = suggestion_response
                    if not suggestion_response_content or suggestion_response_content.strip() == "None":
                        suggestion_response_content = None
            
            classification_data = {
                "tenantId": tenant_id, "threadId": thread_id, "messageId": message_id,
                "senderName": sender_name, "type": type, "subType": subtype, "suggestions": suggestions_list
            }
            
            if suggestion_response_content and suggestion_response_content.strip():
                classification_data["suggestionResponse"] = suggestion_response_content
            
            return EmailClassificationDto(**classification_data)
            
        else:
            return EmailClassificationDto(
                tenantId=tenant_id, threadId=thread_id, messageId=message_id,
                senderName=sender_name, type=type, subType=subtype
            )

    async def send_to_dead_letter_queue(self, request_id, message, error):
        """Send problematic messages to a dead letter topic"""
        error_message = {
            "original_message": message,
            "error": error,
            "request_id": request_id
        }
        
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Failed to send to dead letter queue: {err}")
            else:
                logger.info(f"Message sent to classification_request_dlq_topic")
        
        self.producer.produce(
            KAFKA_CONFIG['classification_request_dlq_topic'], 
            json.dumps(error_message).encode('utf-8'), 
            callback=delivery_callback
        )
        self.producer.poll(0)

    async def consume_messages_parallel(self):
        """Consume messages from Kafka with parallel processing"""
        consumer = Consumer(self.consumer_config)
        
        try:
            consumer.subscribe([self.topic])
            logger.info(f"Successfully subscribed to Kafka topic: {self.topic}")
            logger.info("Starting parallel message consumption loop...")
            
            message_count = 0
            last_heartbeat = datetime.now()
            
            while not self.shutdown_requested:
                msg = consumer.poll(self.system_config.kafka_poll_timeout)
                
                # Send periodic heartbeat logs with stats
                now = datetime.now()
                if (now - last_heartbeat).seconds >= 30:
                    active_count = len(self.active_tasks)
                    self.concurrent_peak = max(self.concurrent_peak, active_count)
                    logger.info(f"Server heartbeat - Status: RUNNING | "
                              f"Processed: {self.processed_count} | Failed: {self.failed_count} | "
                              f"Active: {active_count} | Peak Concurrent: {self.concurrent_peak}")
                    last_heartbeat = now
                
                if msg is None:
                    # Clean up completed tasks
                    completed_tasks = [task for task in self.active_tasks if task.done()]
                    for task in completed_tasks:
                        self.active_tasks.remove(task)
                        try:
                            await task  # Get any exceptions
                            consumer.commit()  # Commit offset for successful tasks
                        except Exception as e:
                            logger.error(f"Task completed with error: {e}")
                    await asyncio.sleep(0.01)  # Prevent busy waiting
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"Kafka consumer error: {msg.error()}")
                    continue
                
                # Process message in parallel
                try:
                    value = msg.value()
                    if isinstance(value, bytes):
                        value = json.loads(value.decode('utf-8'))
                    elif isinstance(value, str):
                        value = json.loads(value)
                    
                    message_count += 1
                    tenant_id = value.get('tenantId', 'unknown')
                    thread_id = value.get('threadId', 'unknown')
                    
                    # Get headers
                    kafka_headers = dict(msg.headers() or [])
                    
                    logger.info(f"Processing message #{message_count} | Tenant: {tenant_id} | Thread: {thread_id}")
                    
                    # Create task for parallel processing
                    task = asyncio.create_task(
                        self.process_message_with_semaphore(
                            value, 
                            msg.key().decode('utf-8') if msg.key() else str(tenant_id),
                            kafka_headers
                        )
                    )
                    self.active_tasks.add(task)
                    
                    # Clean up completed tasks periodically
                    if len(self.active_tasks) > self.system_config.max_concurrent_messages * 2:
                        completed_tasks = [task for task in self.active_tasks if task.done()]
                        for task in completed_tasks:
                            self.active_tasks.remove(task)
                            try:
                                await task
                                consumer.commit()  # Commit offset for successful tasks
                            except Exception as e:
                                logger.error(f"Task failed: {e}")
                    
                except Exception as e:
                    logger.error(f"Error creating processing task for message #{message_count}: {str(e)}", exc_info=True)
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received during message consumption")
        finally:
            # Wait for all active tasks to complete
            if self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                completed, pending = await asyncio.wait(
                    self.active_tasks, 
                    timeout=30.0,  # Give 30 seconds for graceful completion
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel any remaining tasks
                for task in pending:
                    task.cancel()
                    
                logger.info(f"Completed {len(completed)} tasks, cancelled {len(pending)} tasks")
            
            consumer.close()
            logger.info("Kafka consumer closed")

    async def run(self):
        """Main processing loop with parallel execution"""
        logger.info("=" * 60)
        logger.info("STARTING MULTILINGUAL MESSAGE PROCESSOR SERVER")
        logger.info("=" * 60)
        
        try:
            # Perform health check
            if not await self.health_check():
                logger.error("Health check failed. Cannot start server.")
                return
        
            logger.info("Initializing models...")
            await self._initialize_models()
            logger.info("Models initialized successfully")
            
            # Initialize producer
            logger.info("Initializing Kafka producer...")
            self.producer = Producer(self.producer_config)
            logger.info("Kafka producer initialized successfully")
            
            logger.info("=" * 60)
            logger.info("SERVER STARTED SUCCESSFULLY!")
            logger.info(f"Listening for messages on topic: {self.topic}")
            logger.info(f"Consumer group: {self.consumer_config['group.id']}")
            logger.info(f"Max concurrent messages: {self.system_config.max_concurrent_messages}")
            logger.info(f"Thread pool size: {self.system_config.thread_pool_size}")
            logger.info(f"GPU enabled: {self.system_config.has_gpu}")
            logger.info(f"Server ready at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            
            # Start consuming messages with parallel processing
            await self.consume_messages_parallel()
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("FATAL ERROR IN MAIN LOOP")
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
            logger.info(f"Elasticsearch connection healthy - Version: {es_info['version']['number']}")
            
            # Check Kafka connection
            logger.info("Checking Kafka connection...")
            test_consumer = Consumer(self.consumer_config)
            topics = test_consumer.list_topics(timeout=5)
            test_consumer.close()
            logger.info(f"Kafka connection healthy - Available topics: {len(topics.topics)}")
            
            logger.info("All health checks passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def shutdown(self):
        """Graceful shutdown with cleanup"""
        logger.info("=" * 60)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("=" * 60)
        
        try:
            # Signal shutdown to stop accepting new messages
            self.shutdown_requested = True
            
            # Wait for active tasks to complete
            if self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                completed, pending = await asyncio.wait(
                    self.active_tasks, 
                    timeout=60.0,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel remaining tasks if any
                for task in pending:
                    task.cancel()
                
                logger.info(f"Completed {len(completed)} tasks, cancelled {len(pending)} tasks")
            
            # Close HTTP client
            logger.info("Closing HTTP client...")
            await self.http_client.aclose()
            logger.info("HTTP client closed successfully")
            
            # Close Elasticsearch client
            logger.info("Closing Elasticsearch client...")
            await self.es_client.close()
            logger.info("Elasticsearch client closed successfully")
            
            # Flush Kafka producer
            if hasattr(self, 'producer'):
                logger.info("Flushing Kafka producer...")
                self.producer.flush(timeout=10)
                logger.info("Kafka producer flushed successfully")
            
            # Shutdown thread pool
            logger.info("Shutting down thread pool...")
            self.thread_pool.shutdown(wait=True, timeout=30)
            logger.info("Thread pool shutdown completed")
            
            # Print final statistics
            logger.info("=" * 40)
            logger.info("FINAL PROCESSING STATISTICS")
            logger.info(f"Total messages processed: {self.processed_count}")
            logger.info(f"Total failures: {self.failed_count}")
            logger.info(f"Peak concurrent processing: {self.concurrent_peak}")
            logger.info(f"Success rate: {(self.processed_count / (self.processed_count + self.failed_count) * 100):.2f}%" if (self.processed_count + self.failed_count) > 0 else "N/A")
            
            logger.info("=" * 60)
            logger.info("GRACEFUL SHUTDOWN COMPLETED")
            logger.info(f"Shutdown completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("MULTILINGUAL MESSAGE PROCESSOR - STARTING UP")
    logger.info("=" * 80)
    
    try:
        # Use uvloop for better performance if available
        try:
            import uvloop
            uvloop.install()
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")
        
        # Initialize processor
        processor = MultilingualMessageProcessor()
        
        # Run the processor in an asyncio event loop
        asyncio.run(processor.run())
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
    finally:
        logger.info("Application shutdown complete")
        logger.info("=" * 80)