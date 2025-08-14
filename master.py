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
from complaint import ComplaintExtractor
from suggestion import SuggestionExtractor
from response import EmailClassificationDto, Advice,Complaint
from contextvars import ContextVar
import tempfile

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)

tracking_id_var = ContextVar("tracking_id", default="NA")


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
                ca_certs=self.get_clean_cert_path(ES_CONFIG.get('ca_certs')),  # Add this line
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

            self.complaint_extractor = ComplaintExtractor(
                es_client=self.es_client,
                embedding_model=self.st_model
            )

            self.suggestion_extractor = SuggestionExtractor(
                embedding_model=self.st_model
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


    def get_clean_cert_path(self, cert_path: str) -> str:
        pem_lines = []
        inside_cert = False
        with open(cert_path, "r") as f:
            for line in f:
                if "-----BEGIN CERTIFICATE-----" in line:
                    inside_cert = True
                if inside_cert:
                    pem_lines.append(line)
                if "-----END CERTIFICATE-----" in line:
                    break

        # Write to a temporary file automatically
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".pem")
        tmp_file.write("".join(pem_lines))
        tmp_file.close()
        return tmp_file.name

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
        
        # Get current trackingId
        tracking_id = tracking_id_var.get() or "NA"


        self.producer.produce(topic, key=serialized_key, value=serialized_classification, callback=delivery_callback,headers=[("trackingId", tracking_id.encode("utf-8"))])
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
        logger.info(f"Mesage complete: {message}")

        tenant_id = message.get('tenantId')
        thread_id = message.get('threadId')
        message_id = message.get('messageId')
        department: Optional[str] = message.get('department')  # Declare type here
        sender_name = message.get('senderName')
        # Extract text content from message
        # Get content with fallback
        content = message.get('emailBody') or ""
        subject = message.get('subject') or ""
        queryReplyGeneration = message.get('queryReplyGeneration') or False
        autoComplaintTicketGeneration = message.get('autoComplaintTicketGeneration') or False



        # Build complete_content intelligently
        if subject and content:
            complete_content = f"Subject: {subject}, Body: {content}"
        elif subject :
            complete_content = f"Subject: {subject}"
        elif content:
            complete_content = content
        else:
            complete_content =''

        if not complete_content:
            logger.warning("Email Message has no content , tenantId : {tenant_id}, threadId: {thread_id} ")
            return
        

        language = detect_language(content)
        

        if not sender_name :
            sender_name = self.extract_sender_name_multilingual(content,language)


        complete_content = "Subject: "+subject +", Body: "+content


        logging.info(f"language: {language} department: {department} , complete_content {complete_content}")

        if language == 'en' and department :
            type,subtype = await self.email_classifier.process_emails( content, department);
        else :
            type,subtype = await self.categorize_email_using_mistral(complete_content, language, department);


        logger.info(f"Email Type: {type} , SubType: {subtype}")

  
        if type == "query":

            # If query reply generation is not requed
            if not queryReplyGeneration :
                classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type, subType=subtype)
 
            else :
          
                # Extract all questions from the email
                query_response = await self.query_processor.generate_query_responses(tenant_id,thread_id,sender_name,complete_content, language,type,subtype)
                if query_response:
                    response_content = query_response.get("message", {}).get("content", "No answer found")

                    if content:
                        classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type, subType=subtype, queryResponse=response_content)

                    else:
                        classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type, subType=subtype)

                else:
                    classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type, subType=subtype)       
        
        elif type == "complaint":
            
            if not autoComplaintTicketGeneration:
                classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type, subType=subtype)
 
            else :

                complaint_list = await self.complaint_extractor.extract_complaints(tenant_id,complete_content, language)
                logger.info(f" complaint_list {complaint_list}")
                
                if complaint_list:
                    complaints = complaint_list['complaints']  

                    # Create advice objects from the existing advices
                    advices_list = [
                        Advice(query = advice["query"], advice=advice["advice"], url=advice["url"]) 
                        for advice in complaint_list["advices"]
                    ]
                
                    
                    # Create the complaint object directly using the question as is
                    complaint_object = Complaint(
                        complaints=complaints,  
                        advises=advices_list
                    )
                    
                    # Use this in your EmailClassificationDto
                    classification = EmailClassificationDto(
                        tenantId=tenant_id,
                        threadId=thread_id,
                        messageId=message_id,
                        type=type,
                        subType=subtype,
                        complaint=complaint_object
                    )

        elif type == "suggestion":
            suggestion_list = await self.suggestion_extractor.extract_suggestions(complete_content, type, subtype,department, language )
            classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type=type,subType=subtype,suggestions=suggestion_list)

        else:
            classification = EmailClassificationDto(tenantId=tenant_id, threadId=thread_id, messageId=message_id, type="spam",subType="")


      
        await self.publish_classification_to_kafka(tenant_id,KAFKA_CONFIG['classification_response_topic'],classification)
        
        logger.info(f"Classification published {classification}")




    
    

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
                    tracking_id = kafka_headers.get('trackingId', 'NA')
                    tracking_id_var.set(tracking_id)

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