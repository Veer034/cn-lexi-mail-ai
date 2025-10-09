import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
import os
import logging
import sys
import asyncio  # ADD THIS IMPORT

# Configure logging
from logger_config import get_logger
logger = get_logger(__name__)


class EmailClassifier:
    def __init__(self):
        self.spam_tokenizer = None
        self.spam_model = None
        self.spam_label_map = {0: "spam", 1: "query", 2: "complaint", 3: "suggestion"}
        self.type_subtype_tokenizer = None
        self.type_subtype_model = None
        self.department_type_encoders = None
        self.department_subtype_encoders = None
        self.department_subtype_hierarchies = None
        
        # DON'T initialize models here - they're async!
        # Models will be loaded via async initialize() method
    
    async def initialize(self):
        """Initialize models asynchronously - call this after creating the instance"""
        await self._load_spam_model()
        await self._load_type_subtype_model()
    
    async def _load_spam_model(self):
        """Load the spam classification model asynchronously"""
        spam_model_path = './spam_model'
        try:
            # Load tokenizer in thread pool
            self.spam_tokenizer = await asyncio.to_thread(
                DebertaV2Tokenizer.from_pretrained, spam_model_path
            )
            # Load model in thread pool
            self.spam_model = await asyncio.to_thread(
                DebertaV2ForSequenceClassification.from_pretrained, spam_model_path
            )
            logger.info("Spam model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading spam model: {e}")
            sys.exit(1)
    
    async def _load_type_subtype_model(self):
        """Load the type/subtype classification model and encoders asynchronously"""
        type_subtype_model_path = './final_model'
        try:
            encoder_dir = os.path.join(type_subtype_model_path, 'encoders')
            logger.info("Loading type/subtype model asynchronously...")

            # Make DefaultDict available to pickle by adding it to the appropriate module
            import pickle
            
            # Get the original module that DefaultDict was defined in
            class _DefaultDictModule:
                class DefaultDict(dict):
                    def __missing__(self, key):
                        self[key] = DefaultDict()
                        return self[key]

                    def add(self, item):
                        self[item] = set()
            
            # Add the class to the module pickle is looking for
            sys.modules['__main__'].DefaultDict = _DefaultDictModule.DefaultDict

            # Load all pickle files asynchronously using thread pool
            logger.info("Loading model configuration...")
            model_config = await asyncio.to_thread(
                self._load_pickle_file, 
                os.path.join(encoder_dir, 'model_config.pkl')
            )

            logger.info("Loading department type encoders...")
            self.department_type_encoders = await asyncio.to_thread(
                self._load_pickle_file,
                os.path.join(encoder_dir, 'department_type_encoders.pkl')
            )

            logger.info("Loading department subtype encoders...")
            self.department_subtype_encoders = await asyncio.to_thread(
                self._load_pickle_file,
                os.path.join(encoder_dir, 'department_subtype_encoders.pkl')
            )

            logger.info("Loading department subtype hierarchies...")
            self.department_subtype_hierarchies = await asyncio.to_thread(
                self._load_pickle_file,
                os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl')
            )

            # Load configuration asynchronously
            logger.info("Loading DeBERTa configuration...")
            config = await asyncio.to_thread(
                DebertaV2Config.from_pretrained, 
                type_subtype_model_path
            )
            
            # Set configuration parameters
            config.max_types = model_config['max_types']
            config.max_subtypes = model_config['max_subtypes']

            # Load the custom model asynchronously
            logger.info("Loading custom DeBERTa model...")
            self.type_subtype_model = await asyncio.to_thread(
                self._load_custom_model,
                type_subtype_model_path,
                config,
                self.department_type_encoders,
                self.department_subtype_encoders
            )

            # Load tokenizer asynchronously
            logger.info("Loading tokenizer...")
            self.type_subtype_tokenizer = await asyncio.to_thread(
                DebertaV2Tokenizer.from_pretrained,
                type_subtype_model_path
            )

            logger.info("Type/Subtype model and encoders loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading type/subtype model or encoders: {e}")
            sys.exit(1)

    def _load_pickle_file(self, file_path):
        """Load pickle file synchronously (for use in thread pool)"""
        import pickle
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def _load_custom_model(self, model_path, config, dept_type_encoders, dept_subtype_encoders):
        """Load custom DeBERTa model synchronously (for use in thread pool)"""
        return self.DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
            model_path,
            config=config,
            department_type_encoders=dept_type_encoders,
            department_subtype_encoders=dept_subtype_encoders
        )
    
    # Define the model class as an inner class
    class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2ForSequenceClassification):
        def __init__(self, config, department_type_encoders, department_subtype_encoders):
            super().__init__(config)
            self.department_type_encoders = department_type_encoders
            self.department_subtype_encoders = department_subtype_encoders
            self.type_classifier = torch.nn.Linear(config.hidden_size, config.max_types)
            self.subtype_classifier = torch.nn.Linear(config.hidden_size, config.max_subtypes)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            outputs = self.deberta(input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]
            pooled_output = self.pooler(sequence_output)
            type_logits = self.type_classifier(pooled_output)
            subtype_logits = self.subtype_classifier(pooled_output)
            return {'type_logits': type_logits, 'subtype_logits': subtype_logits}
    
    async def classify_spam(self, email: str) -> str:
        """Classify an email as spam or not"""
        try:
            result = await asyncio.to_thread(self._classify_spam_sync, email)
            return result
        except Exception as e:
            logger.error(f"Error in spam classification: {e}")
            return "unknown"

    def _classify_spam_sync(self, email: str) -> str:
        """Synchronous spam classification for thread pool"""
        inputs = self.spam_tokenizer(email, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.spam_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return self.spam_label_map.get(prediction, "unknown")
    
    async def classify_type_subtype(self, email: str, department: str, subtype_threshold=0.3) -> tuple:
        """Classify the type and subtype of an email asynchronously"""
        try:
            result = await asyncio.to_thread(
                self._classify_type_subtype_sync, 
                email, 
                department, 
                subtype_threshold
            )
            return result
        except Exception as e:
            logger.error(f"Error in type/subtype classification: {e}")
            return "unknown", None
    
    def _classify_type_subtype_sync(self, email: str, department: str, subtype_threshold=0.3) -> tuple:
        """Synchronous type/subtype classification for thread pool"""
        input_text = f"{department} [SEP] {email}"
        inputs = self.type_subtype_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.type_subtype_model(**inputs)
        type_logits = outputs['type_logits']
        subtype_logits = outputs['subtype_logits']

        dept_type_encoder = self.department_type_encoders[department]
        dept_subtype_encoder = self.department_subtype_encoders[department]

        valid_type_count = len(dept_type_encoder.classes_)
        valid_subtype_count = len(dept_subtype_encoder.classes_)

        type_probs = torch.nn.functional.softmax(type_logits[0, :valid_type_count], dim=0)
        subtype_probs = torch.nn.functional.softmax(subtype_logits[0, :valid_subtype_count], dim=0)

        predicted_type = dept_type_encoder.inverse_transform([torch.argmax(type_probs).item()])[0]
        max_subtype_prob = torch.max(subtype_probs).item()
        predicted_subtype = None

        if max_subtype_prob >= subtype_threshold:
            predicted_subtype = dept_subtype_encoder.inverse_transform([torch.argmax(subtype_probs).item()])[0]
            valid_subtypes = self.department_subtype_hierarchies[department][predicted_type]
            if predicted_subtype not in valid_subtypes:
                predicted_subtype = None

        return predicted_type, predicted_subtype
    
    async def process_emails(self, complete_content, department):
        """Process an email to determine its type and subtype"""
        try:
            logger.debug(f"Starting email classification for content length: {len(complete_content)}")
            
            # Check if models are loaded
            if self.spam_model is None or self.type_subtype_model is None:
                logger.error("Models not initialized! Call initialize() first.")
                return "unknown", None
            
            logger.debug("Classifying spam...")
            label = await self.classify_spam(complete_content)
            logger.info(f"Spam classification result: {label}")
            
            if label in ["spam"]:
                type_result = label
                subtype_result = ""
            else:
                logger.debug(f"Classifying type/subtype for department: {department}")
                type_result, subtype_result = await self.classify_type_subtype(complete_content, department)
                logger.info(f"Type/Subtype classification result: {type_result}, {subtype_result}")
            
            return type_result, subtype_result
        except Exception as e:
            logger.error(f"Error processing email: {e}", exc_info=True)
            return "unknown", None