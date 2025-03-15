import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)


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
        
        # Initialize models
        self._load_spam_model()
        self._load_type_subtype_model()
    
    def _load_spam_model(self):
        """Load the spam classification model"""
        spam_model_path = './spam_model'
        try:
            self.spam_tokenizer = DebertaV2Tokenizer.from_pretrained(spam_model_path)
            self.spam_model = DebertaV2ForSequenceClassification.from_pretrained(spam_model_path)
            logger.info("Spam model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading spam model: {e}")
            sys.exit(1)
    
    def _load_type_subtype_model(self):
        """Load the type/subtype classification model and encoders"""
        type_subtype_model_path = './final_model'
        try:
            encoder_dir = os.path.join(type_subtype_model_path, 'encoders')
            logger.info("0")

            # Make DefaultDict available to pickle by adding it to the appropriate module
            import sys
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
            # This might need to be adjusted based on the original module name
            sys.modules['__main__'].DefaultDict = _DefaultDictModule.DefaultDict



            with open(os.path.join(encoder_dir, 'model_config.pkl'), 'rb') as file:
                model_config = pickle.load(file)

            config = DebertaV2Config.from_pretrained(type_subtype_model_path)
            config.max_types = model_config['max_types']
            config.max_subtypes = model_config['max_subtypes']
   
            with open(os.path.join(encoder_dir, 'department_type_encoders.pkl'), 'rb') as file:
                self.department_type_encoders = pickle.load(file)

            with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'rb') as file:
                self.department_subtype_encoders = pickle.load(file)

            with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'rb') as file:
                self.department_subtype_hierarchies = pickle.load(file)
  
            self.type_subtype_model = self.DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
                type_subtype_model_path,
                config=config,
                department_type_encoders=self.department_type_encoders,
                department_subtype_encoders=self.department_subtype_encoders
            )
            self.type_subtype_tokenizer = DebertaV2Tokenizer.from_pretrained(type_subtype_model_path)
            logger.info("Type/Subtype model and encoders loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading type/subtype model or encoders: {e}")
            sys.exit(1)
    
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
    
    def classify_spam(self, email: str) -> str:
        """Classify an email as spam or not"""
        try:
            inputs = self.spam_tokenizer(email, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.spam_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            logging.info(f" prediction {prediction}")
            return self.spam_label_map.get(prediction, "unknown")
        except Exception as e:
            logger.error(f"Error in spam classification: {e}")
            return "unknown"
    
    def classify_type_subtype(self, email: str, department: str, subtype_threshold=0.3) -> (str, str):
        """Classify the type and subtype of an email"""
        try:
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
        except Exception as e:
            logger.error(f"Error in type/subtype classification: {e}")
            return "unknown", None
    
    async def process_emails(self, complete_content, department):
        """Process an email to determine its type and subtype"""
        try:
            logger.info(f"Processing email: {complete_content}")
            label = self.classify_spam(complete_content)
            if label in ["spam"]:
                type_result = label
                subtype_result = ""
            else:
                type_result, subtype_result = self.classify_type_subtype(complete_content, department)
            
            return type_result, subtype_result
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            return "unknown", None
