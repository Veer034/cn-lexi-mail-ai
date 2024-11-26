import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
import pickle
import os
from collections import defaultdict
import json
import faust
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka and Faust settings
config_path = 'config/config.json'

try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    logger.error(f"Config file not found: {config_path}")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Error parsing config file: {config_path}")
    sys.exit(1)

KAFKA_BROKER = f"kafka://{config['kafka']['bootstrap_servers'][0]}"
KAFKA_INPUT_TOPIC = config['kafka']['input_topic']
KAFKA_OUTPUT_TOPIC = config['kafka']['output_topic']

# Faust app setup
app = faust.App('email_classification_app', broker=KAFKA_BROKER)

class EmailTypeAssignmentRequestRecord(faust.Record):
    tenantId: str
    department: str
    threadId: str
    emailBody: str

class EmailTypeAssignedRecord(faust.Record):
    tenantId: str
    threadId: str
    type: str
    subType: str

kafka_input_topic = app.topic(KAFKA_INPUT_TOPIC, value_type=EmailTypeAssignmentRequestRecord)
kafka_output_topic = app.topic(KAFKA_OUTPUT_TOPIC, value_type=EmailTypeAssignedRecord)

# Define DefaultDict class (needed for encoders)
class DefaultDict(dict):
    def __missing__(self, key):
        self[key] = DefaultDict()
        return self[key]

    def add(self, item):
        self[item] = set()

# Load Spam Classification Model
spam_model_path = './spam_model'
try:
    spam_tokenizer = DebertaV2Tokenizer.from_pretrained(spam_model_path)
    spam_model = DebertaV2ForSequenceClassification.from_pretrained(spam_model_path)
    spam_label_map = {0: "spam", 1: "query", 2: "complaint", 3: "suggestion"}
    logger.info("Spam model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading spam model: {e}")
    sys.exit(1)

# Load Type/Subtype Classification Model and Encoders
type_subtype_model_path = './final_model'
try:
    encoder_dir = os.path.join(type_subtype_model_path, 'encoders')

    with open(os.path.join(encoder_dir, 'model_config.pkl'), 'rb') as file:
        model_config = pickle.load(file)

    config = DebertaV2Config.from_pretrained(type_subtype_model_path)
    config.max_types = model_config['max_types']
    config.max_subtypes = model_config['max_subtypes']

    with open(os.path.join(encoder_dir, 'department_type_encoders.pkl'), 'rb') as file:
        department_type_encoders = pickle.load(file)

    with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'rb') as file:
        department_subtype_encoders = pickle.load(file)

    with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'rb') as file:
        department_subtype_hierarchies = pickle.load(file)

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

    type_subtype_model = DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
        type_subtype_model_path,
        config=config,
        department_type_encoders=department_type_encoders,
        department_subtype_encoders=department_subtype_encoders
    )
    type_subtype_tokenizer = DebertaV2Tokenizer.from_pretrained(type_subtype_model_path)
    logger.info("Type/Subtype model and encoders loaded successfully.")
except Exception as e:
    logger.error(f"Error loading type/subtype model or encoders: {e}")
    sys.exit(1)

# Spam Classification Function
def classify_spam(email: str) -> str:
    try:
        inputs = spam_tokenizer(email, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = spam_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return spam_label_map.get(prediction, "unknown")
    except Exception as e:
        logger.error(f"Error in spam classification: {e}")
        return "unknown"

# Type/Subtype Classification Function
def classify_type_subtype(email: str, department: str, subtype_threshold=0.3) -> (str, str):
    try:
        input_text = f"{department} [SEP] {email}"
        inputs = type_subtype_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = type_subtype_model(**inputs)
        type_logits = outputs['type_logits']
        subtype_logits = outputs['subtype_logits']

        dept_type_encoder = department_type_encoders[department]
        dept_subtype_encoder = department_subtype_encoders[department]

        valid_type_count = len(dept_type_encoder.classes_)
        valid_subtype_count = len(dept_subtype_encoder.classes_)

        type_probs = torch.nn.functional.softmax(type_logits[0, :valid_type_count], dim=0)
        subtype_probs = torch.nn.functional.softmax(subtype_logits[0, :valid_subtype_count], dim=0)

        predicted_type = dept_type_encoder.inverse_transform([torch.argmax(type_probs).item()])[0]
        max_subtype_prob = torch.max(subtype_probs).item()
        predicted_subtype = None

        if max_subtype_prob >= subtype_threshold:
            predicted_subtype = dept_subtype_encoder.inverse_transform([torch.argmax(subtype_probs).item()])[0]
            valid_subtypes = department_subtype_hierarchies[department][predicted_type]
            if predicted_subtype not in valid_subtypes:
                predicted_subtype = None

        return predicted_type, predicted_subtype
    except Exception as e:
        logger.error(f"Error in type/subtype classification: {e}")
        return "unknown", None

# Faust Agent to Process Emails
@app.agent(kafka_input_topic)
async def process_emails(stream):
    async for email in stream:
        try:
            logger.info(f"Processing email: {email}")
            label = classify_spam(email.emailBody)
            if label in ["spam"]:
                output = EmailTypeAssignedRecord(tenantId=email.tenantId, threadId=email.threadId, type=label, subType="")
            else:
                type_, subtype = classify_type_subtype(email.emailBody, email.department)
                output = EmailTypeAssignedRecord(tenantId=email.tenantId, threadId=email.threadId, type=type_, subType=subtype or "")
            await kafka_output_topic.send(value=output.asdict())
        except Exception as e:
            logger.error(f"Error processing email: {e}")

if __name__ == "__main__":
    try:
        logger.info(f"Starting Faust app with broker: {KAFKA_BROKER}")
        sys.argv = ['faust', 'worker', '-l', 'info']  # Set command-line arguments
        app.main()
    except Exception as e:
        logger.error(f"Faust Application error: {str(e)}")
        sys.exit(1)
