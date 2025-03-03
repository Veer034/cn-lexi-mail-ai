Remove docker images

    docker-compose down

Just for building

    docker-compose build

Build with new code

    docker-compose up --build

Remove old venv

    rm -rf venv

Install Python

    brew install python@3.10

Activiate Env

    #can change the env names
    python3.10 -m venv myvenv
    source myvenv/bin/activate

    python --version

Install library in local VM

    #Required for kafka confluent
    brew install librdkafka


    pip install "numpy<2.0.0"  sentence-transformers torch elasticsearch confluent-kafka httpx aiohttp python-json-logger transformers python-dotenv nltk langdetect fastapi uvicorn gunicorn pydantic asyncio sentencepiece

Start in local

    python3.10 master.py

deactivate your virtual environment if it's active:

    deactivate

Kafka email message for classification :

    echo '{
    "emailBody": "Hello, I need help with my profile details. Can I request deletion of my profile information stored at time of registration?",
    "tenantId": "tenant123",
    "threadId": "67890",
    "department": "ecommerce",
    "senderName": "John Doe"
    }' | jq -c . | docker exec -i broker kafka-console-producer \
    --bootstrap-server localhost:9092 \
    --topic email.classification.ai.request \
    --property "parse.key=false" \
    --property "key.separator=,"
