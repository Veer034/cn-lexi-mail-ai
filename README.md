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

    echo '{"tenantId": "fc569170-7573-428a-9333-eced51b1d712", "department": "ecommerce", "threadId": "1957fee209276d10","messageId": "79138fd0-d24a-4f23-bc22-66f065eb91fc", "subject": "Return order number 100011034", "emailBody": "Hi Team, I purchased an order from your website. And it turns out to be very bad. I want to return it, which I purchased on 01/01/20235. My order number is 100011034 I want details to return, and contact details. -- Ranveer Singh 8884524333", "senderName": "Ranveersingh"}' | jq -c . | docker exec -i broker kafka-console-producer \
    --bootstrap-server localhost:9092 \
    --topic email.classification.ai.request \
    --property "parse.key=false" \
    --property "key.separator=,"
