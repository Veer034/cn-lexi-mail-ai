About

    LexiMail AI is an advanced GenAI-powered email assistant designed to understand, classify, and generate intelligent responses for incoming emails. Whether it's queries, complaints, or suggestions, LexiMail AI ensures quick, accurate, and context-aware replies, seamlessly integrating with your workflow.

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

    pip install "numpy<2.0.0"  sentence-transformers torch elasticsearch confluent-kafka httpx aiohttp python-json-logger transformers python-dotenv nltk langdetect fastapi uvicorn gunicorn pydantic asyncio sentencepiece spacy

Start in local

    python3.10 master.py

deactivate your virtual environment if it's active:

    deactivate

---

# Production Setup

### Login VM

    ssh azureuser@YOUR-VM-PUBLIC-IP

### Install Git

    sudo apt update
    sudo apt install git -y
    git clone https://github.com/Veer034/cn-lexi-mail-ai.git

### Install Python

    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install python3.10 python3.10-venv python3.10-distutils

### Install library in production VM

    pip install "numpy<2.0.0"  sentence-transformers torch elasticsearch confluent-kafka httpx aiohttp python-json-logger transformers python-dotenv nltk langdetect fastapi uvicorn gunicorn pydantic asyncio sentencepiece spacy

### Create Systemd file for as a service execution

    sudo tee /etc/systemd/system/cn-lexi-mail-ai.service > /dev/null << EOF
    [Unit]
    Description=For data forging
    After=network.target ollama.service
    Requires=ollama.service

    [Service]
    Type=simple
    User=azureuser
    WorkingDirectory=/home/azureuser/cn-lexi-mail-ai
    Environment=PATH=/home/azureuser/cn-lexi-mail-ai/myvenv/bin
    ExecStart=/home/azureuser/cn-lexi-mail-ai/myvenv/bin/python master.py
    Restart=always
    RestartSec=10
    StandardOutput=journal
    StandardError=journal

    [Install]
    WantedBy=multi-user.target
    EOF

### Copy trained model from GCP cloud storage

    # Install Google Cloud SDK
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL

    # Initialize and authenticate
    gcloud init
    gcloud auth login

    # Download single file
    gsutil cp gs://your-bucket-name/path/to/file.txt ./

    # Download entire directory
    gsutil -m cp -r gs://your-bucket-name/directory/ ./

    # Download with progress and resume capability
    gsutil -m cp -r -n gs://your-bucket-name/large-directory/ ./

### HuggingFace model storage location

    ~/.cache/huggingface/

### List all services

    systemctl list-units --type=service
    systemctl list-units --type=service | grep cn-

### Reload systemd

    sudo systemctl daemon-reload

### Enable all services to start on boot

    sudo systemctl enable cn-lexi-mail-ai

### Start Service

    sudo systemctl start cn-lexi-mail-ai

### Check Status

    sudo systemctl status cn-lexi-mail-ai

### Stop service

    sudo systemctl stop cn-lexi-mail-ai

### Restart service

    sudo systemctl restart cn-lexi-mail-ai

### Check logs for specific service

    sudo journalctl -u cn-lexi-mail-ai -f

### Check service generated logs

    tail -n 50 ~/cn-lexi-mail-ai/logs/server.log

### Check logs for that service

    journalctl -u cn-lexi-mail-ai.service

### Rotate the journal for that service (so old logs can be vacuumed)

    sudo journalctl --unit=cn-lexi-mail-ai.service --rotate

### Delete old logs for that service

    sudo journalctl --unit=cn-lexi-mail-ai.service --vacuum-time=1s


    # Or to keep only the last 7 days:
    sudo journalctl --unit=cn-lexi-mail-ai.service --vacuum-time=7d

---

### List kafka topics

kafka-topics.sh --list --bootstrap-server 57.159.53.43:9092

### Create a specific topic

kafka-topics.sh --create --topic email.classification.ai.request --bootstrap-server 57.159.53.43:9092

### Describe a specific topic

kafka-topics.sh --describe --topic email.classification.ai.request --bootstrap-server 57.159.53.43:9092

### Describe all topics

kafka-topics.sh --describe --bootstrap-server 57.159.53.43:9092

### Describe multiple specific topics

kafka-topics.sh --describe --topic email.classification.ai.request,email.classification.ai.response,email.classification.ai.request_DLQ --bootstrap-server 57.159.53.43:9092

Kafka email message for classification :

    echo '{"tenantId": "fc569170-7573-428a-9333-eced51b1d712", "department": "ecommerce", "threadId": "1957fee209276d10","messageId": "79138fd0-d24a-4f23-bc22-66f065eb91fc", "subject": "Return order number 100011034", "emailBody": "Hi Team, I purchased an order from your website. And it turns out to be very bad. I want to return it, which I purchased on 01/01/20235. My order number is 100011034 I want details to return, and contact details. -- Ranveer Singh 8884524333", "senderName": "Ranveersingh"}' | jq -c . | docker exec -i broker kafka-console-producer \
    --bootstrap-server localhost:9092 \
    --topic email.classification.ai.request \
    --property "parse.key=false" \
    --property "key.separator=,"
