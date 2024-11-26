# Email-Classification-ai

Module for listing to a kafka topic and determing email is spam/ham. If email is Ham then classify into types(complaint/query/suggestions) and its sub types. Then publish to kafka topic.

Remove docker images

    docker-compose down

Just for building

    docker-compose build

Build with new code

    docker-compose up --build

Remove old venv

    rm -rf /Users/ranveersingh/Desktop/AI/email-classification-ai/venv

Install Python

    brew install python@3.10

Command for local execution

    python3.10 -m venv /Users/ranveersingh/Desktop/AI/email-classification-ai/venv

Activiate Env

    source /Users/ranveersingh/Desktop/AI/email-classification-ai/venv/bin/activate

Install library

    pip install -r requirements.txt
