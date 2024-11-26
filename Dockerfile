# # Start from a Python base image
# FROM python:3.8-slim-buster

# # Set the working directory in the container
# WORKDIR /app

# # Copy only the requirements file first to leverage Docker layer caching
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt
    
    
# # Expose port 5002 to the outside world
# EXPOSE 5002

# # Run the Flask app with Gunicorn for production
# CMD ["gunicorn", "-b", "0.0.0.0:5002", "main:app", "--workers=3", "--threads=2", "--timeout=120"]
