FROM python:3.8

WORKDIR /app

COPY ingest.py .
COPY requirements.txt .

# Install any required dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ingest.py"]