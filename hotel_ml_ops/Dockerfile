FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN mkdir -p data && python scripts/serialize_data.py

CMD ["python", "tests/test_data_quality.py"]
