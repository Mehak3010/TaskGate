FROM python:3.11-slim

WORKDIR /app
COPY data/ /app/data/
COPY tests/ /app/tests/
RUN pip install --no-cache-dir pytest==8.2.0

CMD ["sleep", "infinity"]
