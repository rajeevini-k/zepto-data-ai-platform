FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY support_assistant /app/support_assistant

ENV MOCK_LLM=1

EXPOSE 8000

WORKDIR /app/support_assistant

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
