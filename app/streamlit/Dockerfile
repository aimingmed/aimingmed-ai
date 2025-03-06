FROM python:3.11-slim

WORKDIR /app/streamlit

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY Chatbot.py .
COPY .env .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Chatbot.py"]