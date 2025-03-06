FROM python:3.11-slim

WORKDIR /app/streamlit

COPY Pipfile ./

RUN pip install pipenv && pipenv install --system --deploy

COPY Chatbot.py .
COPY .env .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Chatbot.py"]