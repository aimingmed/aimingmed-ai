FROM python:3.11-slim

WORKDIR /app/streamlit

COPY requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

COPY Chatbot.py .
COPY .env .

# Run python to initialize download of SentenceTransformer model
COPY initialize_sentence_transformer.py .
RUN python initialize_sentence_transformer.py

COPY pages ./pages

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Chatbot.py"]