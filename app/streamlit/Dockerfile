FROM python:3.11-slim

WORKDIR /app/streamlit

COPY Pipfile ./

# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt
RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pipenv -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pipenv install --deploy

COPY Chatbot.py .
COPY .env .

# Run python to initialize download of SentenceTransformer model
COPY initialize_sentence_transformer.py .
RUN pipenv run python initialize_sentence_transformer.py

COPY pages ./pages

EXPOSE 8501

ENTRYPOINT ["pipenv", "run", "streamlit", "run", "Chatbot.py"]