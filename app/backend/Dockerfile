# pull official base image
FROM python:3.11-slim-bookworm

# create directory for the app user
RUN mkdir -p /home/app

# create the app user
RUN addgroup --system app && adduser --system --group app

# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/backend
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=dev
ENV TESTING=1
ENV CUDA_VISIBLE_DEVICES=""

COPY Pipfile $APP_HOME/
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pipenv && rm -rf ~/.cache/pip
RUN pipenv install --deploy --dev --no-cache-dir
RUN pipenv run pip install torch --force-reinstall --no-cache-dir

# remove all cached files not needed to save space
RUN pip cache purge
RUN rm -rf /root/.cache

# add app
COPY . $APP_HOME

# Create cache directory and set permissions
RUN mkdir -p /home/app/.cache/huggingface
RUN chown -R app:app /home/app/.cache/huggingface

RUN chown -R app:app $APP_HOME

# change to the app user
USER app

# Run python to initialize download of SentenceTransformer model
RUN pipenv run python utils/initialize_sentence_transformer.py

# pytest
RUN export DEEPSEEK_API_KEY=sk-XXXXXXXXXX; export TAVILY_API_KEY=tvly-dev-wXXXXXX;\
    pipenv run pytest tests --disable-warnings

# expose the port the app runs on
EXPOSE 80

# run uvicorn
CMD ["pipenv", "run", "uvicorn", "main:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "80"]

