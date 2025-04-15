# pull official base image
FROM python:3.11-slim-bullseye

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
ENV ENVIRONMENT=prod
ENV TESTING=0


# add app
COPY . $APP_HOME

# install python dependencies
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pipenv && rm -rf ~/.cache/pip
RUN pipenv install --deploy --dev

# chown all the files to the app user
RUN chown -R app:app $APP_HOME

# change to the app user
USER app

# expose the port the app runs on
EXPOSE 8765

# run uvicorn
CMD ["pipenv", "run", "uvicorn", "main:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8765"]