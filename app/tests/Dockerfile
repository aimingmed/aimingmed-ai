# TEST
FROM python:3.11-slim-bullseye

WORKDIR /usr/src/app

COPY Pipfile .

RUN pip3 install --upgrade pip &&\
    pip3 install pipenv &&\
    pipenv install --dev

COPY tests ./tests

ENTRYPOINT ["pipenv", "run", "pytest", "--junit-xml=results/results.xml"]
