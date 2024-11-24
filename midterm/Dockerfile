# First install the python 3.8, the slim version uses less space
FROM python:3.10.15-slim-bookworm

RUN apt update && \
    apt-get install -y curl gcc python3-dev pkg-config libxml2-dev libxmlsec1-dev libxmlsec1-openssl

RUN adduser --home=/home/mlz mlz && usermod -a -G 0 mlz

RUN pip install --upgrade pip

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 -

RUN chgrp -R 0 /opt/poetry && \
    chmod -R g=u /opt/poetry

USER mlz
ENV PATH="/opt/poetry/bin:$PATH"

WORKDIR /home/mlz

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --without dev --no-root
COPY ["predict.py", "*.joblib", "./"]

EXPOSE 9696

CMD ["poetry", "run", "gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "predict:app", "--bind", "0.0.0.0:9696"]