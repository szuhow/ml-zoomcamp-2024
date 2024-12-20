# First install the python 3.8, the slim version uses less space
FROM python:3.10.15-slim-bookworm

RUN apt update && \
    apt-get install -y curl gcc python3-dev pkg-config libxml2-dev libxmlsec1-dev libxmlsec1-openssl

RUN adduser --home=/home/mlz mlz && usermod -a -G 0 mlz

RUN pip install --upgrade pip

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements-docker.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]