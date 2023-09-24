FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \ 
    zlib1g-dev \
    git \
    openssh-client \
    curl \
    libffi-dev \
    libbz2-dev \ 
    libreadline-dev \
    libsqlite3-dev \
    libmagic-dev \
    unstructured \
    chromadb

RUN wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
    tar -xvf Python-3.10.9.tgz && \
    cd /Python-3.10.9 && \
    ./configure --enable-optimizations && make && make altinstall

RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip

RUN pip install --upgrade pip

RUN pip install pysen \
    isort \
    flake8 \
    black \
    mypy \
    openai \
    llama-index  \
    langchain \
    tinysegmenter \
    logger \
    python-dotenv \
    google-api-python-client \
    lxml \
    requests_html  \
    requests  \
    wikipedia \
    youtube_transcript_api \
    llama_hub \
    streamlit

WORKDIR /workspaces/modern_chatbot
COPY . /workspaces/modern_chatbot/

