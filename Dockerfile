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
    libmagic-dev

RUN wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
    tar -xvf Python-3.10.9.tgz && \
    cd /Python-3.10.9 && \
    ./configure --enable-optimizations && make && make altinstall

RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip

RUN pip install --upgrade pip

WORKDIR /project
COPY . /project/
RUN pip install pysen isort flake8 black mypy
# COPY ./requirements.txt /project/requirements.txt
# RUN pip install -r requirements.txt
# COPY . /project/

# ENV username="user1"
# RUN useradd -p yy1998  --create-home --shell /bin/bash -G sudo,root $username
# RUN echo 'sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER $username
# ENV PATH="/home/${username}/.local/bin:${PATH}"

WORKDIR /project