FROM ubuntu
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \ 
    zlib1g-dev \
    python3-pip \
    git

RUN chmod 777 .
RUN wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz
RUN tar -xvf Python-3.9.5.tgz
WORKDIR /Python-3.9.5
RUN pip install --upgrade pip
RUN ./configure --enable-optimizations && make && make altinstall
RUN ln -s /usr/local/bin/python3.9 /usr/local/bin/python
RUN python --version

ENV GIT_EMAIL yuuyauchi1998@example.com
ENV GIT_NAME yuyauchi
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

ENV username="user1"
RUN useradd --create-home --shell /bin/bash -G sudo,root $username
RUN echo 'sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $username

ENV PATH="/home/${username}/.local/bin:${PATH}"
WORKDIR /practice