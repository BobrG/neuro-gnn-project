FROM nvidia/cuda:11.1-devel-ubuntu16.04

WORKDIR /home

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget git htop vim && \
    apt-get clean

# install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/conda.sh && \
    /bin/bash ~/conda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# copy neuro project
RUN cd /home/ && git clone https://github.com/BobrG/neuro-gnn-project
WORKDIR /home/neuro-gnn-project

# create enviroment
RUN conda create --name neuro python=3.9
RUN echo "source activate neuro" > ~/.bashrc
ENV PATH $CONDA_DIR/conda3/envs/neuro/bin:$PATH
RUN $CONDA_DIR/envs/neuro/bin/pip install -r requirements.txt
