FROM tensorflow/tensorflow:1.11.0

RUN apt-get update -y && apt-get install apt-transport-https -y

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update -y
RUN apt-get install edgetpu-compiler -y
RUN apt-get install git -y
RUN apt-get install openssh-client -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

ADD /Dockerfile/id_rsa /key/
RUN chmod 600 /key/id_rsa
# Host key verification failed.
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config

# Fixed code base
RUN git clone https://github.com/KodeWorker/tflite_conversion.git /tflite_conversion