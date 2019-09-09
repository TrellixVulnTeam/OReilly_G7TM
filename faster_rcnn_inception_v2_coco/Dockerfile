FROM tensorflow/tensorflow:1.13.2-py3-jupyter
RUN apt-get update && apt-get install -y \
  git \
  nano \
  vim \
  curl \
  wget

WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive

# configure locale
RUN apt-get update
# make sure that locales package is available
RUN apt-get install --reinstall -y locales
# uncomment chosen locale to enable it's generation
RUN sed -i 's/# pl_PL.UTF-8 UTF-8/pl_PL.UTF-8 UTF-8/' /etc/locale.gen
# generate chosen locale
RUN locale-gen pl_PL.UTF-8
# set system-wide locale settings
ENV LANG pl_PL.UTF-8
ENV LANGUAGE pl_PL
ENV LC_ALL pl_PL.UTF-8
# verify modified configuration
RUN dpkg-reconfigure locales

# install the required libraries
RUN apt-get install -y protobuf-compiler \
  python-pil \
  python-lxml \
  python-tk \
	libkrb5-dev

RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user jupyter
RUN pip install --user matplotlib

# clone the models repo
RUN git clone https://github.com/tensorflow/models.git

WORKDIR /
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools /models/research

WORKDIR /models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc
RUN python setup.py install

WORKDIR object_detection

CMD ["echo", "Running tensorflow docker"]
