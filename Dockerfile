FROM quay.io/jupyter/pyspark-notebook:2026-08-07
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.9-src.zip
ENV PATH=${SPARK_HOME}/bin:${PATH}

WORKDIR /home/jovyan/work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
