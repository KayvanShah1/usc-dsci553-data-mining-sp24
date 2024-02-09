FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk python3.6 python3-pip wget software-properties-common

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

RUN apt-get install -y gnupg2 && \
    wget www.scala-lang.org/files/archive/scala-2.12.15.deb && \
    dpkg -i scala-2.12.15.deb && \
    apt-get update && \
    apt-get install -y scala

RUN wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /spark

ENV SPARK_HOME=/spark

ENV PATH=$PATH:$SPARK_HOME/bin:/usr/bin/scala:usr/bin/python3.6

RUN pip3 install pyspark findspark

WORKDIR /workspace

CMD ["/bin/bash"]