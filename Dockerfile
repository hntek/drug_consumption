# Docker file for drug_consumption_analysis

# Hatice Cavusoglu, Jan 27, 2018

# use rocker/tidyverse as the base image and

FROM rocker/tidyverse

# then install the ezknitr packages

RUN Rscript -e "install.packages('ezknitr', repos = 'https://mran.revolutionanalytics.com/snapshot/2017-12-11')"

# install python 3

RUN apt-get update \

  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


# get python package dependencies

RUN apt-get install -y python3-tk



# install packages

RUN pip3 install numpy

RUN pip3 install seaborn

RUN pip3 install pandas

RUN pip3 install scipy

RUN pip3 install sklearn

RUN pip3 install imblearn

RUN apt-get update && \
    pip3 install matplotlib && \

    rm -rf /var/lib/apt/lists/*
