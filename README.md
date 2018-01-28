# Drug Consumption

Drug addiction is one of the most serious issues the society is facing today. In this report that used this repo to create, I studied models predicting drug usage based on the personality profiles and demographics.

The analysis is based on the dataset from [Fehrman et al. (2015)](https://arxiv.org/abs/1506.06297) and accessible at [UCI ML Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data). I also mirrored the dataset [here](data/drug_consumption.csv).

## Key Files

- [Report: Who is \"High\"?: Predicting Drug Usage](documents/project_report.md)
- [Source code of the report](src/project_report.Rmd)
- [Source Code for the analysis](src/q4_script.ipynb)


## Running Relevant Files individually

To create the report, type the following commands on the bash while you are on the project's root directory:

```
python src/analysis.py
python src/exploratory.py
Rscript -e "ezknitr::ezknit('src/project_report.Rmd', out_dir = 'documents')"
```
The rendered report can be seen [here](documents/project_report.md).

## Running Automatic Pipeline from the Local Machine

To create the report with the Makefile, type the following command on the bash while you are on the project's root directory:

```
make all
```

## Building Docker Image from Dockerfile

When you are on the project root directory in which Dockerfile resides, you can build the docker image from the Docker file.

```
docker build -t drug_consumption .
```

## Pulling Docker Image from Dockerhub

I have pushed the docker image built to Docker Hub. It resides at https://hub.docker.com/r/hntek/drug_consumption/. To pull the docker image, type the following on the bash.
```
docker pull hntek/drug_consumption
```
## Running Automatic Pipeline from the Virtual Machine 

### Running make from the docker volume interactively

```
docker run  --rm -it -v ~/Documents/2017lecture/573_DSCI_2017/lab/drug_consumption:/home/drug_consumption drug_consumption /bin/bash

make all
```
### Running make from the docker volume non-interactively
```
docker run  --rm -v ~/Documents/2017lecture/573_DSCI_2017/lab/drug_consumption:/home/drug_consumption drug_consumption make -C 'home/drug_consumption' all
```
## Dependencies

Python and R are used in this project. Here are the dependencies.

## Python Dependencies

- python3
- python3-dev
- pip3
- python3-tk
- numpy
- matplotlib
- seaborn
- pandas
- sklearn
- imblearn


# R Dependencies

- tidyverse
- ezknitr
- knitr








