#!/bin/bash

NUM_ITER=2
NUM_TOPICS_NEWS=20
NUM_CORES=1
FILE_NAME_CONF_CLEAN="dataset/"
OUTPUT_FILE="output/"
# change the compiler
#PyCC="python"
PyCC="/usr/share/anaconda/bin/python"



for i in `seq 1 1`
do
    $PyCC tm_all_default.py $FILE_NAME_CONF_CLEAN $NUM_TOPICS_NEWS $NUM_ITER $NUM_CORES > $OUTPUT_FILE"news_cleanText_"$i
done;

