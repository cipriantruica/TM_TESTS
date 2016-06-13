#!/bin/bash


NUM_ITER=2000
NUM_TOPICS_CONF=5
NUM_TOPICS_NEWS=20
NUM_CORES=1
DB_CONF="conf"
DB_NEWS="news"
FILE_NAME_CONF_LEMMA="mallet_dataset/conf_lemmaText/"
FILE_NAME_CONF_CLEAN="mallet_dataset/conf_cleanText/"
FILE_NAME_NEWS_LEMMA="mallet_dataset/news_lemmaText/"
FILE_NAME_NEWS_CLEAN="mallet_dataset/news_cleanText/"
OUTPUT_FILE="output_tm_defaults/"
PyCC="/usr/share/anaconda/bin/python"



#for i in `seq 1 6`
#do
#    $PyCC tm_all_default.py $FILE_NAME_CONF_LEMMA $DB_CONF $NUM_TOPICS_CONF $NUM_ITER $NUM_CORES > $OUTPUT_FILE"conf_lemmaText_"$i
#done;


#for i in `seq 1 6`
#do
#    $PyCC tm_all_default.py $FILE_NAME_CONF_CLEAN $DB_CONF $NUM_TOPICS_CONF $NUM_ITER $NUM_CORES > $OUTPUT_FILE"conf_cleanText_"$i
#done;


for i in `seq 1 6`
do
    $PyCC tm_all_default.py $FILE_NAME_NEWS_LEMMA $DB_NEWS $NUM_TOPICS_NEWS $NUM_ITER $NUM_CORES > $OUTPUT_FILE"news_lemmaText_t"$i
done;


for i in `seq 1 6`
do
    $PyCC tm_all_default.py $FILE_NAME_NEWS_CLEAN $DB_NEWS $NUM_TOPICS_NEWS $NUM_ITER $NUM_CORES > $OUTPUT_FILE"news_cleanText_t"$i
done;

