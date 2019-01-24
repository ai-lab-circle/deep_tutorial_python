#!/bin/bash

for kfoldi in {1..5}
do
echo "doing fold $kfoldi"
~/caffe/build/tools/convert_imageset -shuffle -backend leveldb   ../DL_tutorial_Code/1-nuclei/subs/ ../DL_tutorial_Code/1-nuclei/train_w32_${kfoldi}.txt ../DL_tutorial_Code/1-nuclei/DB_train_${kfoldi} &
~/caffe/build/tools/convert_imageset -shuffle -backend leveldb   ../DL_tutorial_Code/1-nuclei/subs/ ../DL_tutorial_Code/1-nuclei/test_w32_${kfoldi}.txt ../DL_tutorial_Code/1-nuclei/DB_test_${kfoldi} &
done




FAIL=0
for job in `jobs -p`
do
    echo $job
    wait $job || let "FAIL+=1"
done




echo "number failed: $FAIL"



for kfoldi in {1..5}
do
echo "doing fold $kfoldi"
~/caffe/build/tools/compute_image_mean ../DL_tutorial_Code/1-nuclei/DB_train_${kfoldi} ../DL_tutorial_Code/1-nuclei/DB_train_w32_${kfoldi}.binaryproto -backend leveldb
done



