#! /bin/bash

# create directory .dataset/test

directory=$1
tif_dir=$2
shp_dir=$3

if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi
# saved dataset into numpy form in .dataset/test directory
python3 prepareDataset.py --path $directory --tif_dir $tif_dir --shp_dir $shp_dir