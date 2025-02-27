#!/bin/bash

# Download from: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
#wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz
#wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz
#wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz
#wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz

# For information https://cvg.cit.tum.de/data/datasets/rgbd-dataset

root_dir="/home/tuandang/workspace/datasets/tum"
download="${root_dir}/download"
extract="${root_dir}/extract"
# mkdir -p $download
# mkdir -p $extract 
echo $download
echo $extract

urls=("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
       "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz")


for i in "${!urls[@]}"; do
    url=${urls[$i]}
    file=$(echo "$url" | cut -d'/' -f7)
    wget $url -P $download 
    echo "Extrating : ${download}/${file}"
    tar -xvzf "${download}/${file}" -C $extract
done

# wget $url - P $download 
# 