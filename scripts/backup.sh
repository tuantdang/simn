#!/bin/bash
message=$1
echo "Backing up with commit message: ${message}"

# Remove some heavy log folders
base_dir="./log/"
tumdataset="${base_dir}tum/"
for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then

        if [ "$dir" == "$tumdataset" ]; then
            for sub_dir in "$dir"*/; do
                if [ -d "$sub_dir" ]; then
                    echo "Remove files and folders in ${sub_dir}"
                    # rm -rf "${sub_dir}*.csv" || true
                    for sub_sub_dir in "$sub_dir"*/; do
                        if [ -d "$sub_sub_dir" ]; then
                            echo "  Folder ${sub_sub_dir}"
                            rm -rf $sub_sub_dir || true
                        fi
                    done
                fi
            done
        else
            echo "Remove files and folders in ${dir}"
            for sub_dir in "$dir"*/; do
                if [ -d "$sub_dir" ]; then
                    echo "  Folder ${sub_sub_dir}"
                    rm -rf $sub_dir || true
                fi
            done
        fi
    fi
done

# Remove weights
# rm -f weights/*

# Upload to github
git add . 
git commit -m "${message}"
git push
