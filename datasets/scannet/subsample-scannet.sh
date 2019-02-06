#!/bin/bash

# Download and compile ClassyVoxelizer tool
if [ ! -f ClassyVoxelizer/bin/classy_voxelizer ]; then
    git clone https://github.com/drethage/ClassyVoxelizer.git
    cd ClassyVoxelizer
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    cd ../../
fi

mkdir data

# Uniformly sample training and validation sets
cat metadata/train_split.txt metadata/validation_split.txt | while read scene; do
    input_suffix="_vh_clean_2.labels.ply"
    output_suffix="_vh_clean_2.labels.uniform_2cm.ply"
    mkdir data/$scene
    mv scans/$scene/$scene$input_suffix data/$scene/$scene$input_suffix
    ClassyVoxelizer/bin/classy_voxelizer data/$scene/$scene$input_suffix data/$scene/$scene$output_suffix 0.02 labels false
    echo "Subsampled: $scene"
done

echo "Finished subsampling training and validation sets."

# Uniformly sample test set
cat metadata/test_split.txt | while read scene; do
    input_suffix="_vh_clean_2.ply"
    output_suffix="_vh_clean_2.uniform_2cm.ply"
    mkdir data/$scene
    mv scans_test/$scene/$scene$input_suffix data/$scene/$scene$input_suffix
    ClassyVoxelizer/bin/classy_voxelizer data/$scene/$scene$input_suffix data/$scene/$scene$output_suffix 0.02 none false
    echo "Subsampled: $scene"
done


# Clean Up
rm -r tasks
rm -r scans
rm -r scans_test
rm -r scannetv2-labels.combined.tsv

echo "Finished Subsampling ScanNet."
