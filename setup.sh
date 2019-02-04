#/bin/bash

# Install python dependencies
pip install -r requirements.txt

# Compile custom CUDA op
cd tf_grouping && ./tf_grouping_compile.sh && cd ..

