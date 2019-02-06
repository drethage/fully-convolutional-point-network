Datasets for trainings Fully-Convolutional Point Network
====

This project was originally developed using [ScanNet](http://www.scan-net.org), but is not ScanNet specific. Minimal effort should be required to train and evaluate this method on new datasets.

### Dataset Structure

A dataset consists of **data** (a set of PLY files) and some **metadata** organized as below:

* `dataset_root/`
  * `data/` contains N folders for N items in the dataset. The folder name is a prefix of the PLY file contained therein.
  * `metadata/` contains a set of text files describing the subsets of the dataset as well as the semantic classes present.
    * `class_names.txt` contains one class name per line where the 0-based line number of a particular class is the ID of the label attribute on points in the dataset that are labeled with this class.
    * `colors.txt` The set of colors used to represent each class uniquely. Colors are defined one per line by space-separated RGB values.
    * `train_split.txt` The set of items in the dataset that are in the training set. Each line in this file should correspond to a folder in `data/`.
    * `test_split.txt` The set of items in the dataset that are in the test set. Each line in this file should correspond to a folder in `data/`
    * `validation_split.txt` The set of items in the dataset that are in the validation set. Each line in this file should correspond to a folder in `data/`

This structure is exemplified by the `scannet/` folder in `datasets/`.

### Uniform Sampling

It is advantageous, although not necessary, to ensure uniform point density throughout all items in the dataset during training. However, real-world acquired scans often exhibit non-uniform point density. To mitigate this, one can use the [ClassyVoxelizer](https://github.com/drethage/ClassyVoxelizer) tool to convert non-uniformly dense meshes or point clouds into uniformly-sampled point clouds. A 2cm sampling is recommended. During training, point-dropout augmentation is used to improve generalizability to non-uniform point density.

### ScanNet

This project was developed and evaluated with the help of [ScanNet](http://www.scan-net.org). Therefore, metadata and a convenience script for preparing ScanNet have been provided. For re-training or evaluating a model on ScanNet, perform the following steps:

1. Fill out an agreement to the [ScanNet Terms of Use](http://dovahkiin.stanford.edu/scannet-public/ScanNet_TOS.pdf) and send it to [scannet@googlegroups.com](scannet@googlegroups.com). Upon approval, you will receive a `download-scannet.py` that should be placed in `fcpn/dataset/scannet/`.
2. Download all labeled scans via `python download-scannet.py -o . --type _vh_clean_2.labels.ply`. Can take 1-2 hours depending on your network connection.
3. Download all unlabeled scans via `python download-scannet.py -o . --type _vh_clean_2.ply`. Can take 1-2 hours depending on your network connection.
4. Run `./subsample-scannet.sh`.

This will clone and compile the [ClassyVoxelizer](https://github.com/drethage/ClassyVoxelizer) tool and subsequently uniformly subsample every scan at a 2cm point density.