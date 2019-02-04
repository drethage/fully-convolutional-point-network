config.json - Configuring a training session
----

The parameters present in a `config.json` file allow one to configure a training session. Each of these parameters is described below:

### Dataset
Where to find and how to handle the data
* **classes**: *list[string]* A list of class names representing the subset of classes to train with all other classes will be masked
* **original_file_suffix**: *string* The suffix of all PLY files present in the dataset featuring non-uniform point sampling
* **path**: *string* An absolute path to the root of the dataset
* **refresh_cache**: *bool* Whether to invalidate an existing cache and create it again from scratch
* **special_weights**:
  * **empty**: *float* The weight given to empty *unoccupied* voxels in the loss function
  * **masked**: *float* The weight given to masked voxels in the loss function
* **training_samples**:
  * **num_points**: *int* Number of points present in a training sample
  * **spatial_size**: *list[float]* The spatial size in x, y and z dimensions of a training sample. Must be divisible by 0.6
* **uniform_file_suffix**: *string* The suffix of all PLY files present in the dataset featuring uniform point sampling *use ClassyVoxelizer to uniformly sample meshes/point clouds*
* **validity_thresholds**:
  * **empty**: *float* The maximum percentage of empty volume relative to a sample's entire volume permitted for that sample to be considered valid
  * **empty**: *float* The maximum percentage of masked points relative to a sample's entire points permitted for that sample to be considered valid

### Model
Hyperparameters of the model
* **filters**:
  * **abstraction**:
    * **points_to_15cm**: *list[int]* Depths of the filters in the simplified PointNets
    * **15cm_to_30cm**: *list[int]* Depths of the 3D convolution and following 1x1x1 convolution filters abstracting from the 15cm to the 30cm spatial scale
    * **30cm_to_60cm**: *list[int]* Depths of the 3D convolution and following 1x1x1 convolution filters abstracting from the 30cm to the 60cm spatial scale
  * **skip**:
    * **15cm**: *int* Filter depth of the skip connection propagating features at the 15cm spatial scale
    * **30cm**: *int* Filter depth of the skip connection propagating features at the 30cm spatial scale
    * **45cm**: *int* Filter depth of the skip connection propagating features at the 45cm spatial scale
    * **60cm**: *int* Filter depth of the skip connection propagating features at the 60cm spatial scale
    * **90cm**: *int* Filter depth of the skip connection propagating features at the 90cm spatial scale
    * **180cm**: *int* Filter depth of the skip connection propagating features at the 180cm spatial scale
    * **spatial_pool**: *int* Filter depth of the skip connection propagating spatially-pooled features at the 60cm spatial scale
  * **upsampling**:
    * **60cm_to_30cm**: *list[int]* Depths of the 3D deconvolution and following 1x1x1 convolution filters upsampling from the 60cm to the 30cm spatial scale
    * **30cm_to_15cm**: *list[int]* Depths of the 3D convolution and following 1x1x1 convolution filters abstracting from the 30cm to the 15cm spatial scale
    * **15cm_to_5cm**: *list[int]* Depths of the 3D convolution and following 1x1x1 convolution filters abstracting from the 15cm to the 5cm spatial scale
* **pointnet**:
  * **neighbors**: *int* Number of neighboring points to group around a PointNet's center
  * **spacing**: *float* Distance between the centers of two PointNets
* **spatial_pool_radius**: *float* The radial distance from the center of a voxel where the responses of its neighboring voxels will be weighted the highest

### Training
How training will be performed
* **batch_size**: *int* Number of samples in a batch
* **checkpoint_path**: *string* Path to a Tensorflow checkpoint. If non-empty, this new training session will start with weights from the specified checkpoint
* **checkpoints_to_keep**: *int* Number of checkpoints to keep in session directory before deleting
* **data_augmentation**:
  * **jitter**:
    * **max**: *float* Maximum distance permitted to move a point
    * **sigma**: *float* Sigma value of the normal distribution used to jitter points
  * **random_dropout**:
    * **max**: *float* Maximum permitted percentage of a training sample that can be removed
  * **random_rotation**: *bool* Whether to randomly rotate a training sample about the +Z axis
  * **shuffle**: *bool* Whether to shuffle the samples in the training set at the beginning of each epoch
* **description**: *string* Textual description of the training session
* **finetune_on_new_classes**: *bool* Set to true when finetuning a pretrained model on a new dataset with a different set of trainable classes. If true, the weights of the model's final layer will not be loaded
* **gpu**:
  * **allow_growth**: *bool* Whether to incrementally reserve more memory on the GPU as opposed to allocating all available memory from the beginning
  * **allow_soft_placement**: *bool* Whether to permit certain ops being executed by another device *cpu/gpu* if its not possible to execute it on the preferred device
  * **id**: *int* Numerical identifier of the GPU to use for training
* **log_every_n_batches**: *int* Number of batches after which to log the current state of the training
* **max_epochs**: *int* Maximum number of epochs to train for
* **optimizer**: Model is trained using the Adam optimizer
  * **batch_normalization**:
    * **decay_rate**: *float* Rate at which batch normalizatio momentum exponentially decays
    * **decay_step**: *int* Number of training samples after which the batch normalization momentum is decreased one step
    * **initial_decay**: *float* Initial decay of batch normalization momentum
  * **learning_rate**:
    * **decay_rate**: *float* Rate at which the learning rate exponentially decays
    * **decay_step**: *int* Number of training samples after which the learning rate is decreased one step
    * **initial**: *float* Initial learning rate
  * **random_rotation**: *bool* Whether to randomly rotate a training sample about the +Z axis
  * **shuffle**: *bool* Whether to shuffle the samples in the training set at the beginning of each epoch
* **save_every_n_batches**: *int* Number of batches after which to save a snapshot of the model
* **session_id**: *int* Numerical identifier for this session
* **upsampling_only**: *bool* Whether to only train the final upsampling layers of the model. Useful for finetuning a pretrained model on a smaller dataset