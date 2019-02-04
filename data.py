""" data.py - Defines the Dataset class, which handles all interactions with the dataset used to train a model. """

from __future__ import division
import os
import util
import pickle
import joblib
import json
import random
import multiprocessing
import collections
ProcessedItem = collections.namedtuple(
    'ProcessedItem', ['points', 'classes', 'sample_locations', 'class_counts'])
import math
import numpy as np

class Dataset(object):

    def __init__(self, config):
        """ Creates a Dataset object.

        Args:
        config: dict, session configuration parameters
        
        """

        if not os.path.exists(config['dataset']['path']):
            print 'Error: No dataset found at specified path.'
            exit()

        self._dataset_path = config['dataset']['path']
        self._dataset_data_path = os.path.join(self._dataset_path, 'data/')
        self._dataset_metadata_path = os.path.join(self._dataset_path, 'metadata/')

        # Suffix of files in the dataset featuring uniform sampling
        self._uniform_file_suffix_ = config['dataset']['uniform_file_suffix']

        # The spatial extent of a single sample
        self._training_sample_spatial_size = np.array(config['dataset']['training_samples']['spatial_size'])

        # The amount by which to move spatially when extracting samples from an item in the dataset
        self._sample_step_size = np.mean(
            [self._training_sample_spatial_size[0], self._training_sample_spatial_size[1]]) / 3
        
        self._voxel_size = config['model']['output_voxel_size']
        self._validity_thresholds = config['dataset']['validity_thresholds']
        self._special_weights = config['dataset']['special_weights']
        
        # The locations in each dataset item where a valid sample can be extacted from
        self._sample_locations = {}


        self._num_samples = {'train': 0, 'val': 0}
        self._points = {}
        self._classes = {}

        # Load metadata
        self._item_ids = {
            'train': util.read_file_to_list(os.path.join(self._dataset_metadata_path, 'train_split.txt')),
            'val': util.read_file_to_list(os.path.join(self._dataset_metadata_path, 'validation_split.txt'))
        }
        self._all_classes = util.read_file_to_list(os.path.join(self._dataset_metadata_path, 'class_names.txt'))
        self._learnable_classes = list(config['dataset']['classes']) # Make a copy
        self._colors = [[int(c) for c in color.split(' ')] for color in util.read_file_to_list(os.path.join(self._dataset_metadata_path, 'colors.txt'))]

        # Map dataset classes to learnable classes
        self._all_to_learnable_class_mapping = {}
        for learnable_class_i, learnable_class in enumerate(self._learnable_classes):

            if learnable_class not in self._all_classes:
                print "Error: classes list, not all classes present in class_names.txt"
                exit()

            self._all_to_learnable_class_mapping[self._all_classes.index(learnable_class)] = learnable_class_i

        self._learnable_classes += ['empty', 'masked']
        self._masked_class_id = len(self._learnable_classes) - 1
        self._empty_class_id = len(self._learnable_classes) - 2
        self._num_dataset_classes = len(self._all_classes)
        self._num_learnable_classes = len(self._learnable_classes)        

        # Add these values to config so they are available during evaluation/inference
        config['dataset']['empty_class_id'] = self._empty_class_id
        config['dataset']['num_learnable_classes'] = self._num_learnable_classes

        config_for_hash = config['dataset'].copy()
        config_for_hash.pop('refresh_cache', None)

        self.cache_hash_ = hash(
            "".join(self._learnable_classes) +
            "".join(self._item_ids['train']) +
            "".join(self._item_ids['val']) +
            str(self._sample_step_size) +
            json.dumps(config_for_hash, sort_keys=True)
        )

    def get_dataset_metadata_path(self):
        """ Get path of the metadata folder.

        Returns: string

        """

        return self._dataset_metadata_path

    def get_dataset_data_path(self):
        """ Get path of the data folder.

        Returns: string

        """

        return self._dataset_data_path    

    def get_colors(self):
        """ Get path of the metadata folder.

        Returns: string

        """

        return self._colors

    def get_learnable_classes_strings(self):
        """ Get the names of classes that will be considered during training.

        Returns: list[string]

        """

        return self._learnable_classes

    def get_class_weights(self):
        """ Get precomputed class weights.

        Returns:
        class_weights: np.array

        """

        return self._class_weights

    def get_num_dataset_classes(self):
        """ Get the number of classes present in the dataset.

        Returns: int

        """
        return self._num_dataset_classes

    def get_num_learnable_classes(self):
        """ Get the number of classes that will be considered during training.

        Returns:
        num_classes: int

        """
        return self._num_learnable_classes

    def get_num_samples(self, set_type):
        """ Get the number of samples in set_type.
            A sample refers to a particular region within a dataset item.

        Args:
        set_type: string, either 'train' or 'val'

        Returns: int

        """
        return self._num_samples[set_type]

    def get_num_batches(self, set_type, batch_size):
        """ Return number of batches in set_type for a given batch_size.

        Args:
        set_type: string, either 'train' or 'val'
        batch_size: int

        Returns: int

        """
        return int(math.ceil(self._num_samples[set_type] / batch_size))

    def get_num_dataset_items(self, set_type):
        """ Return number of samples in set_type provided by the dataset.

        Args:
        set_type: string, either 'train' or 'val'

        Returns: int

        """
        return len(self._item_ids[set_type])

    def get_num_output_voxels(self):
        """ Return number of output elements (voxels), which are produced by the last layer of the network.

        Returns: int

        """
        return np.prod(np.ceil(self._training_sample_spatial_size / self._voxel_size))

    def get_empty_class(self):
        """ Return the ID of the class representing unoccupied space.

        Returns: int

        """
        return self._empty_class_id

    def is_sample_valid(self, voxelgrid, occupancy_threshold, annotation_threshold):
        """ Tests whether the given voxelgrid represents a valid training sample, based on its percent occupancy and percent annotated

        Args:
        voxelgrid: np.array, a 3D grid
        occupancy_threshold: float
        annotation_threshold: float

        Returns: bool

        """

        num_unoccupied_voxels = (voxelgrid == self._empty_class_id).sum()
        num_voxels = voxelgrid.size
        percent_unoccupied = num_unoccupied_voxels / num_voxels
        percent_occupied = 1 - percent_unoccupied

        if percent_occupied < occupancy_threshold:
            return False

        num_unannotated_voxels = (voxelgrid == self._masked_class_id).sum()
        num_occupied_voxels = (voxelgrid != self._empty_class_id).sum()
        percent_unannotated = num_unannotated_voxels / num_occupied_voxels
        percent_annotated = 1 - percent_unannotated

        if percent_annotated < annotation_threshold:
            return False

        return True

    def map_all_to_learnable_classes(self, classes):
        """ Map the set of all classes present in the dataset to the set of learnable classes.

        Args:
        classes: np.array

        Returns: np.array

        """

        classes_present = np.unique(classes)

        for class_present in classes_present:
            if class_present in self._all_to_learnable_class_mapping:
                classes[classes == class_present] = self._all_to_learnable_class_mapping[class_present]
            else:
                classes[classes == class_present] = self._masked_class_id

        return classes

    def get_voxel_weights(self, voxelgrid):
        """ Get an np.array of equivalent shape to the input voxelgrid where each item represents the weight of the corresponding voxel.

        Args:
        voxelgrid: np.array

        Returns: np.array

        """

        voxel_weights = np.empty(voxelgrid.shape)

        for class_i, weight in enumerate(self._class_weights):
            voxel_weights[voxelgrid == class_i] = weight

        return voxel_weights

    def get_sample_locations(self, points, classes):
        """ Returns a list of locations at which valid samples can be extracted from the given point cloud and a vector containing the sum of occurrences of each class in these samples.

        Args:
        points: np.array
        classes: np.array

        Returns:
        sample_locations: list[np.array]
        class_counts: np.array

        """

        min_point = np.amin(points, axis=0)
        max_point = np.amax(points, axis=0)

        sample_locations = []
        class_counts = np.zeros(self._num_learnable_classes, dtype=int)

        start_pos = min_point - self._training_sample_spatial_size / 3
        end_pos = max_point - self._training_sample_spatial_size / 3

        for x in np.arange(start_pos[0], end_pos[0], self._sample_step_size):
            for y in np.arange(start_pos[1], end_pos[1], self._sample_step_size):

                sample_location = np.array([x, y, 0])

                points_inliers, classes_inliers = util.cuboid_cutout(
                    points, classes, sample_location, self._training_sample_spatial_size)

                if points_inliers.size == 0:
                    continue

                voxelgrid = util.voxelize(points_inliers, classes_inliers, sample_location,
                                          self._training_sample_spatial_size, self._voxel_size, self._empty_class_id)

                if self.is_sample_valid(voxelgrid, self._validity_thresholds['empty'], self._validity_thresholds['masked']):

                    voxelvector = np.reshape(voxelgrid, [-1])
                    classes_in_sample = np.unique(voxelvector)

                    for class_in_sample in classes_in_sample:
                        class_counts[class_in_sample] += (voxelvector == class_in_sample).sum()

                    sample_locations.append(sample_location)

        return sample_locations, class_counts

    def preprocess_dataset_item(self, item_id, s):
        """ Loads a given item from the dataset into memory, removes invalid classes and finds valid sample locations. 

        Args:
        item_id: string
        s: string, 'train' or 'val'

        Returns: ProcessedItem

        """

        ply_path = self._dataset_data_path + \
            item_id + '/' + item_id + self._uniform_file_suffix_

        points, classes, _ = util.read_ply(ply_path)

        points = points.copy()
        classes = classes.copy()
        classes = self.map_all_to_learnable_classes(classes)

        points -= np.amin(points, axis=0)

        sample_locations, class_counts = self.get_sample_locations(
            points, classes)

        if not sample_locations:
            print 'No sample locations found for item: ' + item_id
            return ProcessedItem([], [], [], [])

        print 'Extracted %s samples from %s' % (len(sample_locations), item_id)

        return ProcessedItem(points, classes, sample_locations, class_counts)

    def compute_class_weights(self, class_counts):
        """ Computes class weights based on the inverse logarithm of a normalized frequency of class occurences.

        Args:
        class_counts: np.array

        Returns: list[float]

        """

        class_counts /= np.sum(class_counts[0:self._empty_class_id])
        class_weights = (1 / np.log(1.2 + class_counts))

        class_weights[self._empty_class_id] = self._special_weights['empty']
        class_weights[self._masked_class_id] = self._special_weights['masked']

        return class_weights.tolist()

    def prepare(self, refresh_cache=False):
        """ Prepares the dataset, loading a cache if present

        Args:
        refresh_cache: bool, whether to preprocess from scratch even when a cache is present

        """

        cache_filepath = self._dataset_data_path + \
            'cached_%s.pckl' % self.cache_hash_

        # Load cache if present
        if not refresh_cache and os.path.isfile(cache_filepath):
            with open(cache_filepath, 'rb') as cache_file:
                self._points, self._classes, self._sample_locations, self._num_samples, self._class_weights = pickle.load(
                    cache_file)
                print 'Loaded dataset from cache: %s' % cache_filepath

            for s in ['train', 'val']:
                print '%s set contains %d samples from %d items' % (s, self.get_num_samples(s), self.get_num_samples(s))

            return

        # Parallelize
        num_parallel_workers = multiprocessing.cpu_count()
        print 'Preparing dataset on %d threads' % num_parallel_workers

        class_counts = np.zeros(self._num_learnable_classes)

        for s in ['train', 'val']:
            print 'Processing set: %s' % s

            item_data = joblib.Parallel(n_jobs=num_parallel_workers)(joblib.delayed(
                self.preprocess_dataset_item)(item_id, s) for item_id in self._item_ids[s])

            for item_i, item_id in enumerate(self._item_ids[s]):

                if not item_data[item_i].sample_locations:
                    self._sample_locations[item_id] = []
                    continue

                if s == 'train':
                    class_counts += item_data[item_i].class_counts

                self._points[item_id] = item_data[item_i].points
                self._classes[item_id] = item_data[item_i].classes
                self._sample_locations[item_id] = item_data[
                    item_i].sample_locations
                self._num_samples[
                    s] += len(item_data[item_i].sample_locations)

        self._class_weights = self.compute_class_weights(class_counts)

        # Save cache
        if refresh_cache or not os.path.isfile(cache_filepath):
            with open(cache_filepath, 'wb') as cache_file:
                pickle.dump([self._points, self._classes, self._sample_locations,
                             self._num_samples, self._class_weights], cache_file, protocol=-1)
                print 'Saved dataset cache to: %s' % cache_filepath

        for s in ['train', 'val']:
            print 'Extracted %d samples from %d items in %s set' % (self.get_num_samples(s), self.get_num_dataset_items(s), s)

        print ''

    def sample_generator(self, s, num_points, augmentations=None):
        """ Generates samples of a given set s each with num_points points, optionally with augmentations

        Args:
        s: string, 'train' or 'val'
        num_points: int
        augmentations: dict

        Yields: np.array, np.array

        """

        samples = []
        for item_id in self._item_ids[s]:
            for sample_location in self._sample_locations[item_id]:
                samples.append((item_id, sample_location))

        while True:

            # Optionally shuffle items in the set
            if augmentations and augmentations['shuffle']:
                random.shuffle(samples)

            tossed_samples = 0
            for sample in samples:

                sample_is_valid = False

                while not sample_is_valid:

                    item_id, sample_location = sample

                    # Make copy of points to be able to transform them without changing the dataset
                    item_points = self._points[item_id].copy()
                    sample_location = sample_location.copy()

                    # Optionally augment with random translation/rotation
                    if augmentations and augmentations['random_rotation']:

                        sample_location = util.random_translate_xy(
                            sample_location, self._sample_step_size)

                        center_of_volume = sample_location + \
                            np.array([self._training_sample_spatial_size[
                                     0], self._training_sample_spatial_size[1], 0]) / 2

                        item_points = util.translate_xy(
                            item_points, -center_of_volume)
                        random_angle = np.random.uniform(0, 2 * np.pi)
                        item_points = util.rotate_around_z(
                            item_points, random_angle)
                        item_points = util.translate_xy(
                            item_points, center_of_volume)

                    # Cutout the sample
                    points, classes = util.cuboid_cutout(item_points, self._classes[
                                                         item_id], sample_location, self._training_sample_spatial_size)

                    # If there are no points in the cloud, skip
                    if points.shape[0] == 0:
                        continue

                    # Voxelize the points to generate ground-truth voxelized output
                    voxelgrid = util.voxelize(
                        points, classes, sample_location, self._training_sample_spatial_size, self._voxel_size, self._empty_class_id)

                    # Only need to test if sample is valid if it was augmented
                    if augmentations:
                        sample_is_valid = self.is_sample_valid(
                            voxelgrid, self._validity_thresholds['empty'], self._validity_thresholds['masked'])
                    else:
                        sample_is_valid = True

                    # Translate points to origin
                    points -= sample_location

                    # Enforce num_points points
                    points = util.random_sample(points, num_points)

                    # Optionally further augment with random jitter and dropout
                    if augmentations:
                        if augmentations['jitter']:
                            points = util.jitter_points(
                                points, sigma=augmentations['jitter']['sigma'], clip=augmentations['jitter']['max'])
                            
                        if augmentations['random_dropout']:
                            points = util.random_dropout_points(
                                points, max_dropout=augmentations['random_dropout']['max'])

                    # Only yield sample if its valid
                    if sample_is_valid:
                        yield points, voxelgrid
                    else:
                        tossed_samples += 1

            print 'Tossed %d samples attempting to generate %d samples of the %s set' % (tossed_samples, len(samples), s)
