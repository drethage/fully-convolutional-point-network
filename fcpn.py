""" fcpn.py - Defines the FCPN class, representing a Fully-Convolutional Point Network. """

import sys
sys.path.append('tf_grouping')
import tensorflow as tf
import numpy as np
import util
import math
import tf_grouping

class FCPN(object):


    def __init__(self, config):
        """ Creates an FCPN object.

            Args:
            config: dict

        """

        # Output voxls have a size of 5cm per dimension
        self._output_voxel_size = 0.05
        
        # FCPN features 3 levels of abstraction
        self._abstraction_levels = 3
        
        self._config = config

    @staticmethod
    def print_layer_weights():
        """ Prints the name, shape and size of each layer in model containing weights. """

        print '\nLayers in model: (Name - Shape - # weights)'
        for variable in tf.trainable_variables():
            if not variable.name.endswith('weights:0'):
                continue
            print variable.name + ' - ' + str(variable.get_shape()) + ' - ' + str(np.prod(variable.get_shape().as_list()))

    def get_num_parameters(self):
        """ Get the total number of parameters (weights) in the model.

            Returns: int

        """

        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]).astype(np.int)

    def print_num_parameters(self):
        """ Print the total number of parameters (weights) in the model. """
            
        print '\nParameters in model: %d' % self.get_num_parameters()

    def get_max_centroid_spacing(self):
        """ Get the maximum distance between the centers of two cells in the model.

            Returns: float

        """
        return FCPN.get_centroid_spacing(self._config['model']['pointnet']['spacing'], self._abstraction_levels)

    @staticmethod
    def get_centroid_spacing(initial_spacing, abstraction_layer):
        """ Get the distance between the centers of two cells in the given abstraction_layer.

            Args:
            initial_spacing: float
            abstraction_layer: int

            Returns: float

        """
        return initial_spacing * 2 ** (abstraction_layer - 1)

    @staticmethod
    def get_pointnet_radius(pointnet_spacing):
        """ Computes a radius given a pointnet_spacing so that PointNet's fully cover the 3D space of the receptive field.

            Args:
            pointnet_spacing: float

            Returns: float

        """
        return math.sqrt(3 * (pointnet_spacing / 2) ** 2)

    @staticmethod
    def get_feature_volume_shape(input_volume_size, pointnet_spacing, abstraction_layer):
        """ Get the shape of a intermediate feature volume given an input_volume_size, pointnet_spacing and abstraction_layer.

            Args:
            input_volume_size: np.array
            pointnet_spacing: float
            abstraction_layer: int

            Returns: np.array

        """
        return np.round(np.array(input_volume_size) / FCPN.get_centroid_spacing(pointnet_spacing, abstraction_layer)).astype(int)

    @staticmethod
    def get_output_volume_shape(input_volume_size, output_spacing):
        """ Get the shape of the output volume.

            Args:
            input_volume_size: np.array
            output_spacing: float

            Returns: np.array

        """
        return np.round(np.array(input_volume_size) / output_spacing).astype(int)

    @staticmethod
    def get_spatial_pool_weighting(sphere_radius, top_level_centroid_locations):
        """ Compute a spatial weighting for every cell.

            Args:
            sphere_radius: float, the weight of neighboring cells will be greatest on this sphere's surface
            top_level_centroid_locations: tf.tensor, locations of cells in metric space

            Returns: tf.tensor

        """

        top_level_centroid_locations_repeated = tf.tile(tf.expand_dims(top_level_centroid_locations, axis=1), [1, top_level_centroid_locations.get_shape()[1].value, 1, 1]) #row-wise repeated sample locations
        difference = tf.subtract(top_level_centroid_locations_repeated, tf.transpose(top_level_centroid_locations_repeated, [0, 2, 1, 3]))
        # Euclidean distance from every centroid to every other centroid
        distance = tf.norm(difference, axis=3, ord=2, keepdims=True)
        # Clipped distance in [sphere_radius - 1, sphere_radius + 1] range
        clipped_distance = tf.clip_by_value(distance, sphere_radius - 1, sphere_radius + 1)
        # Neighboring voxels weighting based on (cos(3(x-1.5)) + 1) / 2, max weighting on voxels sphere_radius away from a given voxel
        cos_distance_to_sphere_surface = (tf.cos(3 * (clipped_distance - sphere_radius)) + 1) / 2
        # Normalized weighting
        return cos_distance_to_sphere_surface / tf.reduce_sum(cos_distance_to_sphere_surface, axis=2, keepdims=True)

    def get_output_voxel_spacing(self):
        """ Get the distance between the centers of two voxels in the predicted output.

            Returns: float

        """

        return self._output_voxel_size

    def get_pointnet_locations(self, batch_size=0):
        """ Get the locations of PointNets in the model.

            Args:
            batch_size: int

            Returns:
            pointnet_locations: np.array if batch_size == 0 else tf.tensor

        """
        return util.get_uniformly_spaced_point_grid(self.min_xyz_, self.max_xyz_, self._config['model']['pointnet']['spacing'], batch_size)

    @staticmethod
    def radius_search_and_group(centroids_xyz, radius, num_neighbors, points_xyz, points_features):
        """ Perform radius search and grouping of points_xyz around each centroids_xyz

            Args:
            centroids_xyz: tf.tensor, xyz locations of centroids
            radius: float, radius of spherical region around centroid
            num_neighbors: int, number of neighbors to include in grouping per centroid
            points_xyz: tf.tensor, xyz locations of points
            points_features: tf.tensor, features of points

            Returns tf.tensor, grouped points and point features

        """
        
        # Radius search around each centroid, returning num_neighbors point indices within radius of centroid
        point_indices, _ = tf_grouping.query_ball_point(radius, num_neighbors, points_xyz, centroids_xyz)
        
        # Group neighboring points (and corresponding point features) together
        grouped_points_xyz = tf_grouping.group_point(points_xyz, point_indices) # (batch_size, num_centroids, num_neighbors, 3)
        grouped_points_features = tf_grouping.group_point(points_features, point_indices) # (batch_size, num_centroids, num_neighbors, num_features)
        
        # Normalize points' xyz locations in local region by subtracting the xyz of the centroid of that region
        grouped_points_xyz -= tf.tile(tf.expand_dims(centroids_xyz, 2), [1,1, num_neighbors ,1])
        grouped_points_xyz_and_features = tf.concat([grouped_points_xyz, grouped_points_features], axis=-1) # (batch_size, num_centroids, num_neighbors, 3+num_features)

        return grouped_points_xyz_and_features

    def build_model(self, batch_size, spatial_size, points_xyz, points_features, is_training, num_class, batch_normalization_decay=None):
        """ Build a Fully-Convolutional Point Network.

            Args:
            batch_size: int
            spatial_size: np.array
            points_xyz: tf.placeholder
            points_features: tf.placeholder
            is_training: tf.placeholder
            num_class: int
            batch_normalization_decay: float

            Returns: tf.tensor

        """

        self.min_xyz_ = np.array([0, 0, 0])
        self.max_xyz_ = spatial_size
        self.use_batch_normalization_ = self._config['training']['optimizer']['batch_normalization'] != False

        pointnet_locations = util.get_uniformly_spaced_point_grid(self.min_xyz_, self.max_xyz_, self._config['model']['pointnet']['spacing'], batch_size)
        top_level_centroid_locations = util.get_uniformly_spaced_point_grid(self.min_xyz_, self.max_xyz_, self.get_centroid_spacing(self._config['model']['pointnet']['spacing'], self._abstraction_levels), batch_size)

        with tf.variable_scope("abstraction"):

            with tf.variable_scope("points_to_15cm"):

                with tf.variable_scope("simplified_pointnet"):

                    with tf.device('/gpu:' + str(self._config['training']['gpu']['id'])):
                        # Radius search and Grouping
                        grouped_points_xyz_and_features = self.radius_search_and_group(pointnet_locations, self.get_pointnet_radius(self._config['model']['pointnet']['spacing']), self._config['model']['pointnet']['neighbors'], points_xyz, points_features)

                    # 3x 1x1 Convolutions
                    features = util.conv2d(grouped_points_xyz_and_features, self._config['model']['filters']['abstraction']['points_to_15cm'][0], [1, 1], padding='VALID', stride=[1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1_conv_1', bn_decay=batch_normalization_decay)
                    features = util.conv2d(features, self._config['model']['filters']['abstraction']['points_to_15cm'][1], [1, 1], padding='VALID', stride=[1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1_conv_2', bn_decay=batch_normalization_decay)

                    # Max-pooling for permutation invariance
                    features = tf.reduce_max(features, axis=[2], keepdims=True)
                    features = tf.squeeze(features, [2])

                    num_dims = self.get_feature_volume_shape(spatial_size, self._config['model']['pointnet']['spacing'], 1)
                    features = tf.reshape(features, [batch_size, num_dims[0], num_dims[1], num_dims[2], features.get_shape().as_list()[-1]])

                with tf.variable_scope("skip_15cm"):
                
                    skip_15cm = util.conv3d(features, self._config['model']['filters']['skip']['15cm'], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv', bn_decay=batch_normalization_decay)

                with tf.variable_scope("skip_45cm"):
                    
                    padded = tf.pad(features, [[0,0], [1,1], [1,1], [1,1], [0,0]], "SYMMETRIC")
                    skip_45cm = util.conv3d(padded, self._config['model']['filters']['skip']['45cm'], [3, 3, 3], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='3x3x3_conv', bn_decay=batch_normalization_decay)

            with tf.variable_scope("15cm_to_30cm"):

                with tf.variable_scope("3d_convolution"):

                    features = util.conv3d(features, self._config['model']['filters']['abstraction']['15cm_to_30cm'][0], [2, 2, 2], padding='VALID', stride=[2, 2, 2], bn=self.use_batch_normalization_, is_training=is_training, scope='2x2x2_conv', bn_decay=batch_normalization_decay)
                    features = util.conv3d(features, self._config['model']['filters']['abstraction']['15cm_to_30cm'][1], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1_conv_1', bn_decay=batch_normalization_decay)

                with tf.variable_scope("skip_30cm"):

                    skip_30cm = util.conv3d(features, self._config['model']['filters']['skip']['30cm'], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv', bn_decay=batch_normalization_decay)

                with tf.variable_scope("skip_90cm"):

                    padded = tf.pad(features, [[0,0], [1,1], [1,1], [1,1], [0,0]], "SYMMETRIC")
                    skip_90cm = util.conv3d(padded, self._config['model']['filters']['skip']['90cm'], [3, 3, 3], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='3x3x3_conv', bn_decay=batch_normalization_decay)

            with tf.variable_scope("30cm_to_60cm"):

                with tf.variable_scope("3d_convolution"):

                    features = util.conv3d(features, self._config['model']['filters']['abstraction']['30cm_to_60cm'][0], [2, 2, 2], padding='VALID', stride=[2, 2, 2], bn=self.use_batch_normalization_, is_training=is_training, scope='2x2x2_conv', bn_decay=batch_normalization_decay)
                    features = util.conv3d(features, self._config['model']['filters']['abstraction']['30cm_to_60cm'][1], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_1', bn_decay=batch_normalization_decay)
                    
                with tf.variable_scope("skip_60cm"):

                    skip_60cm = util.conv3d(features, self._config['model']['filters']['skip']['60cm'], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_1', bn_decay=batch_normalization_decay)

                with tf.variable_scope("skip_180cm"):

                    padded = tf.pad(features, [[0,0], [1,1], [1,1], [1,1], [0,0]], "SYMMETRIC")
                    skip_180cm = util.conv3d(padded, self._config['model']['filters']['skip']['180cm'], [3, 3, 3], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='3x3x3_conv', bn_decay=batch_normalization_decay)

        with tf.variable_scope("spatial_pool"):

            num_cells_in_current_layer = self.get_feature_volume_shape(spatial_size, self._config['model']['pointnet']['spacing'], 3)

            with tf.variable_scope("reshape_and_repeat"):

                # Reshape and repeat feature volume to apply weighted spatial pooling
                features = tf.reshape(features, [batch_size, top_level_centroid_locations.get_shape()[1].value, self._config['model']['filters']['abstraction']['30cm_to_60cm'][-1]])
                features = tf.tile(tf.expand_dims(features, axis=1), [1, top_level_centroid_locations.get_shape()[1].value, 1, 1])

            with tf.variable_scope("pool"):
             
                spatial_pooling_weights = self.get_spatial_pool_weighting(self._config['model']['spatial_pool_radius'], top_level_centroid_locations)
                skip_spatial_pool = features * spatial_pooling_weights
                skip_spatial_pool = tf.reduce_sum(skip_spatial_pool, axis=2)
                skip_spatial_pool = tf.reshape(skip_spatial_pool, [batch_size, num_cells_in_current_layer[0], num_cells_in_current_layer[1], num_cells_in_current_layer[2], self._config['model']['filters']['abstraction']['30cm_to_60cm'][-1]])
                
            with tf.variable_scope("skip_spatial_pool"):
                skip_spatial_pool = util.conv3d(skip_spatial_pool, self._config['model']['filters']['skip']['spatial_pool'], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv', bn_decay=batch_normalization_decay)

        with tf.variable_scope("upsampling"):
            
            with tf.variable_scope("60cm_to_30cm"):

                features = tf.concat([skip_60cm, skip_180cm, skip_spatial_pool], axis=4)

                features = util.conv3d_transpose(features, self._config['model']['filters']['upsampling']['60cm_to_30cm'][0], [2, 2, 2], padding='VALID', stride=[2, 2, 2], bn=self.use_batch_normalization_, is_training=is_training, scope='2x2x2_deconv', bn_decay=batch_normalization_decay)
                features = util.conv3d(features, self._config['model']['filters']['upsampling']['60cm_to_30cm'][1], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_1', bn_decay=batch_normalization_decay)
                features = util.conv3d(features, self._config['model']['filters']['upsampling']['60cm_to_30cm'][2], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_2', bn_decay=batch_normalization_decay)
                
            with tf.variable_scope("30cm_to_15cm"):

                features = tf.concat([features, skip_30cm, skip_90cm], axis=4)
                features = util.conv3d_transpose(features, self._config['model']['filters']['upsampling']['30cm_to_15cm'][0], [2, 2, 2], padding='VALID', stride=[2, 2, 2], bn=self.use_batch_normalization_, is_training=is_training, scope='2x2x2_deconv', bn_decay=batch_normalization_decay)
                features = util.conv3d(features, self._config['model']['filters']['upsampling']['30cm_to_15cm'][1], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_1', bn_decay=batch_normalization_decay)
                features = util.dropout(features, keep_prob=0.5, is_training=is_training, scope='dropout')

                features = tf.concat([features, skip_45cm], axis=4)
                features = util.conv3d(features, self._config['model']['filters']['upsampling']['30cm_to_15cm'][2], [1, 1, 1], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='1x1x1_conv_3', bn_decay=batch_normalization_decay)

            with tf.variable_scope("15cm_to_5cm"):

                features = tf.concat([features, skip_15cm], axis=4)
                features = util.dropout(features, keep_prob=0.5, is_training=is_training, scope='dropout')

                upsample_factor = int(math.ceil(self._config['model']['pointnet']['spacing'] / self._output_voxel_size))
                features = util.conv3d_transpose(features, self._config['model']['filters']['upsampling']['15cm_to_5cm'][0], [upsample_factor, upsample_factor, upsample_factor], padding='VALID', stride=[upsample_factor, upsample_factor, upsample_factor], bn=self.use_batch_normalization_, is_training=is_training, scope='final_deconv', bn_decay=batch_normalization_decay)
                features = tf.pad(features, [[0,0], [1,1], [1,1], [1,1], [0,0]], "SYMMETRIC")
                output = util.conv3d(features, num_class, [3, 3, 3], padding='VALID', stride=[1, 1, 1], bn=self.use_batch_normalization_, is_training=is_training, scope='final_conv', bn_decay=batch_normalization_decay, activation_fn=None)

                num_output_elements = np.prod(self.get_output_volume_shape(spatial_size, self._output_voxel_size))
                output = tf.reshape(output, [batch_size, num_output_elements, num_class])

        return output

    def get_loss(self, predictions, labels, weights):
        """ Get an op to compute crossentropy loss given a set of predictions and a set of labels weighted by weights.

            Args:
            predictions: tf.tensor
            labels: tf.tensor
            weights: tf.tensor

            Note: predictions.shape == labels.shape == weights.shape

            Returns: tf.float32

        """

        crossentropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predictions, weights=weights)
        loss = tf.reduce_mean(crossentropy_loss, name='mean_crossentropy_loss')
        tf.summary.scalar('mean_crossentropy_loss', loss)
        return loss

    def get_confusion_matrix_ops(self, predictions, labels, num_classes, unoccupied_class):
        """ Get ops for maintaining a confusion matrix during training.

            Args:
            predictions: tf.tensor
            labels: tf.tensor
            num_classes: int
            unoccupied_class: int, id of unoccupied class

            Returns: tf.tensor, tf.tensor, tf.tensor

        """

        labels = tf.reshape(labels, [-1])
        predictions_argmax = tf.reshape(tf.argmax(predictions, axis=2), [-1])
        batch_confusion = tf.confusion_matrix(labels, predictions_argmax, num_classes=num_classes, name='batch_confusion')
        confusion = tf.Variable( tf.zeros([num_classes, num_classes], dtype=tf.int32 ), name='confusion' )
        confusion_update = confusion.assign( confusion + batch_confusion )
        confusion_clear = confusion.assign(tf.zeros([num_classes, num_classes], dtype=tf.int32))

        return confusion, confusion_update, confusion_clear
