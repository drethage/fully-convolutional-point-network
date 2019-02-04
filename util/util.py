""" util.py - Utility functions for training and performing inference with a Fully-Convolutional Point Network """

from __future__ import division
import sys
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import numpy as np
from plyfile import PlyData, PlyElement
import json

""" Saving/loading files from disk """

def load_config(config_filepath):
    """ Load a session configuration from a JSON-formatted file.

    Args:
    config_filepath: string

    Returns: dict

    """

    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print 'No readable config file at path: ' + config_filepath
    else:
        with config_file:
            return json.load(config_file)

def save_config(config_filepath, config):
    """ Save a session configuration to a JSON-formatted file.

    Args:
    config_filepath: string
    config: dict

    """


    try:
        config_file = open(config_filepath, 'w')
    except IOError:
        print 'No readable config file at path: ' + config_filepath
    else:
        json.dump(config, config_file, indent=4, sort_keys=True)

def read_file_to_list(file_path):
        """ Reads a text file line by line.

        Args:
        file_path: string
        
        Returns: list[string]

        """

        with open(file_path) as f:
            return [x.strip() for x in f.readlines() if x.strip() != '']

""" Pretty printing """

class MultiPrint(object):
    """ A class that enables printing to several locations at once. """

    def __init__(self, print_destinations):
        """ Set the print destinations. """

        self.print_destinations = print_destinations

    def write(self, s):
        """ Write the given string s to every print destination. """

        for f in self.print_destinations:
            f.write(s)
            f.flush()

    def flush(self):
        """ Flush the output buffers of each print destination. """

        for f in self.print_destinations:
            f.flush()


def set_print_to_screen_and_file(filepath):
    """ Overrides sys.stdout to point to MultiPrint, enabling printing to stdout and a file on disk in parallel.

    Args:
    filepath: string

    """

    log_file = open(filepath, 'w')
    sys.stdout = MultiPrint([sys.stdout, log_file])

def pretty_print_confusion_matrix(cm, class_labels):
    """ Pretty print a confusion matrix given a set of class_labels.

    Args:
    cm: np.array
    class_labels: list[string]

    """

    print '\nConfusion Matrix: \n'

    columnwidth = 10
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + \
        "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * \
            (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    print "    " + fst_empty_cell,

    for label in class_labels:
        print "%{0}s".format(columnwidth) % label[0:columnwidth],

    print ""

    for i, label in enumerate(class_labels):

        print "    %{0}s".format(columnwidth) % label[0:columnwidth],
        for j in range(len(class_labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            print cell,
        print ""    

def pretty_print_per_class_statistics(class_labels, true_positives, false_negatives, false_positives, ious):
    """ Pretty print a per-class statistics.

    Args:
    class_labels: list[string]
    true_positives: np.array
    false_negatives: np.array
    false_positives: np.array
    ious: np.array

    """

    
    print '\nPer Class Statistics: \n'

    print "    ", 
    columnwidth = 10
    for label in class_labels:
        print "%{0}s".format(columnwidth) % label[0:columnwidth],
    print ''

    print " TP ",
    for tp in true_positives:
        print "%{0}d".format(columnwidth) % tp,
    print ''

    print " FP ",
    for fp in false_positives:
        print "%{0}d".format(columnwidth) % fp,
    print ''

    print " FN ",
    for fn in false_negatives:
        print "%{0}d".format(columnwidth) % fn,
    print ''

    print "IOU ",
    for iou in ious:
        print "%{0}.3f".format(columnwidth) % iou,
    print ''

    avg_iou = np.mean(ious)
    print '\nAverage IOU: %f' % avg_iou


def compute_per_class_statistics(confusion_matrix):
    """ Compute per-class statistics (true-positives, false-negatives, false-positives and intersection-over-unions).

    Args:
    confusion_matrix: np.array
    
    Returns: np.array, np.array, np.array, np.array

    """

    true_positives = confusion_matrix.diagonal()
    false_negatives = np.sum(confusion_matrix, axis=0) - true_positives
    false_positives = np.sum(confusion_matrix, axis=1) - true_positives

    eps = 0.00001
    ious = true_positives / \
        (true_positives + false_positives + false_negatives + eps)

    return true_positives, false_negatives, false_positives, ious


""" Point Clouds and Voxelgrids """

def get_point_cloud_min_max_size(points):
    """ Get the minimum, maximum and size of the space occupied by points.

    Args:
    points: np.array
    
    Returns: np.array, np.array, np.array

    """

    points_min = np.amin(points, axis=0)
    points_max = np.amax(points, axis=0)
    points_size = points_max - points_min

    return points_min, points_max, points_size

def translate_xy(points, t):
    """ Translate a set of points on the x-y plane by t.

    Args:
    points: np.array
    t: np.array
    
    Returns: np.array

    """

    if (t.shape[0] == 2):
        t = np.append(t, [0])

    tiled_t = np.tile(t, (points.shape[0], 1))
    points[0:2] += tiled_t[0:2]
    return points


def rotate_around_z(points, angle):
    """ Rotate a set of points about the z-axis by angle (in radians).

    Args:
    points: np.array
    angle: float
    
    Returns: np.array

    """

    c = np.cos(angle)
    s = np.sin(angle)
    R_z = np.array([[c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]])
    return np.dot(points, R_z)


def jitter_points(points, sigma=0.01, clip=0.025):
    """ Jitter a set of points following a normal distribution with sigma and maximum jitter of clip.

    Args:
    points: np.array
    sigma: float
    clip: float
    
    Returns: np.array

    """

    return points + np.clip(sigma * np.random.randn(points.shape[0], points.shape[1]), -clip, clip)


def random_translate_xy(points, t_range):
    """ Randomly translate a set of points on the x-y plane in range t_range.

    Args:
    points: np.array
    t_range: float
    
    Returns: np.array

    """

    translate = np.random.uniform(-t_range, t_range, points.shape)
    points[0:2] += translate[0:2]
    return points


def random_dropout_points(points, max_dropout=0.9):
    """ Randomly dropout a portion of a set of points by replacing them with the first point to maintain cardinality.

    Args:
    points: np.array
    max_dropout: float
    
    Returns: np.array

    """

    dropout_rate = np.random.random() * max_dropout
    drop_indices = np.where(np.random.random(
        (points.shape[0])) < dropout_rate)[0]
    points[drop_indices, :] = points[0, :]  # set to the first point
    return points

def random_sample(points, num_points):
    """ Randomly sample num_points from a set of points.

    Args:
    points: np.array
    num_points: int
    
    Returns: np.array

    """

    while num_points > points.shape[0]:
        missing_points = np.minimum(
            points.shape[0], num_points - points.shape[0])
        try:
            sample_indices = np.random.choice(
                points.shape[0], missing_points, replace=False)
        except:
            print 'had greater'
            print points.shape
            print missing_points
        points = np.concatenate((points, points[sample_indices, :]), axis=0)

    if num_points < points.shape[0]:
        print 'Removing %d points from sample' % (points.shape[0] - num_points)

        try:
            sample_indices = np.random.choice(
                points.shape[0], num_points, replace=False)
        except:
            print points.shape
            print num_points
        points = points[sample_indices, :]

    return points

def voxelize(points, labels, origin, voxelgrid_size, voxel_size, unoccupied_class):
    """ Voxelize all points lying between origin and origin+voxelgrid_size at a voxel size of voxel_size.
        Each voxel holds the label of the point occupying its space or unoccupied_class if no point is present.

    Args:
    points: np.array
    labels: np.array
    origin: np.array
    voxelgrid_size: np.array
    voxel_size: float
    unoccupied_class: int
    
    Returns: np.array

    """

    num_voxels_per_dim = np.ceil(voxelgrid_size / voxel_size).astype(int)
    voxelgrid = np.ones(num_voxels_per_dim, dtype=int) * unoccupied_class

    points_at_origin = points - origin

    labels_in_cloud = np.unique(labels)

    for label in labels_in_cloud:

        if label == unoccupied_class:
            continue

        points_with_label = points_at_origin[labels == label]
        point_indices_in_grid = np.floor(
            points_with_label / voxel_size).astype(int)
        voxelgrid[point_indices_in_grid[:, 0], point_indices_in_grid[
            :, 1], point_indices_in_grid[:, 2]] = label

    return voxelgrid

def voxelgrid_to_point_cloud(voxelgrid, voxel_size):
    """ Converts a voxelgrid to a point cloud.

    Args:
    voxelgrid: np.array
    voxel_size: float
    
    Returns: np.array, np.array

    """

    points = []
    labels = []

    for i in range(voxelgrid.shape[0]):
        for j in range(voxelgrid.shape[1]):
            for k in range(voxelgrid.shape[2]):
                points.append(
                    np.array([i * voxel_size, j * voxel_size, k * voxel_size]))
                labels.append(voxelgrid[i, j, k])

    # Translate all points by half a voxel size
    points = np.array(points) + \
        np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
    labels = np.array(labels)

    return points, labels

def cuboid_cutout(points, labels, location, size):
    """ Cuts a cuboid shaped region out of points at location.

    Args:
    points: np.array
    labels: np.array
    location: np.array
    size: np.array
    
    Returns: np.array, np.array

    """

    CUTOUT_MARGIN = 0.025

    min_xyz = np.array([location[0] + CUTOUT_MARGIN,
                        location[1] + CUTOUT_MARGIN, CUTOUT_MARGIN])
    max_xyz = np.array([location[0] + size[0] - CUTOUT_MARGIN,
                        location[1] + size[1] - CUTOUT_MARGIN, size[2] - CUTOUT_MARGIN])

    inliers = np.all(np.logical_and(
        min_xyz < points, points < max_xyz), axis=1)

    if labels is None:
        return points[inliers]

    return points[inliers], labels[inliers]


def remove_class(points, classes, remove_class):
    """ Filters out all points labeled with remove_class.

    Args:
    points: np.array
    classes: np.array
    remove_class: int
    
    Returns: np.array, np.array

    """

    annotated_point_indices = np.logical_not(classes == remove_class)
    return points[annotated_point_indices], classes[annotated_point_indices]

def get_uniformly_spaced_point_grid(min_xyz, max_xyz, distance_between_points, batch_size=0):
    """ Generates a uniformly spaced point grid starting at min_xyz and ending at max_xyz with distance_between_points between each point.

    Args:
    min_xyz: np.array
    max_xyz: np.array
    distance_between_points: float
    batch_size: int
    
    Returns: np.array

    """

    points = []
    for x in np.arange(min_xyz[0] + distance_between_points / 2, max_xyz[0], distance_between_points):
        for y in np.arange(min_xyz[1] + distance_between_points / 2, max_xyz[1], distance_between_points):
            for z in np.arange(min_xyz[2] + distance_between_points / 2, max_xyz[2], distance_between_points):
                points.append([x, y, z])

    if batch_size == 0:
        points = np.array(points, dtype=np.float32)
    else:
        num_points = len(points)
        points = tf.convert_to_tensor(points)
        points = tf.reshape(points, [-1])
        points = tf.tile(points, [batch_size])
        points = tf.reshape(points, [batch_size, num_points, 3])

    return points

""" PLYs """

def read_ply(filename):
    """ Reads a PLY file from disk.

    Args:
    filename: string
    
    Returns: np.array, np.array, np.array

    """

    file = open(filename, 'rb')
    plydata = PlyData.read(file)

    points = np.stack((plydata['vertex']['x'], plydata['vertex'][
                      'y'], plydata['vertex']['z'])).transpose()

    try:
        labels = plydata['vertex']['label']
    except:
        labels = np.array([])

    try:
        faces = np.array(plydata['face'].data['vertex_indices'].tolist())
    except:
        faces = np.array([])

    file.close()

    return points, labels, faces

def write_ply(filename, points, faces=None, labels=None, colormap=None):
    """ Writes a set of points, optionally with faces, labels and a colormap as a PLY file to disk.

    Args:
    filename: string
    points: np.array
    faces: np.array
    labels: np.array
    colormap: np.array

    """

    with open(filename, 'w') as file:

        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n' % points.shape[0])
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if labels is not None:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')
            file.write('property ushort label\n')

        if faces is not None:
            file.write('element face %d\n' % faces.shape[0])
            file.write('property list uchar int vertex_indices\n')

        file.write('end_header\n')

        if labels is None or colormap is None:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f\n' % (points[point_i, 0], points[
                           point_i, 1], points[point_i, 2]))
        else:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f %d %d %d %d %d\n' % (points[point_i, 0], points[point_i, 1], points[point_i, 2], colormap[labels[point_i]][0], colormap[labels[point_i]][1], colormap[labels[point_i]][2], 255, labels[point_i]))

        if faces is not None:
            for face_i in range(faces.shape[0]):
                file.write('3 %d %d %d\n' % (
                    faces[face_i, 0], faces[face_i, 1], faces[face_i, 2]))

""" Tensorflow utility functions """

def conv3d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 3D convolution transpose with non-linear operation.

    Args:
      inputs: 5-D tf.tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: list[int]
      scope: string
      stride: list[int]
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns: tf.tensor

    Note: conv3d(conv3d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a

    """

    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride

        # from slim.convolution3d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= kernel_size
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        depth = inputs.get_shape()[1].value
        height = inputs.get_shape()[2].value
        width = inputs.get_shape()[3].value
        out_depth = get_deconv_dim(depth, stride_d, kernel_d, padding)
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_depth,
                        out_height, out_width, num_output_channels]

        outputs = tf.nn.conv3d_transpose(inputs, kernel, output_shape,
                                         [1, stride_d, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def tf_safe_div(numerator, denominator):
    """ Computes a safe division which returns 0 if the denominator is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.

    Args:
      numerator: tf.tensor
      denominator: tf.tensor

    Returns: tf.tensor

    """

    return array_ops.where(
        math_ops.greater(denominator, 0),
        math_ops.div(numerator,
                     array_ops.where(
                         math_ops.equal(denominator, 0),
                         array_ops.ones_like(denominator), denominator)),
        array_ops.zeros_like(numerator))

def get_learning_rate(batch, batch_size, init_learning_rate, decay_rate, decay_steps, min_learning_rate=0.00001):
    """ Get exponentially decaying learning rate clipped at min_learning_rate.

    Args:
    batch: int
    batch_size: int
    init_learning_rate: float
    decay_rate: float
    decay_steps: int
    min_learning_rate: float

    Returns: float

    """
    learning_rate = tf.train.exponential_decay(
        init_learning_rate, batch * batch_size, decay_steps, decay_rate, staircase=True)
    return tf.maximum(learning_rate, min_learning_rate)


def get_batch_normalization_decay(batch, batch_size, init_decay, decay_rate, decay_steps, max_decay=0.99):
    """ Get batch normalization decay starting from init_decay clipped at max_momentum.

    Args:
    batch: int
    batch_size: int
    init_decay: float
    decay_rate: float
    decay_steps: int
    max_decay: float

    Returns: float

    """
    bn_momentum = tf.train.exponential_decay(
        init_decay, batch * batch_size, decay_steps, decay_rate, staircase=True)
    return tf.minimum(max_decay, 1 - bn_momentum)

# From Charles R. Qi
# (github.com/charlesq34/pointnet/blob/master/utils/tf_util.py)

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    # For support of GAN
    #bn_decay = bn_decay if bn_decay is not None else 0.9
    # return tf.contrib.layers.batch_norm(inputs,
    #                                    center=True, scale=True,
    #                                    is_training=is_training, decay=bn_decay,updates_collections=None,
    #                                    scope=scope)
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(
            inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
        # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = tf.cond(is_training,
                                   lambda: ema.apply([batch_mean, batch_var]),
                                   lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(
            inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.

    Args:
      inputs: 5-D tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: a list of 3 ints
      scope: string
      stride: a list of 3 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(
                              inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs
