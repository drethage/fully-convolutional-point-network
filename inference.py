""" inference.py - Implements functions for evaluating a model and using it to predict on new data. """

from __future__ import division
import sys
sys.path.append('util')
import os
import time
import util
import numpy as np
import tensorflow as tf
import fcpn
import data
import glob

def voxelgrid_predictions_to_point_cloud(voxelgrid, voxel_size, points, unoccupied_class):
    """ Maps predictions made by the model onto points. Consider all voxels adjacent to a point.

        Args:
        voxelgrid: np.array
        voxel_size: float
        points: np.array
        unoccupied_class: int, id of unoccupied class

        Returns: list[int]

    """

    predicted_labels = []
    for point in points:

        point_index_in_predictions = (point / voxel_size).astype(dtype=int)
        
        predicted_label = None
        for x_offset in [0, -1, 1]:
            for y_offset in [0, -1, 1]:
                for z_offset in [0, -1, 1]:
                    predicted_label = voxelgrid[point_index_in_predictions[0] + x_offset, point_index_in_predictions[1] + y_offset, point_index_in_predictions[2] + z_offset]

                    if predicted_label != unoccupied_class:
                        break

                if predicted_label != unoccupied_class:
                        break

            if predicted_label != unoccupied_class:
                        break

        predicted_labels.append(predicted_label)

    return predicted_labels

def setup_model(model, receptive_field_size, num_input_points, pointnet_spacing, num_learnable_classes, checkpoint_path, device):
    """ Sets up the model, restoring weights from a checkpoint at checkpoint_path.

        Args:
        receptive_field_size: np.array
        num_input_points: int
        pointnet_spacing: float
        num_learnable_classes: int
        checkpoint_path: string
        device: string

        Returns: tf.session, dict, tf.tensor, np.array, np.array

    """

    input_volume_origin = np.array([0,0,0])
    num_pointnets = np.prod(model.get_feature_volume_shape(receptive_field_size, pointnet_spacing, 1))

    # Necessary to allow tf to place ops on CPU when no GPU implementation exists
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    # Create a session
    sess = tf.Session(config=tf_config)

    with sess.as_default():
        with tf.device('/' + device + ':0'):

            is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')
            points_xyz_pl = tf.placeholder(tf.float32, shape=(1, num_input_points+num_pointnets, 3), name='points_xyz_pl')
            points_features_pl = tf.placeholder(tf.float32, shape=(1, num_input_points+num_pointnets, 1), name='points_features_pl')
            placeholders = {'is_training_pl': is_training_pl, 'points_xyz_pl': points_xyz_pl, 'points_features_pl': points_features_pl}
            pred_op = model.build_model(1, receptive_field_size, points_xyz_pl, points_features_pl, is_training_pl, num_learnable_classes)

            tf.train.Saver().restore(sess, checkpoint_path)
            model.print_num_parameters()

            pointnet_locations = model.get_pointnet_locations()
            point_features = np.ones(num_input_points)
            pointnet_features = np.zeros(pointnet_locations.shape[0])
            constant_features = np.expand_dims(np.expand_dims(np.concatenate([point_features, pointnet_features]), axis=1), axis=0)

            return sess, placeholders, pred_op, pointnet_locations, constant_features

def get_validation_set_item_ids(dataset_metadata_path):
    """ Loads the items from the dataset belonging to the validation set.

        Args:
        dataset_metadata_path: string

        Returns: list[string]

    """

    with open(os.path.join(dataset_metadata_path, 'validation_split.txt')) as f:
        return [x.strip() for x in f.readlines() if x.strip() != '']

def get_latest_checkpoint_path(session_dir):
    """ Returns the path of the most recent checkpoint in session_dir.

        Args:
        session_dir: string

        Returns: string

    """

    checkpoints = glob.glob(session_dir+'model.ckpt-*.meta')

    if not checkpoints:
        return ''
    checkpoints.sort(key=lambda f: int(filter(str.isdigit, f)))
    return checkpoints[-1][:-5]

def run_model(sess, placeholders, pred_op, points, pointnet_locations, constant_features):
    """ Passes points through the model.

        Args:
        sess: tf.session
        placeholders: dict
        pred_op: tf.tensor
        points: np.array
        pointnet_locations: np.array
        constant_features: np.array

        Returns: np.array

    """

    points_and_pointnet_locations = np.expand_dims(np.concatenate([points, pointnet_locations], axis=0), axis=0)

    start_time = time.time()
    predictions = sess.run(pred_op, feed_dict={
        placeholders['is_training_pl']: False,
        placeholders['points_xyz_pl']: points_and_pointnet_locations,
        placeholders['points_features_pl']: constant_features
    })
    end_time = time.time()
    print 'Prediction took: %ds' % (end_time - start_time)

    return predictions

def predict(config_path, input_path, device, colors_path):
    """ Predicts semantics of an unseen input to a trained model.

        Args:
        config_path: string
        input_path: string
        device: string
        colors_path: string

    """

    config = util.load_config(config_path)
    print 'Loaded configuration from: %s' % config_path

    session_dir = config_path[:config_path.rfind('/')+1]

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:
        print 'Error: No checkpoint found in same directory as configuration file.'
        return
    
    predict_path = os.path.join(session_dir, 'predict')
    if not os.path.exists(predict_path): os.mkdir(predict_path)

    model = fcpn.FCPN(config)

    points, _, faces = util.read_ply(input_path)
    points_min, points_max, points_size = util.get_point_cloud_min_max_size(points)

    print 'Size: %f, %f, %f, # Points: %d' % (points_size[0], points_size[1], points_size[2], points.shape[0])

    receptive_field_size = np.ceil(points_size / model.get_max_centroid_spacing()) * model.get_max_centroid_spacing()
    print 'Model Receptive Field Size: %f, %f, %f' % (receptive_field_size[0], receptive_field_size[1], receptive_field_size[2])

    sess, placeholders, pred_op, pointnet_locations, constant_features = setup_model(model, receptive_field_size, points.shape[0], config['model']['pointnet']['spacing'], config['dataset']['num_learnable_classes'], checkpoint_path, device)

    if not colors_path: colors_path = 'util/colors.txt'
    with open(colors_path) as f:
        colors = np.array([[int(c) for c in line.strip().split(' ')] for line in f.readlines()])

    # Translate input point cloud to be centered at origin
    translate_to_origin = np.tile(points_min, (points.shape[0], 1))
    translate_to_padded_origin = np.tile(np.array([0, 0, 0]), (points.shape[0], 1))

    points -= translate_to_origin
    points += translate_to_padded_origin

    predictions = run_model(sess, placeholders, pred_op, points, pointnet_locations, constant_features)
    predictions = predictions[0,:,:config['dataset']['empty_class_id']+1]
    predictions = np.argmax(predictions, axis=1)
    predictions = np.reshape(predictions, np.round(receptive_field_size / model.get_output_voxel_spacing()).astype(np.int32))
    predicted_labels = voxelgrid_predictions_to_point_cloud(predictions, model.get_output_voxel_spacing(), points, config['dataset']['empty_class_id'])

    points += translate_to_origin
    points -= translate_to_padded_origin

    predicted_filepath = os.path.join(predict_path, input_path[input_path.rfind('/')+1:-4] + '.predicted.ply')
    util.write_ply(predicted_filepath, points, faces, predicted_labels, colormap=colors)

def evaluate(config_path, device):
    """ Evaluates a trained model associated with the configuration file at config_path.

        Args:
        config_path: string
        device: string

    """

    config = util.load_config(config_path)
    print 'Loaded configuration from: %s' % config_path

    session_dir = config_path[:config_path.rfind('/')+1]

    model = fcpn.FCPN(config)
    dataset = data.Dataset(config)

    sample_ids = get_validation_set_item_ids(dataset.get_dataset_metadata_path())
    points_list = []
    faces_list = []
    labels_list = []
    max_size = np.array([0,0,0])
    max_points_count = 0

    print 'Loading test set.'
    for sample_id in sample_ids:
        ply_path = os.path.join(dataset.get_dataset_data_path(), sample_id, sample_id + config['dataset']["original_file_suffix"])
        points, labels, faces = util.read_ply(ply_path)
        points_list.append(points)
        labels_list.append(labels)
        faces_list.append(faces)
        points_min, points_max, points_size = util.get_point_cloud_min_max_size(points)
        max_size = np.maximum(max_size, points_size)
        max_points_count = np.maximum(max_points_count, points.shape[0])

    print 'Max Input Size: %f, %f, %f' % (max_size[0], max_size[1], max_size[2])
    print 'Max Input Points: %d' % max_points_count
    receptive_field_size = np.ceil(max_size / model.get_max_centroid_spacing()) * model.get_max_centroid_spacing()
    print 'Model Receptive Field Size: %f, %f, %f' % (receptive_field_size[0], receptive_field_size[1], receptive_field_size[2])

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:
        print 'Error: No checkpoint found in same directory as configuration file.'
        return

    evaluation_path = os.path.join(session_dir, 'evaluation')
    if not os.path.exists(evaluation_path): os.mkdir(evaluation_path)

    sess, placeholders, pred_op, pointnet_locations, constant_features = setup_model(model, receptive_field_size, max_points_count, config['model']['pointnet']['spacing'], config['dataset']['num_learnable_classes'], checkpoint_path, device)    
    confusion_matrix = np.zeros((dataset.get_num_learnable_classes(), dataset.get_num_learnable_classes()), dtype=int) # rows = actual, columns = predicted

    for input_i, sample_id in enumerate(sample_ids):

        print sample_id

        points = points_list[input_i]
        labels = labels_list[input_i]
        faces = faces_list[input_i]
        labels_remapped = dataset.map_all_to_learnable_classes(labels)

        points_min, points_max, points_size = util.get_point_cloud_min_max_size(points)

        # Translate input point cloud to be centered at origin
        translate_to_origin = np.tile(points_min, (points.shape[0], 1))
        translate_to_padded_origin = np.tile(np.array([0, 0, 0]), (points.shape[0], 1))

        points -= translate_to_origin
        points += translate_to_padded_origin

        resampled_points = util.random_sample(points, max_points_count)
        predictions = run_model(sess, placeholders, pred_op, resampled_points, pointnet_locations, constant_features)
        predictions = predictions[0,:,:config['dataset']['empty_class_id']+1]
        predictions = np.argmax(predictions, axis=1)
        predictions = np.reshape(predictions, np.round(receptive_field_size / model.get_output_voxel_spacing()).astype(np.int32))
        predicted_labels = voxelgrid_predictions_to_point_cloud(predictions, model.get_output_voxel_spacing(), points, dataset.get_empty_class())

        points += translate_to_origin
        points -= translate_to_padded_origin

        for point_i in range(points.shape[0]):
            confusion_matrix[labels_remapped[point_i], predicted_labels[point_i]] += 1

        predicted_filepath = os.path.join(evaluation_path, sample_id + '.predicted.ply')
        util.write_ply(predicted_filepath, points, faces, predicted_labels, colormap=dataset.get_colors())

    ious_log_string = '\nClass IoUs: \n'
    ious = []

    confusion_matrix = confusion_matrix[:dataset.get_empty_class(),:dataset.get_empty_class()]  
    labels_strings = dataset.get_learnable_classes_strings()
    for class_i in range(confusion_matrix.shape[0]):

        TP = confusion_matrix[class_i, class_i]
        FP = np.sum(confusion_matrix[:, class_i]) - TP
        FN = np.sum(confusion_matrix[class_i, :]) - TP
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        ious.append(IoU)
        ious_log_string += labels_strings[class_i] + ': %.3f\n' % IoU

    avg_iou = np.mean(ious)
    ious_log_string += 'Average IoU: %.3f\n' % avg_iou

    print ious_log_string

    with open(os.path.join(evaluation_path, "statistics.txt"), "w") as ious_file:
        ious_file.write(ious_log_string)
