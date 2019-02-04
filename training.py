""" training.py - All functionality specific to training a model. """

from __future__ import division
import sys
sys.path.append('util')
import os
import threading
import util
import tensorflow as tf
import numpy as np
import fcpn
import data
np.set_printoptions(threshold=np.nan)

def create_new_session(config):
    """ Creates a new folder to save all session artifacts to. 

    Args:
    config: dict, session configuration parameters

    """
    
    if not os.path.exists('sessions'): os.mkdir('sessions')
    config['training']['session_dir'] = os.path.join('sessions', 'session_' + str(config['training']['session_id']))
    if not os.path.exists(config['training']['session_dir']): os.mkdir(config['training']['session_dir'])

def load_data_into_queue(sess, enqueue_op, queue_placeholders, coord, model, dataset, config):
    """ Fills a FIFO queue with one epoch of training samples, then one epoch of validation samples. Alternatingly, for config['training']['max_epochs'] epochs.

    Args:
    sess: tf.Session
    enqueue_op: tf.FIFOQueue.enqueue
    queue_placeholders: dict
    coord: tf.train.Coordinator()
    model: FCPN
    dataset: Dataset
    config: dict, session configuration parameters

    """

    sample_generators = {
        'train': dataset.sample_generator('train', config['dataset']['training_samples']['num_points'], config['training']['data_augmentation']),
        'val': dataset.sample_generator('val', config['dataset']['training_samples']['num_points'])
    }

    pointnet_locations = model.get_pointnet_locations()
    point_features = np.ones(config['dataset']['training_samples']['num_points'])
    pointnet_features = np.zeros(config['model']['pointnet']['num'])

    constant_features = np.expand_dims(np.concatenate([point_features, pointnet_features]), axis=1)

    for _ in range(config['training']['max_epochs']):
        for s in ['train', 'val']:
            num_enqueued_samples = 0

            for sample_i in range(dataset.get_num_samples(s)):

                if coord.should_stop():
                    return

                input_points_xyz, output_voxelgrid = next(sample_generators[s])
                output_voxelvector = output_voxelgrid.reshape(-1)

                points_xyz_and_pointnet_locations = np.concatenate(
                    (input_points_xyz, pointnet_locations), axis=0)
                voxel_weights = dataset.get_voxel_weights(output_voxelvector)

                feed_dict = {queue_placeholders['input_points_pl']: points_xyz_and_pointnet_locations,
                             queue_placeholders['input_features_pl']: constant_features,
                             queue_placeholders['output_voxels_pl']: output_voxelvector,
                             queue_placeholders['output_voxel_weights_pl']: voxel_weights}

                sess.run(enqueue_op, feed_dict=feed_dict)
                num_enqueued_samples += 1

                # If its the last sample of the batch, repeat it to complete
                # the last batch
                if num_enqueued_samples == dataset.get_num_samples(s):
                    num_duplicate_samples = dataset.get_num_batches(s, config['training']['batch_size']) * config['training']['batch_size'] - num_enqueued_samples
                    for _ in range(num_duplicate_samples):
                        sess.run(enqueue_op, feed_dict=feed_dict)

def setup_queue(num_input_points, num_output_voxels, batch_size, queue_size=100):
    """ Setup a tf.FIFOQueue for preloading samples during training

    Args:
    num_input_points: int
    num_output_voxels: int
    batch_size: int
    queue_size: int

    Returns:
    enqueue_op: tf.FIFOQueue.enqueue
    queue_placeholders: dict
    queue_batch_placeholders: dict
    get_size_op: tf.FIFOQueue.size

    """

    input_points_pl = tf.placeholder(
        tf.float32, shape=[num_input_points, 3], name='input_points_pl')
    input_features_pl = tf.placeholder(
        tf.float32, shape=[num_input_points, 1], name='input_features_pl')
    output_voxels_pl = tf.placeholder(
        tf.int32, shape=[num_output_voxels], name='output_voxels_pl')
    output_voxel_weights_pl = tf.placeholder(
        tf.float32, shape=[num_output_voxels], name='output_voxel_weights_pl')

    queue_placeholders = {'input_points_pl': input_points_pl,
                          'input_features_pl': input_features_pl,
                          'output_voxels_pl': output_voxels_pl,
                          'output_voxel_weights_pl': output_voxel_weights_pl}

    q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32, tf.float32], shapes=[
                     [num_input_points, 3], [num_input_points, 1], [num_output_voxels], [num_output_voxels]])

    enqueue_op = q.enqueue([input_points_pl, input_features_pl,
                            output_voxels_pl, output_voxel_weights_pl])
    batch_input_points_pl, batch_input_features_pl, batch_output_voxels_pl, batch_output_voxel_weights_pl = q.dequeue_many(
        batch_size)
    queue_batch_placeholders = {
        'input_points_pl': batch_input_points_pl,
        'input_features_pl': batch_input_features_pl,
        'output_voxels_pl': batch_output_voxels_pl,
        'output_voxel_weights_pl': batch_output_voxel_weights_pl
    }
    get_size_op = q.size(name='get_q_size_op')

    return enqueue_op, queue_placeholders, queue_batch_placeholders, get_size_op


def start_data_loader(sess, enqueue_op, queue_placeholders, model, dataset, config):
    """ Starts a data loader thread coordinated by a tf.train.Coordinator()

    Args:
    sess: tf.Session
    enqueue_op: tf.FIFOQueue.enqueue
    queue_placeholders: dict
    model: FCPN
    dataset: Dataset
    config: dict, session configuration parameters

    Returns:
    coord: tf.train.Coordinator
    loader_thread: Thread

    """

    coord = tf.train.Coordinator()
    loader_thread = threading.Thread(target=load_data_into_queue, args=(
        sess, enqueue_op, queue_placeholders, coord, model, dataset, config))
    loader_thread.daemon = True
    loader_thread.start()
    return coord, loader_thread


def train(config_path):
    """ Trains a model for a maximum of config.max_epochs epochs

    Args:
    config_path: string, path to a config.json file

    """

    # Load configuration
    if not os.path.exists(config_path):
        print 'Error: No configuration file present at specified path.'
        return
        
    config = util.load_config(config_path)
    print 'Loaded configuration from: %s' % config_path

    # Create session directory
    if 'session_dir' not in config['training'] or os.path.exists(config['training']['session_dir']): create_new_session(config)

    # Direct all output to screen and log file
    util.set_print_to_screen_and_file(
        os.path.join(config['training']['session_dir'], 'session.log'))

    model = fcpn.FCPN(config)
    dataset = data.Dataset(config)
    dataset.prepare(config['dataset']['refresh_cache'])

    config['model']['pointnet']['num'] = np.prod(model.get_feature_volume_shape(
        config['dataset']['training_samples']['spatial_size'], config['model']['pointnet']['spacing'], 1))

    enqueue_op, queue_placeholders, queue_batch_placeholders, get_queue_size_op = setup_queue(
        config['dataset']['training_samples']['num_points'] + config['model']['pointnet']['num'], dataset.get_num_output_voxels(), config['training']['batch_size'])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = config['training']['gpu']['allow_growth']
    tf_config.allow_soft_placement = config['training']['gpu']['allow_soft_placement']

    sess = tf.Session(config=tf_config)

    with sess.as_default():
        with tf.device('/gpu:' + str(config['training']['gpu']['id'])):

            # Batch normalization
            batch_i = tf.Variable(0, name='batch_i')
            batch_normalization_decay = util.get_batch_normalization_decay(
                batch_i, config['training']['batch_size'], config['training']['optimizer']['batch_normalization']['initial_decay'], config['training']['optimizer']['batch_normalization']['decay_rate'], config['training']['optimizer']['batch_normalization']['decay_step'])
            tf.summary.scalar('batch_normalization_decay',
                              batch_normalization_decay)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Build model
            pred_op = model.build_model(config['training']['batch_size'], config['dataset']['training_samples']['spatial_size'], queue_batch_placeholders['input_points_pl'],
                                        queue_batch_placeholders['input_features_pl'], is_training_pl, dataset.get_num_learnable_classes(), batch_normalization_decay)
            
            # Loss
            loss_op = model.get_loss(
                pred_op, queue_batch_placeholders['output_voxels_pl'], queue_batch_placeholders['output_voxel_weights_pl'])

            model.print_num_parameters()
            model.print_layer_weights()

            # Confusion matrix
            confusion_matrix_op, confusion_matrix_update_op, confusion_matrix_clear_op = model.get_confusion_matrix_ops(
                pred_op, queue_batch_placeholders['output_voxels_pl'], dataset.get_num_learnable_classes(), dataset.get_empty_class())

            # Optimizer
            learning_rate_op = util.get_learning_rate(
                batch_i, config['training']['batch_size'], config['training']['optimizer']['learning_rate']['initial'], config['training']['optimizer']['learning_rate']['decay_rate'], config['training']['optimizer']['learning_rate']['decay_step'])
            tf.summary.scalar('learning_rate', learning_rate_op)

            optimizer_op = tf.train.AdamOptimizer(learning_rate_op)
            if config['training']['train_upsampling_only']:
                upsampling_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "upsampling")
                optimization_op = optimizer_op.minimize(loss_op, var_list=upsampling_weights, global_step=batch_i)
            else:
                optimization_op = optimizer_op.minimize(loss_op, global_step=batch_i)

            # Summary and Saving
            saver = tf.train.Saver(max_to_keep=config['training']['checkpoints_to_keep'])
            merged_summary_op = tf.summary.merge_all()
            summary_writers = {
                'train': tf.summary.FileWriter(os.path.join(config['training']['session_dir'], 'train'), sess.graph),
                'val': tf.summary.FileWriter(os.path.join(config['training']['session_dir'], 'val'))
            }

            # Initialize variables in graph
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            sess.run([init_g, init_l], {is_training_pl: True})

            # Restore model weights from disk
            if config['training']['checkpoint_path']:

                weights_to_be_restored = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                # If finetuning on a new dataset, don't load last layer weights or confusion matrix
                if config['training']['finetune_new_classes']:
                    final_layer_weights =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="upsampling/15cm_to_5cm/final_conv")
                    confusion_variables =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="confusion")
                    weights_to_be_restored = list(set(weights_to_be_restored) - set(final_layer_weights) - set(confusion_variables))

                restorer = tf.train.Saver(var_list=weights_to_be_restored)
                restorer.restore(sess, config['training']['checkpoint_path'])
                print 'Model weights restored from checkpoint file: %s' % config['training']['checkpoint_path']

            num_batches = {
                'train': dataset.get_num_batches('train', config['training']['batch_size']),
                'val': dataset.get_num_batches('val', config['training']['batch_size'])
            }
            ops = {
                'train': [loss_op, merged_summary_op, optimization_op],
                'val': [loss_op, merged_summary_op, confusion_matrix_update_op]
            }

            # Start loading samples into FIFO queue
            coord, loader_thread = start_data_loader(
                sess, enqueue_op, queue_placeholders, model, dataset, config)

            # Save configuration file (with derived parameters) to session directory
            util.save_config(os.path.join(config['training']['session_dir'], 'config.json'), config)                

            # Start training
            sample_i = 0
            for epoch_i in range(config['training']['max_epochs']):
                print '\nEpoch: %d' % epoch_i

                for s in ['train', 'val']:

                    is_training = (s == 'train')

                    if s == 'train':
                        is_training = True
                        print 'Training set\nBatch/Total Batches | Loss | Items in Queue'
                    else:                        
                        print 'Validation set\nBatch/Total Batches | Loss | Items in Queue'

                    for epoch_batch_i in range(num_batches[s]):

                        loss, summary, _ = sess.run(
                            ops[s], feed_dict={is_training_pl: is_training})

                        # Log statistics
                        if epoch_batch_i % config['training']['log_every_n_batches'] == 0:
                            summary_writers[s].add_summary(summary, sample_i)
                            summary_writers[s].flush()
                            print '%i/%i | %f | %d' % (epoch_batch_i + 1, num_batches[s], loss, get_queue_size_op.eval())

                        # Only do when in training phase
                        if s == 'train':
                            sample_i += config['training']['batch_size']

                            # Save snapshot of model
                            if epoch_batch_i % config['training']['save_every_n_batches'] == 0:
                                save_path = saver.save(sess, os.path.join(
                                    config['training']['session_dir'], "model.ckpt"), global_step=epoch_i)
                                print 'Checkpoint saved at batch %d to %s' % (
                                    epoch_batch_i, save_path)

                    # Only do at the end of the validation phase
                    if s == 'train':
                        save_path = saver.save(sess, os.path.join(
                            config['training']['session_dir'], "model.ckpt"), global_step=epoch_i)
                        print 'Checkpoint saved at batch %d to %s' % (epoch_batch_i, save_path)
                    elif s == 'val':
                        confusion_matrix = confusion_matrix_op.eval()

                        # Compute and print per-class statistics
                        true_positives, false_negatives, false_positives, ious = util.compute_per_class_statistics(confusion_matrix[:dataset.get_empty_class(),:dataset.get_empty_class()])
                        util.pretty_print_confusion_matrix(confusion_matrix, dataset.get_learnable_classes_strings())
                        util.pretty_print_per_class_statistics(dataset.get_learnable_classes_strings()[:dataset.get_empty_class()], true_positives, false_negatives, false_positives, ious)
                        avg_iou = np.mean(ious)

                        summary = tf.Summary()
                        summary.value.add(
                            tag='avg_iou', simple_value=avg_iou)

                        # Add per-class IoUs to summary to be viewable in Tensorboard
                        for class_i, class_label in enumerate(dataset.get_learnable_classes_strings()[:dataset.get_empty_class()]):
                            summary.value.add(
                                tag=class_label + '_iou', simple_value=ious[class_i])

                        summary_writers[s].add_summary(summary, sample_i)
                        summary_writers[s].flush()
                        confusion_matrix_clear_op.eval()

            coord.request_stop()
            coord.join([loader_thread])

            print 'Training complete.'
