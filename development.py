"""
Script with development code - once project is completed this code can be removed.
"""

import os
import logging

import tensorflow as tf
import vlogging
import numpy as np
import cv2

import helper


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("segmentation")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def log_training_data(data_dir, image_shape):

    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    batch_generator = get_batches_fn(4)

    logger = get_logger("/tmp/segmentation.html")

    for batch_index in range(500):

        images, labels = next(batch_generator)

        images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
        labels = 255 * labels.astype(np.uint8)

        non_road_labels = [label[:, :, 0] for label in labels]
        road_labels = [label[:, :, 1] for label in labels]

        logger.info(vlogging.VisualRecord("Image batch {}".format(batch_index), images))
        # logger.info(vlogging.VisualRecord("Non-road", non_road_labels))
        logger.info(vlogging.VisualRecord("Road batch {}".format(batch_index), road_labels))


def get_accuracy(ground_truth, prediction):

    all_ones = np.ones(shape=ground_truth.shape[:-1])

    classification = (prediction > 0.5).astype(int)

    non_road_ground_truth = ground_truth[:, :, :, 0]
    road_ground_truth = ground_truth[:, :, :, 1]

    non_road_classification = classification[:, :, :, 0]
    road_classification = classification[:, :, :, 1]

    non_road_intersection = all_ones[(non_road_ground_truth == True) & (non_road_classification == True)]
    non_road_union = all_ones[(non_road_ground_truth == True) | (non_road_classification == True)]
    non_road_accuracy = np.sum(non_road_intersection) / np.sum(non_road_union)

    road_intersection = all_ones[(road_ground_truth == True) & (road_classification == True)]
    road_union = all_ones[(road_ground_truth == True) | (road_classification == True)]
    road_accuracy = np.sum(road_intersection) / np.sum(road_union)

    return non_road_accuracy, road_accuracy


def load_vgg(session, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param session: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(session, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    input_op = session.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_probability_op = session.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3_out_op = session.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4_out_op = session.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7_out_op = session.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_op, keep_probability_op, layer_3_out_op, layer_4_out_op, layer_7_out_op


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Make op same size as vgg_layer4_out
    upscaled_layer_7 = tf.contrib.layers.conv2d_transpose(
        vgg_layer7_out, num_classes, kernel_size=(4, 4), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 4, scale down values so they are of similar
    # range as layer 7 outputs
    scaled_vgg_layer4_out = tf.layers.conv2d(
        0.002 * vgg_layer4_out, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer4_out")

    merged_7_4 = upscaled_layer_7 + scaled_vgg_layer4_out

    # Make op same size as vgg_layer3_out
    upscaled_7_4 = tf.contrib.layers.conv2d_transpose(
        merged_7_4, num_classes, kernel_size=(4, 4), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 3, scale down values so they are of similar
    # range as layer upscaled_7_4 outputs
    scaled_vgg_layer3_out = tf.layers.conv2d(
        0.001 * vgg_layer3_out, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer3_out")

    return merged_7_4, scaled_vgg_layer3_out
    #
    # merged_op = upscaled_7_4 + scaled_vgg_layer3_out
    #
    # # Upscale to original image size
    # upscaled_op = tf.contrib.layers.conv2d_transpose(
    #     merged_op, num_classes, kernel_size=(16, 16), stride=(8, 8), activation_fn=tf.nn.relu)
    #
    # return upscaled_op


def get_batching_function(data_dir, image_shape, batch_size):

    get_batches_fn = helper.gen_batch_function(data_dir, image_shape)
    batches_generator = get_batches_fn(batch_size)

    def get_batch(batch_size):

        images, labels = next(batches_generator)
        return images, labels

    return get_batch


def main():

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    label_shape = (None,) + image_shape + (num_classes,)
    correct_label_placeholder = tf.placeholder(tf.float32, label_shape)
    learning_rate_placeholder = tf.placeholder(tf.float32, [])

    epochs = 50
    batch_size = 8

    training_data_dir = os.path.join(data_dir, 'data_road/training')

    get_training_batches = get_batching_function(training_data_dir, image_shape, batch_size)

    with tf.Session() as session:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image_placeholder, keep_probability_placeholder, layer_3_out_op, layer_4_out_op, layer_7_out_op = \
            load_vgg(session, vgg_path)

        tf.summary.image("Input images", input_image_placeholder)

        road_label_op = tf.expand_dims(correct_label_placeholder[:, :, :, 1], axis=3)
        tf.summary.image("Road label", road_label_op)

        merged_summary_op = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter("/tmp/network", session.graph)
        session.run(tf.global_variables_initializer())

        for index in range(100):

            print(index)

            images, labels = get_training_batches(batch_size)

            feed_dictionary = {
                input_image_placeholder: images,
                correct_label_placeholder: labels,
                keep_probability_placeholder: 1.0
            }

            summary = session.run(merged_summary_op, feed_dictionary)
            train_writer.add_summary(summary)


if __name__ == "__main__":

    main()
