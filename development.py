"""
Script with development code - once project is completed this code can be removed.
"""

import os
import logging

import tensorflow as tf
import vlogging
import numpy as np
import cv2

import project_tests as tests
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

    images, labels = next(batch_generator)

    logger = get_logger("/tmp/segmentation.html")

    for image, label in zip(images, labels):
        label = 255 * label.astype(np.uint8)

        logger.info(vlogging.VisualRecord("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
        logger.info(vlogging.VisualRecord("Non-road", label[:, :, 0]))
        logger.info(vlogging.VisualRecord("Road", label[:, :, 1]))


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


def main():

    data_dir = './data'

    with tf.Session() as session:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image_placeholder, keep_probability_placeholder, layer_3_out_op, layer_4_out_op, layer_7_out_op = \
            load_vgg(session, vgg_path)

        # nodes = [node for node in session.graph.get_operations()]
        #
        # names = [node.name for node in nodes]
        # print(*names, sep="\n")

        variables = tf.global_variables()

        bias_variable = [variable for variable in variables if "conv1_1/biases:0" in variable.name][0]
        conv_variable = [variable for variable in variables if "conv3_2/filter:0" in variable.name][0]

        session.run(tf.global_variables_initializer())

        bias = session.run(bias_variable)

        print("bias")
        print(np.min(bias))
        print(np.mean(bias))
        print(np.max(bias))

        convolution = session.run(conv_variable)

        print("convolution")
        print(np.min(convolution))
        print(np.mean(convolution))
        print(np.max(convolution))




if __name__ == "__main__":

    main()
