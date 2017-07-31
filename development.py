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
        logger.info(vlogging.VisualRecord("label 1", label[:, :, 1]))


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

    # Op is same size as vgg_layer4_out
    upscaled_layer_7 = tf.contrib.layers.conv2d_transpose(
        vgg_layer7_out, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 4, scale down values so they are of similar
    # range as layer 7 outputs
    scaled_vgg_layer4_out = 0.01 * tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer4_out")

    merged_7_4 = upscaled_layer_7 + scaled_vgg_layer4_out

    upscaled_7_4 = tf.contrib.layers.conv2d_transpose(
        merged_7_4, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 3, scale down values so they are of similar
    # range as layer upscaled_7_4 outputs
    scaled_vgg_layer3_out = 0.01 * tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer3_out")

    merged_op = upscaled_7_4 + scaled_vgg_layer3_out

    upscaled_op = tf.contrib.layers.conv2d_transpose(
        merged_op, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

    upscaled_op = tf.contrib.layers.conv2d_transpose(
        upscaled_op, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

    logits_op = tf.contrib.layers.conv2d_transpose(
        upscaled_op, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

    return logits_op


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, shape=[-1, num_classes])
    flattened_correct_label = tf.reshape(correct_label, shape=[-1, num_classes])

    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flattened_correct_label, logits=logits)
    mean_loss = tf.reduce_mean(cross_entropy_loss)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

    return logits, train_op, mean_loss


def train_nn(session, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param session: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass

tests.test_train_nn(train_nn)


def main():

    logger = get_logger("/tmp/segmentation.html")

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # print("Kitti test")
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    log_training_data(data_dir, image_shape)

    label_shape = (None,) + image_shape + (num_classes,)

    correct_label_placeholder = tf.placeholder(tf.float32, label_shape)
    learning_rate_placeholder = tf.placeholder(tf.float32, (None,))

    with tf.Session() as session:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        batches_generator = get_batches_fn(4)
        images, labels = next(batches_generator)

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_op, keep_probability_op, layer_3_out_op, layer_4_out_op, layer_7_out_op = load_vgg(session, vgg_path)
        logits_op = layers(layer_3_out_op, layer_4_out_op, layer_7_out_op, num_classes)

        logits, train_op, loss_op = optimize(
            logits_op, correct_label_placeholder, learning_rate_placeholder, num_classes)

        session.run(tf.global_variables_initializer())
        feed_dictionary = {input_op: images, correct_label_placeholder: labels, keep_probability_op: 1.0}

        loss = session.run(loss_op, feed_dictionary)
        print(loss)

        print(loss.shape)


if __name__ == "__main__":

    main()
