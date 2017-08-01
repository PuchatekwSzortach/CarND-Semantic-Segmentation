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

    classification = (prediction > 0.5).astype(int)


    non_road_ground_truth = ground_truth[:, :, :, 0]
    road_ground_truth = ground_truth[:, :, :, 1]

    non_road_classification = classification[:, :, :, 0]
    road_classification = classification[:, :, :, 1]

    non_road_union = (non_road_ground_truth == non_road_classification).astype(np.int)
    non_road_accuracy = np.sum(non_road_union) / np.prod(non_road_ground_truth.shape)

    road_union = (road_ground_truth == road_classification).astype(np.int)
    road_accuracy = np.sum(road_union) / np.prod(road_ground_truth.shape)

    return non_road_accuracy, road_accuracy


def main():

    # data_dir = './data'
    # image_shape = (160, 576)
    #
    # log_training_data(data_dir, image_shape)

    ground_truth = np.zeros(shape=(1, 10, 10, 2))

    # Non-road surface
    ground_truth[:, :5, :5, 0] = 1

    # Road surface
    ground_truth[:, :2, :2, 1] = 1

    prediction = np.zeros(shape=(1, 10, 10, 2))

    # Non-road surface
    prediction[:, :4, :4, 0] = 0.7

    # Road surface
    prediction[:, :6, :6, 1] = 0.7

    nonroad_accuracy, road_accuracy = get_accuracy(ground_truth, prediction)

    print("Expected nonroad accuracy: {}".format(0.91))
    print("Actual nonroad accuracy: {}".format(nonroad_accuracy))
    print()
    print("Expected road accuracy: {}".format(0.68))
    print("Actual road accuracy: {}".format(road_accuracy))




if __name__ == "__main__":

    main()
