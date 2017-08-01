import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def get_accuracy(ground_truth, prediction, image_shape):

    ground_truth = ground_truth.reshape((-1,) + image_shape + (2, ))
    prediction = prediction.reshape((-1,) + image_shape + (2,))

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

tests.test_load_vgg(load_vgg, tf)


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

    # Don't train old network
    vgg_layer3_out_stopped = tf.stop_gradient(vgg_layer3_out)
    vgg_layer4_out_stopped = tf.stop_gradient(vgg_layer4_out)
    vgg_layer7_out_stopped = tf.stop_gradient(vgg_layer7_out)

    # Make op same size as vgg_layer4_out
    upscaled_layer_7 = tf.contrib.layers.conv2d_transpose(
        vgg_layer7_out_stopped, num_classes, kernel_size=(4, 4), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 4, scale down values so they are of similar
    # range as layer 7 outputs
    scaled_vgg_layer4_out = tf.layers.conv2d(
        0.002 * vgg_layer4_out_stopped, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer4_out")

    merged_7_4 = tf.add(upscaled_layer_7, scaled_vgg_layer4_out)

    # Make op same size as vgg_layer3_out
    upscaled_7_4 = tf.contrib.layers.conv2d_transpose(
        merged_7_4, num_classes, kernel_size=(4, 4), stride=(2, 2), activation_fn=tf.nn.relu)

    # Create proper kernels number from layer 3, scale down values so they are of similar
    # range as layer upscaled_7_4 outputs
    scaled_vgg_layer3_out = tf.layers.conv2d(
        0.001 * vgg_layer3_out_stopped, num_classes, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
        name="scaled_vgg_layer3_out")

    merged_op = tf.add(upscaled_7_4, scaled_vgg_layer3_out)

    # Upscale to original image size
    upscaled_op = tf.contrib.layers.conv2d_transpose(
        merged_op, num_classes, kernel_size=(16, 16), stride=(8, 8), activation_fn=tf.nn.relu)

    return upscaled_op

tests.test_layers(layers)


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

    logits = tf.reshape(tf.cast(nn_last_layer, tf.float32), shape=[-1, num_classes])
    flattened_correct_label = tf.reshape(tf.cast(correct_label, tf.float32), shape=[-1, num_classes])

    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flattened_correct_label, logits=logits)
    mean_loss = tf.reduce_mean(cross_entropy_loss)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

    return logits, train_op, mean_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
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
    # There are 290 images in both training and test datasets
    batches_per_epoch = int(290 / batch_size) + 1

    learning_rate_value = 0.0001

    for epoch_index in range(epochs):

        print("Epoch {} start".format(epoch_index))

        for batch_index in range(batches_per_epoch):

            images, labels = get_batches_fn(batch_size)

            feed_dictionary = {
                input_image: images,
                correct_label: labels,
                keep_prob: 1.0,
                learning_rate: learning_rate_value * (0.99 ** epoch_index)
            }

            sess.run(train_op, feed_dictionary)

            if batch_index % 10 == 0:
                loss = sess.run(cross_entropy_loss, feed_dictionary)
                print("\tBatch loss: {}".format(loss))

        print("Epoch {} end".format(epoch_index))

tests.test_train_nn(train_nn)


def get_batching_function(data_dir, image_shape, batch_size):

    get_batches_fn = helper.gen_batch_function(data_dir, image_shape)
    batches_generator = get_batches_fn(batch_size)

    def get_batch(batch_size):

        images, labels = next(batches_generator)
        return images, labels

    return get_batch


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    label_shape = (None,) + image_shape + (num_classes,)
    correct_label_placeholder = tf.placeholder(tf.float32, label_shape)
    learning_rate_placeholder = tf.placeholder(tf.float32, [])

    epochs = 100
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

        segmentation_op = layers(layer_3_out_op, layer_4_out_op, layer_7_out_op, num_classes)

        logits, train_op, loss_op = optimize(
            segmentation_op, correct_label_placeholder, learning_rate_placeholder, num_classes)

        images, labels = get_training_batches(batch_size)

        feed_dictionary = {
            input_image_placeholder: images,
            correct_label_placeholder: labels,
            keep_probability_placeholder: 1.0
        }

        session.run(tf.global_variables_initializer())

        predictions = session.run(tf.nn.softmax(logits), feed_dictionary)
        non_road_accuracy, road_accuracy = get_accuracy(labels, predictions, image_shape)
        print("Before training:\nNon road accuracy: {}\nRoad accuracy: {}".format(non_road_accuracy, road_accuracy))

        # TODO: Train NN using the train_nn function
        train_nn(
            session, epochs, batch_size, get_training_batches, train_op, loss_op,
            input_image_placeholder, correct_label_placeholder,
            keep_probability_placeholder, learning_rate_placeholder)

        predictions = session.run(tf.nn.softmax(logits), feed_dictionary)
        non_road_accuracy, road_accuracy = get_accuracy(labels, predictions, image_shape)
        print("After training:\nNon road accuracy: {}\nRoad accuracy: {}".format(non_road_accuracy, road_accuracy))

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(
            runs_dir, data_dir, session, image_shape, logits, keep_probability_placeholder, input_image_placeholder)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
