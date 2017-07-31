import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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

    upscaled_op = tf.contrib.layers.conv2d_transpose(
        upscaled_op, num_classes, kernel_size=(2, 2), stride=(2, 2), activation_fn=tf.nn.relu)

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

    logits = tf.reshape(nn_last_layer, shape=[-1, num_classes])
    flattened_correct_label = tf.reshape(correct_label, shape=[-1, num_classes])

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
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        images, labels = get_batches_fn(batch_size)

        feed_dictionary = {

            input_image: images,
            correct_label: labels,
            keep_prob: 0.5,
            learning_rate: 0.001
        }

        loss, _ = sess.run([cross_entropy_loss, train_op], feed_dictionary)
        print(loss)

tests.test_train_nn(train_nn)


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

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
