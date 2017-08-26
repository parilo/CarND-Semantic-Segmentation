import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# I cecided to make this parameters global
# because it is inconvenient to pass it from func to func
num_classes = 2
image_shape = (160, 576)
train_keep_prob_value = 0.5
learning_rate_value = 1e-4
data_dir = './data'
runs_dir = './runs'
epochs = 500 # on 200 epchs it becomes greater than 80 IOU on train set, 500 is about 85 IOU
batch_size = 24


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    g = tf.get_default_graph()
    return (
        g.get_tensor_by_name(vgg_input_tensor_name),
        g.get_tensor_by_name(vgg_keep_prob_tensor_name),
        g.get_tensor_by_name(vgg_layer3_out_tensor_name),
        g.get_tensor_by_name(vgg_layer4_out_tensor_name),
        g.get_tensor_by_name(vgg_layer7_out_tensor_name)
    )
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

    # 1 x 1 conv
    middle_layer = tf.layers.conv2d(tf.stop_gradient(vgg_layer7_out), 4096, 1, 1, kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='middle_layer')

    upconv3 = tf.layers.conv2d_transpose(middle_layer, 512, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1))
    upconv3 = tf.multiply(0.5, tf.add(upconv3, tf.stop_gradient(vgg_layer4_out)), name='upconv3')

    upconv4 = tf.layers.conv2d_transpose(upconv3, 256, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1))
    upconv4 = tf.multiply(0.5, tf.add(upconv4, tf.stop_gradient(vgg_layer3_out)), name='upconv4')

    upconv5 = tf.layers.conv2d_transpose(upconv4, 128, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv5')
    upconv6 = tf.layers.conv2d_transpose(upconv5,  64, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv6')
    upconv7 = tf.layers.conv2d_transpose(upconv6, num_classes, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='model_output')

    return tf.identity(upconv7, name='model_output_op')
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

    # for educational purposes I decided to calculate IOU by myself
    im_softmax = tf.nn.softmax(nn_last_layer)
    segmentation = tf.where(im_softmax > 0.5, tf.ones_like(im_softmax), tf.zeros_like(im_softmax))
    batch_size = tf.shape(segmentation)[0]

    # calculating intersection count for each image in the batch
    intersection_image = tf.multiply(segmentation[:,:,:,1], correct_label[:,:,:,1])
    intersection_set = tf.reshape(intersection_image, [batch_size, -1])
    intersection = tf.cast(tf.reduce_sum(intersection_set, axis=1, name='intersection'), dtype=tf.float32)

    # calculating union count for each image in the batch
    union_image = tf.where(tf.add(segmentation, correct_label) > 0.5, tf.ones_like(im_softmax), tf.zeros_like(im_softmax))[:,:,:,1]
    union_set = tf.reshape(union_image, [batch_size, -1])
    union = tf.cast(tf.reduce_sum(union_set, axis=1, name='union'), dtype=tf.float32)

    iou = tf.identity(intersection / union, name='iou')

    optimizer = tf.train.AdamOptimizer(learning_rate)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def calc_iou(sess, data_dir, image_shape, nn_last_layer, input_image, keep_prob):
    """
    Calculate mean IOU on training dataset. Function prints IOU of batches during calculation
    :param sess: TensorFLow session
    :param data_dir: Path to the folder that contains the datasets
    :param image_shape: input image size
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param input_image: TF Placeholder for input images
    :param keep_prob: TF Placeholder for dropout keep probability
    :return: float32 mean IOU over train dataset
    """

    gen_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    batch_generator = gen_batches_fn(16)

    g = tf.get_default_graph()
    correct_label = g.get_tensor_by_name('correct_label:0')
    iou = g.get_tensor_by_name('iou:0')

    all_iou = []
    print('--- train dataset iou calculation')
    for image, labels in batch_generator:
        iou_v = sess.run(
            iou,
            {
                keep_prob: 1.0,
                input_image: image,
                correct_label: labels
            })

        mean_batch_iou = np.mean(iou_v)
        all_iou.append(mean_batch_iou)
        print('   batch iou: {}'.format(mean_batch_iou))

    mean_iou = np.mean(all_iou)
    print('--- dataset iou: {}'.format(mean_iou))
    return mean_iou


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

    for i in range(epochs):
        print('--- epoch: {}'.format(i))

        batches = get_batches_fn(batch_size)
        for image_batch, label_batch in batches:

            loss, _ = sess.run(
                [cross_entropy_loss, train_op],
                {
                    input_image: image_batch,
                    correct_label: label_batch,
                    keep_prob: 0.5,
                    learning_rate: 1e-4
                }
            )

            print ('train batch loss: {}'.format(loss))

        if i%100 == 99:

            g = tf.get_default_graph()
            nn_last_layer = g.get_tensor_by_name('model_output_op:0')
            calc_iou(sess, data_dir, image_shape, nn_last_layer, input_image, keep_prob)

            saver = tf.train.Saver(max_to_keep=None)
            save_path = saver.save(sess, 'ckpt/model-{}.ckpt'.format(i))
            print("Model saved in file: %s" % save_path)

    pass
tests.test_train_nn(train_nn)


# for not to rewrite VGG variables during initialization
# code taken from
# https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables/43601894#43601894
def initialize_uninitialized(sess):
    """
    Initialize only not initialized variables in tensorflow.
    """
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def load_checkpoint(sess, ckpt_path):
    """
    Load saved graph checkpoint
    :param sess: TensorFlow session
    :param ckpt_path: path like path/to/your/*.ckpt
    """
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    saver.restore(sess, ckpt_path)


def test_checkpoint(runs_dir, data_dir, image_shape, sess, ckpt, num_classes):

    load_checkpoint(sess, ckpt)
    g = tf.get_default_graph()
    nn_last_layer = g.get_tensor_by_name('model_output_op:0')
    image_input = g.get_tensor_by_name('image_input:0')
    keep_prob = g.get_tensor_by_name('keep_prob:0')

    initialize_uninitialized(sess)

    calc_iou(sess, data_dir, image_shape, nn_last_layer, image_input, keep_prob)


def run():

    parser = argparse.ArgumentParser(description='Train or evaluate semantic segmentation FCN')
    parser.add_argument('-t', '--train', action='store_true', help='train semantic segmentation FCN', required=False)
    parser.add_argument('-e', '--evaluate', metavar=('CKPT_PATH'), type=str, help='evaluate semantic segmentation FCN from checkpoint. Path like path/to/your/*.ckpt', required=False)

    args = parser.parse_args()

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        if args.evaluate is not None:
            test_checkpoint(runs_dir, data_dir, image_shape, sess, args.evaluate, num_classes)
            return

        correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 2), name='correct_label')
        learning_rate = tf.placeholder(tf.float32, shape=())

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        (image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out) = load_vgg(sess, os.path.join(data_dir, 'vgg'))
        nn_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        (logits, train_op, cross_entropy_loss) = optimize(nn_output, correct_label, learning_rate, num_classes)

        initialize_uninitialized(sess)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                     correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
