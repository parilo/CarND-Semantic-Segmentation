import os.path
import tensorflow as tf
import numpy as np
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

vgg_out = None
middle_layer = None
upconv2 = None

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    global middle_layer, vgg_out
    # print(vgg_layer3_out) # 256
    # print(vgg_layer4_out) # 512
    # print(vgg_layer7_out)

    vgg_out = vgg_layer7_out

    # 1 x 1 conv
    middle_layer = tf.layers.conv2d(tf.stop_gradient(vgg_layer7_out), 4096, 1, 1, kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='middle_layer')

    # upconv1 = tf.layers.conv2d_transpose(middle_layer, 512, [7, 7], [1, 1], padding='VALID', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv1')
    # upconv2 = tf.layers.conv2d_transpose(upconv1, 512, [3, 3], [2, 2], padding='VALID', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv2')

    upconv3 = tf.layers.conv2d_transpose(middle_layer, 512, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1))
    upconv3 = tf.multiply(0.5, tf.add(upconv3, tf.stop_gradient(vgg_layer4_out)), name='upconv3')

    upconv4 = tf.layers.conv2d_transpose(upconv3, 256, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1))
    upconv4 = tf.multiply(0.5, tf.add(upconv4, tf.stop_gradient(vgg_layer3_out)), name='upconv4')

    upconv5 = tf.layers.conv2d_transpose(upconv4, 128, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv5')
    upconv6 = tf.layers.conv2d_transpose(upconv5,  64, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='upconv6')
    upconv7 = tf.layers.conv2d_transpose(upconv6, num_classes, [3, 3], [2, 2], padding='same', kernel_initializer=tf.truncated_normal_initializer(0, 1e-1), name='model_output')


    # print(middle_layer)
    # print(upconv1)
    # print(upconv2)
    # print(upconv3)
    # print(upconv4)
    # print(upconv5)
    # print(upconv6)
    # print(upconv7)

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

    # print (nn_last_layer)
    # print (correct_label)

    # intersection over union
    # iou, iou_op = tf.metrics.mean_iou(correct_label, nn_last_layer, num_classes)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def check_iou(sess, data_dir, image_shape, logits, input_image, keep_prob, correct_label, iou):

    print('--- check iou')
    for image, label in helper.gen_test_images(os.path.join(data_dir, 'data_road/training'), image_shape):
        iou = sess.run(
            [iou],
            {
                keep_prob: 1.0,
                input_image: [image],
                correct_label: [label]
            }
        )
        print(iou)
        return
        # print(image.shape)
        # print(label.shape)
        # im_softmax = sess.run(
        #     [tf.nn.softmax(logits)],
        #     {keep_prob: 1.0, input_image: [image]})
        # im_softmax = im_softmax[0][0]
        # segmentation_indices = (im_softmax > 0.5)
        # segmentation = np.zeros_like(im_softmax)
        # segmentation[segmentation_indices] = 1
        # print(segmentation.shape)
        # print(segmentation)
        # return
        # im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        # segmentation = (im_softmax > 0.5)
        # print(segmentation.shape)
        # print(segmentation)
        # mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        # mask = scipy.misc.toimage(mask, mode="RGBA")
        # street_im = scipy.misc.toimage(image)
        # street_im.paste(mask, box=None, mask=mask)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, nn_output, save_inference_samples_func):
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


    saver = tf.train.Saver(max_to_keep=None)
    # saver.restore(self.sess, "submissions/6/model-16500000.ckpt")

    for i in range(epochs):
        print('--- epoch: {}'.format(i))

        batches = get_batches_fn(batch_size)
        for image_batch, label_batch in batches:

            # s = sess.run(tf.shape(nn_output),
            # {
            #     input_image: image_batch,
            #     correct_label: label_batch,
            #     keep_prob: 0.5,
            #     learning_rate: 1e-4
            # })
            # print('--- nn output')
            # print(s)
            # return

            loss, _ = sess.run(
                [cross_entropy_loss, train_op],
                {
                    input_image: image_batch,
                    correct_label: label_batch,
                    keep_prob: 0.5,
                    learning_rate: 1e-4
                }
            )

            print ('loss: {}'.format(loss))

        if i%100 == 0:
            save_path = saver.save(sess, 'ckpt/model-{}.ckpt'.format(i))
            print("Model saved in file: %s" % save_path)
            save_inference_samples_func()

    pass
# tests.test_train_nn(train_nn)


def load_checkpoint(sess, ckpt_path):
    saver = tf.train.import_meta_graph('ckpt/model.ckpt.meta')
    saver.restore(sess, ckpt_path)
    # for v in tf.global_variables():
    #     print(v.name)
    # for op in tf.get_default_graph().get_operations():
    #     print(op.name)

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

    epochs = 1001
    batch_size = 24

    with tf.Session() as sess:

        correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 2))
        learning_rate = tf.placeholder(tf.float32, shape=())

        # https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables/43601894#43601894
        def initialize_uninitialized(sess):
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        load_checkpoint(sess, 'ckpt/model-100.ckpt')
        g = tf.get_default_graph()
        logits = g.get_tensor_by_name('model_output_op:0')
        input_image = g.get_tensor_by_name('image_input:0')
        keep_prob = g.get_tensor_by_name('keep_prob:0')
        iou = tf.metrics.mean_iou(tf.argmax(correct_label, 3), tf.argmax(logits, 3), num_classes)

        initialize_uninitialized(sess)
        sess.run(tf.local_variables_initializer())

        check_iou(sess, data_dir, image_shape, logits, input_image, keep_prob, correct_label, iou)

        return

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        (image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out) = load_vgg(sess, os.path.join(data_dir, 'vgg'))
        nn_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        print(nn_output)

        (logits, train_op, cross_entropy_loss) = optimize(nn_output, correct_label, learning_rate, num_classes)

        # saver = tf.train.Saver(max_to_keep=None)
        # save_path = saver.save(sess, 'ckpt/model.ckpt')
        # return

        def save_inference_samples_func():
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                     correct_label, keep_prob, learning_rate, nn_output, save_inference_samples_func)


        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
