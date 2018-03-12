import os
import nin
import tensorflow as tf
import sys

# test package
import numpy as np
import cv2

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_NUM_CLASSES = 2
_DEFAULT_IMAGE_BYTES = _HEIGHT*_WIDTH*_NUM_CHANNELS
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1

_NUM_IMAGES = {
    'train': 5000,
    'validation': 1000,
}


##################################################
# Get the file names in the data/datatest directory
##################################################
def GetFilenames(is_training, data_dir):
    assert os.path.exists(data_dir), (
        'Can not find the directory of data')
    if is_training:
        data_dir = os.path.join(data_dir, 'data')
    else:
        data_dir = os.path.join(data_dir, 'datatest')
    filenames = sorted(os.listdir(data_dir))
    filenames = [os.path.join(data_dir, i) for i in filenames]
    return filenames


###################################################
# Parse the record label and image from the raw data
###################################################
def parse_record(raw_record, is_training):
    record = tf.decode_raw(raw_record, tf.uint8)
    label = tf.cast(record[0], tf.int32)
    image = tf.reshape(record[1: _RECORD_BYTES], [
                       _WIDTH, _HEIGHT, _NUM_CHANNELS])
    image = tf.cast(image, tf.float32)
    image = preprocess_image(image, is_training)
    return image, label


#######################################
# randomly flip the image left or right
#######################################
def preprocess_image(image, is_training):
    if is_training:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_parallel_calls=1):
    filenames = GetFilenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    return nin.ProcessRecordDataset(dataset, is_training, batch_size,
                                    _NUM_IMAGES['train'], parse_record, num_epochs, num_parallel_calls)


class Cifar10Model(nin.Model):
    """docstring for Cifar10Model"""

    def __init__(self, blocks, data_format=None, num_classes=_NUM_CLASSES):
        super(Cifar10Model, self).__init__(
            blocks=blocks,
            num_classes=num_classes,
            num_filters=[128, 64, 64],
            data_format=data_format)


def Cifar10ModelFunction(features, labels, mode, params):
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = nin.LearningRateWithDecay(batch_size=params['batch_size'], batch_denom=128,
                                                 num_images=_NUM_IMAGES['train'], boundary_epochs=[
                                                     100, 150, 200],
                                                 decay_rates=[1, 0.1, 0.01, 0.001])

    weight_decay = 2e-4

    def loss_filter_fn(name):
        return True

    return nin.nin_model_fn(features, labels, mode, Cifar10Model,
                            blocks=params['blocks'],
                            weight_decay=weight_decay,
                            learning_rate_fn=learning_rate_fn,
                            beta1=0.9,
                            beta2=0.999,
                            data_format=params['data_format'],
                            loss_filter_fn=loss_filter_fn)


def main(unused_argv):
    nin.NinMain(FLAGS, Cifar10ModelFunction, input_fn)


if __name__ == '__main__':
    # dataset = input_fn('True', './Dataset', 1)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # sess = tf.Session()
    # image, label = sess.run(next_element)
    # print(label)
    # cv2.imshow(' ', image[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = nin.NinArgParser()
    parser.set_defaults(data_dir='./Dataset',
                        model_dir='./cifar10_model',
                        train_epochs=250,
                        epochs_per_eval=10,
                        batch_size=128)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]]+unparsed)
