import tensorflow as tf
import argparse
import os


#############################################################################################################
# process the raw dataset and parse each record into images and labels and return a iterator over the records
#############################################################################################################
def ProcessRecordDataset(dataset, is_training, batch_size, shuffle_buffur,
                         parse_record_fn, num_epochs=1, num_parallel_calls=1):
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffur)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda value: parse_record_fn(
        value, is_training), num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


################################
# pad zero in both side of image
################################
def FixedPadding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total//2
    pad_end = pad_total-pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_input = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


###########################
# conv2d with fixed padding
###########################
def Conv2dFixedPadding(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = FixedPadding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                            padding=('SAME' if strides == 1 else 'VALID'), use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                            data_format=data_format)


##################################
# construct a block in nin network
##################################
def BlockLayer(inputs, infilters, outfilters, strides, training, data_format):
    inputs = Conv2dFixedPadding(inputs=inputs, filters=infilters,
                                kernel_size=3, strides=strides, data_format=data_format)
    inputs = tf.nn.relu(inputs)
    inputs = Conv2dFixedPadding(inputs=inputs, filters=infilters,
                                kernel_size=1, strides=strides, data_format=data_format)
    inputs = tf.nn.relu(inputs)
    inputs = Conv2dFixedPadding(inputs=inputs, filters=outfilters,
                                kernel_size=1, strides=strides, data_format=data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
    inputs = tf.layers.dropout(inputs, 0.5, training=training)
    return inputs


#####################################
# the model class
# Args:
#   blocks:
#   num_classes:
#   num_filters:
#   data_format:
#####################################
class Model(object):
    def __init__(self, blocks, num_classes, num_filters, data_format):
        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        self.blocks = blocks
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.data_format = data_format

    def __call__(self, inputs, training):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        for i, filters in enumerate(self.num_filters):
            if i is not len(self.num_filters) - 1:
                inputs = BlockLayer(
                    inputs, self.num_filters[i], self.num_filters[i + 1], 1, training, self.data_format)
            else:
                inputs = BlockLayer(
                    inputs, self.num_filters[i], self.num_classes, 1, training, self.data_format)
        if self.data_format == 'channels_last':
            inputs = tf.reduce_mean(inputs, [1, 2])
            inputs = tf.identity(inputs, 'global_avg_pool')
        else:
            inputs = tf.reduce_mean(inputs, [2, 3])
            inputs = tf.identity(inputs, 'global_avg_pool')
        inputs = tf.reshape(inputs, [-1, self.num_classes])
        return inputs


#############################
# learing rate decay function
#############################
def LearningRateWithDecay(batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    initial_learning_rate = 0.1*batch_size/batch_denom
    batches_per_epoch = num_images/batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


##################################################
# shared functionality for different nin model_fns
##################################################
def nin_model_fn(features, labels, mode, model_class,
                 blocks, weight_decay, learning_rate_fn, beta1,
                 beta2, data_format, loss_filter_fn=None):
    tf.summary.image('images', features, max_outputs=6)

    model = model_class(blocks, data_format)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    if not loss_filter_fn:
        def loss_filter_fn(name):
            return 'batch_normalization' not in name

    loss = cross_entropy + weight_decay * \
        tf.add_n([tf.nn.l2_loss(v)
                  for v in tf.trainable_variables() if loss_filter_fn(v.name)])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        labels, predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


###################
# Nin main function
###################
def NinMain(flags, model_function, input_function):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABEL_WINOGRAD_NONFUSED'] = '1'

    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags.model_dir, config=run_config,
        params={
            'blocks': flags.blocks,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size
        })

    for _ in range(flags.train_epochs // flags.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')

    def input_fn_train():
        return input_function(True, flags.data_dir, flags.batch_size,
                              flags.epochs_per_eval, flags.num_parallel_calls)

    classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

    print('Starting to evaluate.')

    def input_fn_eval():
        return input_function(False, flags.data_dir, flags.batch_size,
                              1, flags.num_parallel_calls)
    eval_results = classifier.evaluate(input_fn=input_fn_eval)
    print(eval_results)


class NinArgParser(argparse.ArgumentParser):
    """docstring for NinArgParser"""

    def __init__(self):
        super(NinArgParser, self).__init__()
        self.add_argument(
            '--data_dir', type=str, default='./Dateset',
            help='The directory where the input data is stored.')
        self.add_argument(
            '--num_parallel_calls', type=int, default=5,
            help='The number of records that are precessed in parallel'
            'during input precessing. This can be optimized per data set but'
            'for generally homogeneous data sets, should be approximately the'
            'number of available CPU cores.')
        self.add_argument(
            '--model_dir', type=str, default='./nin_model',
            help='The directory where the model will be stored.')
        self.add_argument(
            '--blocks', type=int, default=3,
            help='The number of blocks in the nin.')
        self.add_argument(
            '--train_epochs', type=int, default=100,
            help='The number of epochs to use for training.')
        self.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='The number of training epochs to run between evaluations')
        self.add_argument(
            '--batch_size', type=int, default=32,
            help='Batch size for training and evaluation.')
        self.add_argument(
            '--data_format', type=str, default=None,
            help='channels_first or channels_last.')


# if __name__ == '__main__':
#     model = Model(3, 2, [128, 64, 64])
#     a = tf.get_variable(
#         'a', [10, 32, 32, 3], initializer=tf.truncated_normal_initializer, dtype=tf.float32)
#     b = model(a, True)
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run(session=sess)
#         print(sess.run(tf.shape(b)))
