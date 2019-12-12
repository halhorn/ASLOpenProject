import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)
    })

    # [0 ~ 255] uint8
    images = tf.decode_raw(features['data'], tf.uint8)
    images = tf.reshape(images, [3, 64, 64])
    images = tf.transpose(images, [1, 2, 0])

    # [0.0 ~ 255.0] float32
    images = tf.cast(images, tf.float32) / 255
    return images

def read_dataset(mode, files_pattern, batch_size, epochs=None):
    tffiles = tf.io.gfile.glob(files_pattern)
    dataset = tf.data.TFRecordDataset(tffiles)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=8)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epochs)
        dataset = dataset.prefetch(buffer_size=None)
    else:
        dataset.repeat(1)
    dataset = dataset.batch(batch_size)
   
    return dataset

def generator_helper(
    noise, is_conditional, one_hot_labels, weight_decay, is_training):
    
    net = tf.layers.dense(noise, 1024, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
    net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=is_training)
    net = tf.nn.relu(net)

    if is_conditional:
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)

    net = tf.layers.dense(net, 8 * 8 * 128, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
    net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=is_training)
    net = tf.nn.relu(net)

    # [-1, 8, 8, 128]
    net = tf.reshape(net, [-1, 8, 8, 128])

    # [-1, 16, 16, 64]
    net = tf.layers.conv2d_transpose(net, 64, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
    net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=is_training)
    net = tf.nn.relu(net)

    # [-1, 32, 32, 32]
    net = tf.layers.conv2d_transpose(net, 32, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
    net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=is_training)
    net = tf.nn.relu(net)
    
    # [-1, 64, 64, 16]
    net = tf.layers.conv2d_transpose(net, 16, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
    net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=is_training)
    net = tf.nn.relu(net)

    # Output should have 3 pixel (grayscale).
    net = tf.layers.conv2d(net, 3, 4, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))

    # Make sure that generator output is in the same range as `inputs`
    # ie [0, 1].
    net = tf.sigmoid(net)

    return net

def discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
    sn_gettr = tfgan.features.spectral_normalization_custom_getter
    with tf.variable_scope('sn', custom_getter=sn_gettr(training=True)):
        print(img)
        net = tf.layers.conv2d(img, 16, 4, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        net = tf.nn.leaky_relu(net, alpha=0.01)

        net = tf.layers.conv2d(net, 32, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        net = tf.nn.leaky_relu(net, alpha=0.01)
        
        net = tf.layers.conv2d(net, 64, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        net = tf.nn.leaky_relu(net, alpha=0.01)
        
        net = tf.layers.conv2d(net, 128, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        net = tf.nn.leaky_relu(net, alpha=0.01)

        net = tf.compat.v1.layers.flatten(net)

        if is_conditional:
          net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)

        net = tf.layers.dense(net, 1024, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        net = tf.layers.batch_normalization(net, momentum=0.999, epsilon=0.001, training=True)
        net = tf.nn.leaky_relu(net, alpha=0.01)

    return net

def generator_fn(noise, mode):
    ## TODO: fix this temporary patch.
    inputs = noise
    if mode == tf.estimator.ModeKeys.PREDICT:
        if dict == type(noise):
            inputs = noise['feature']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    return generator_helper(inputs, False, None, 2.5e-5, is_training)

def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5):
    net = discriminator_helper(img, False, None, weight_decay)
    return tf.layers.dense(net, 1, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))

def input_fn(data_path, mode, params):
    if 'batch_size' not in params:
        raise ValueError('batch_size must be in params')
    if 'noise_dims' not in params:
        raise ValueError('noise_dims must be in params')
        
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat().map(lambda _: tf.random.normal([params['batch_size'], params['noise_dims']])))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return noise_ds
    
    image_ds = read_dataset(mode, data_path, params["batch_size"])
    
    return tf.data.Dataset.zip((noise_ds, image_ds))

def serving_input_receiver_fn(noise_dims):
    receiver_tensors = {
        "noise" : tf.placeholder(dtype=tf.float32, shape=[1, noise_dims], name='serving_noise_dims'),
    }
    features = receiver_tensors["noise"]
    
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors
    )

def train_and_evaluate(params):
    estimator = tfgan.estimator.GANEstimator(
        model_dir=params['model_dir'],
        
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        
        params=params,
        generator_optimizer=tf.train.AdamOptimizer(params['generator_lr'], 0.5),
        
        discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(params['discriminator_lr'], 0.5),
        #add_summaries=tfgan.estimator.SummaryType.IMAGES,
        add_summaries=None,
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(params["data_path"], tf.estimator.ModeKeys.TRAIN, params),
        max_steps=params["num_train_steps"]
    )
    
    exporter = tf.estimator.FinalExporter(
        name="exporter",
        serving_input_receiver_fn=lambda:serving_input_receiver_fn(params["noise_dims"]),
    )
    eval_spec = tf.estimator.EvalSpec(
        name='default',
        input_fn=lambda: input_fn(params["data_path"], tf.estimator.ModeKeys.EVAL, params),
        steps=params["num_eval_steps"],
        exporters=exporter
    )
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator
