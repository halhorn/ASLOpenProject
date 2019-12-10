import tensorflow as tf
CLASS_NUM = 3862
RGB_DIM = 1024
AUDIO_DIM = 128


def multi_hot(indices):
    return tf.reduce_sum(tf.one_hot(indices, CLASS_NUM), axis=-2)


def parse_row(row):
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64),
        "mean_rgb": tf.VarLenFeature(tf.float32),
        "mean_audio": tf.VarLenFeature(tf.float32)
    }
    data, _ = tf.parse_single_sequence_example(row, context_features)
    label = multi_hot(tf.sparse.to_dense(data['labels']))
    label.set_shape([CLASS_NUM])
    mean_rgb = tf.sparse.to_dense(data['mean_rgb'])
    mean_rgb.set_shape([RGB_DIM])
    mean_audio = tf.sparse.to_dense(data['mean_audio'])
    mean_audio.set_shape([AUDIO_DIM])
    features = {
        'id': data['id'],
        'mean_rgb': mean_rgb,
        'mean_audio': mean_audio,
    }
    return features, label


def read_dataset(files_pattern, mode, batch_size=128):
    tffiles = tf.io.gfile.glob(files_pattern)
    dataset = tf.data.TFRecordDataset(tffiles)
    dataset = dataset.map(
        parse_row,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(batch_size*10).repeat().batch(batch_size)
    else:
        dataset = dataset.repeat(1).batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
