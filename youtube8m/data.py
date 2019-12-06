import tensorflow as tf

def parse_row(row):
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64),
        "mean_rgb": tf.VarLenFeature(tf.float32),
        "mean_audio": tf.VarLenFeature(tf.float32)
    }
    data, _ = tf.parse_single_sequence_example(row, context_features)
    label = data.pop('labels')
    return data, label


def read_dataset(files_pattern, mode, batch_size=128):
    tffiles = tf.io.gfile.glob(files_pattern)
    dataset = tf.data.TFRecordDataset(tffiles).map(parse_row)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(batch_size*10).repeat().batch(batch_size)
    else:
        dataset = dataset.repeat(1).batch(batch_size)
    return dataset
