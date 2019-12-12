'''
frame level のデータ
'''

import tensorflow as tf
from .category_util import get_id_category_id_table, CATEGORIES, CATEGORY_NUM

CLASS_NUM = 3862
RGB_DIM = 1024
AUDIO_DIM = 128
MAX_LEN = 300
id_category_id_table = get_id_category_id_table(CLASS_NUM)

def multi_hot(indices, class_num):
    return tf.reduce_sum(tf.one_hot(indices, class_num), axis=-2)

def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    '''
    8bit に圧縮されているデータを float32 に戻します
    see: https://github.com/linrongc/youtube-8m/blob/master/utils.py#L28
    '''
    feat_vector = tf.cast(feat_vector, tf.float32)
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def adjust_length(feature, max_len=MAX_LEN):
    shape = tf.unstack(tf.shape(feature))
    f = tf.reshape(feature, [-1, shape[-2], shape[-1]])
    f = f[:, :max_len, :]
    l = tf.unstack(tf.shape(f))[1]
    f = tf.cond(
        l < max_len,
        lambda: tf.pad(f, [[0, 0], [max_len - l, 0], [0, 0]]),
        lambda: f
    )
    shape[-2] = max_len
    return tf.reshape(f, shape), l

def decode(feature, dim):
    '''
    バイト列になっているフィーチャーを float32 の配列にして返します。
    '''
    f = tf.reshape(
        tf.decode_raw(feature, tf.uint8),
        [-1, dim],  # [len, dim]
    )
    f = dequantize(f)
    f, length = adjust_length(f)
    f.set_shape([MAX_LEN, dim])
    return f, length

def parse_row(row):
    context_features = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
    }
    sequence_features = {
        "rgb": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        "audio": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
    }
    context_data, sequence_data = tf.parse_single_sequence_example(row, context_features, sequence_features)
    label_ids = tf.sparse.to_dense(context_data['labels'])
    label = multi_hot(label_ids, CLASS_NUM)
    category, _ = tf.unique(tf.gather(id_category_id_table, label_ids))
    category = multi_hot(category, CATEGORY_NUM)
    label = tf.concat([label, category], axis=-1)
    label.set_shape([CLASS_NUM + CATEGORY_NUM])
    
    rgb, rgb_len = decode(sequence_data['rgb'], RGB_DIM)
    audio, audio_len = decode(sequence_data['audio'], AUDIO_DIM)
    features = {
        'id': context_data['id'],
        'rgb': rgb,
        'rgb_len': rgb_len,
        'audio': audio,
        'audio_len': audio_len,
    }
    return features, label


def read_dataset(files_pattern, mode, batch_size=128):
    tffiles = tf.data.Dataset.list_files(files_pattern)
    dataset = tffiles.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
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
