import csv
import tensorflow as tf

CLASS_NUM = 3862

RGB_DIM = 1024
AUDIO_DIM = 128

CATEGORIES = [
    '(Unknown)',
    'Arts & Entertainment',
    'Autos & Vehicles',
    'Beauty & Fitness',
    'Books & Literature',
    'Business & Industrial',
    'Computers & Electronics',
    'Finance',
    'Food & Drink',
    'Games',
    'Health',
    'Hobbies & Leisure',
    'Home & Garden',
    'Internet & Telecom',
    'Jobs & Education',
    'Law & Government',
    'News',
    'People & Society',
    'Pets & Animals',
    'Real Estate',
    'Reference',
    'Science',
    'Shopping',
    'Sports',
    'Travel',
]
CATEGORY_NUM = len(CATEGORIES)

def category2id(category_name):
    id_ = CATEGORIES.index(category_name)
    assert id_ >= 0
    return id_

id_category_id_table = [0] * CLASS_NUM
with open('./vocabulary.csv', 'r') as c:
    reader = csv.reader(c)
    next(reader)
    for r in reader:
        id_category_id_table[int(r[0])] = category2id(r[5])

def multi_hot(indices, class_num):
    return tf.reduce_sum(tf.one_hot(indices, class_num), axis=-2)


def parse_row(row):
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64),
        "mean_rgb": tf.VarLenFeature(tf.float32),
        "mean_audio": tf.VarLenFeature(tf.float32)
    }
    data, _ = tf.parse_single_sequence_example(row, context_features)
    label_ids = tf.sparse.to_dense(data['labels'])
    label = multi_hot(label_ids, CLASS_NUM)
    category, _ = tf.unique(tf.gather(id_category_id_table, label_ids))
    category = multi_hot(category, CATEGORY_NUM)
    label = tf.concat([label, category], axis=-1)
    label.set_shape([CLASS_NUM + CATEGORY_NUM])
    
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
