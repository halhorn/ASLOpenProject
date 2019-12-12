from typing import Dict
import tensorflow as tf
from .data import read_dataset, CLASS_NUM, CATEGORY_NUM
from .models import create_model


def multi_hot(indices, class_num):
    return tf.reduce_sum(tf.one_hot(indices, class_num), axis=-2)


def recall_topk(probabilities, labels, top_k, name):
    predicted_topk = tf.math.top_k(probabilities, k=top_k).indices
    predicted_topk_multihot = multi_hot(predicted_topk, labels.shape[-1])
    return tf.metrics.recall(
        labels=labels,
        predictions=predicted_topk_multihot,
        name=name,
    )

def create_metrics(probabilities, labels, params):
    tag_prob, cat_prob = tf.split(probabilities, [CLASS_NUM, CATEGORY_NUM], axis=-1)
    tag_label, cat_label = tf.split(labels, [CLASS_NUM, CATEGORY_NUM], axis=-1)
    metrics = create_sub_metrics(tag_prob, tag_label, params, name='tag')
    metrics.update(create_sub_metrics(cat_prob, cat_label, params, name='category'))
    return metrics
    
def create_sub_metrics(probabilities, labels, params, name):
    threshold = params.get('threshold', 0.5)
    top_k = params.get('top_k', 5)
    predicted_bool = tf.cast(probabilities >= threshold, tf.float32)
    metrics = {
        name + '_accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_bool, name=name + '_accuracy'),
        name + '_recall': tf.metrics.recall(labels=labels, predictions=predicted_bool, name=name + '_recall'),
        name + '_precision': tf.metrics.precision(labels=labels, predictions=predicted_bool, name=name + '_precision'),
        name + '_auc': tf.metrics.auc(labels=labels, predictions=predicted_bool, name=name + '_auc'),
        name + '_predicted_tag_count': tf.metrics.mean(tf.math.count_nonzero(predicted_bool, axis=-1), name=name + '_predicted_tag_count'),
    }
    for k in [top_k // 2, top_k, top_k * 2]:
        metrics[name + '_recall_top{}'.format(k)] = recall_topk(probabilities, labels, top_k=k, name=name + '_recall_top{}'.format(k))
    return metrics


def create_prediction(probabilities, params):
    tag_prob, cat_prob = tf.split(probabilities, [CLASS_NUM, CATEGORY_NUM], axis=-1)
    prediction = create_sub_prediction(tag_prob, params)
    prediction.update(create_sub_prediction(cat_prob, params, prefix='category_'))
    return prediction

def create_sub_prediction(probabilities, params, prefix=''):
    return {
        prefix + 'probabilities': probabilities,
        prefix + 'predicted_topk': tf.math.top_k(probabilities, k=params.get('top_k', 5)).indices,
    }

def model_fn(
    features: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    mode: tf.estimator.ModeKeys,
    params: Dict,
) -> tf.estimator.EstimatorSpec:
    threshold = params.get('threshold', 0.5)
    params['output_dim'] = CLASS_NUM + CATEGORY_NUM
    model = create_model(params)
    logits = model(features['mean_rgb'], features['mean_audio'])
    probabilities = tf.nn.sigmoid(logits)

    loss = None
    train_op = None
    eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(  # [batch, class + category]
            labels=labels,
            logits=logits,
        )
        category_weight = params.get('category_weight', 1.0)
        loss_weights = tf.expand_dims(
            tf.constant([1.0] * CLASS_NUM + [category_weight] * CATEGORY_NUM),
            axis=0
        )
        loss = tf.reduce_mean(cross_entropy * loss_weights)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 0.001))
        # Batch Normalization 用
        update_ops = tf.get_collection(key = tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(control_inputs = update_ops):
            train_op = optimizer.minimize(
                loss,
                global_step=tf.train.get_or_create_global_step()
            )
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = create_metrics(probabilities, labels, params)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=create_prediction(probabilities, params),
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )

def serving_input_fn():
    feature_placeholders = {
        'mean_rgb': tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='serving_mean_rgb'),
        'mean_audio': tf.placeholder(dtype=tf.float32, shape=[None, 128], name='serving_mean_audio'),
    }
    return tf.estimator.export.ServingInputReceiver(
        features=feature_placeholders,
        receiver_tensors=feature_placeholders,
    )


def train_and_evaluate(output_dir: str, params: Dict) -> None:
    tf.compat.v1.summary.FileWriterCache.clear()
    eval_interval_step = params.get('eval_interval_step', 1000)

    config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_checkpoints_steps=eval_interval_step,
        log_step_count_steps=params.get('log_interval_step', 100),
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params,
    )
    batch_size = params.get('batch_size', 512)
    train_spec = tf.estimator.TrainSpec(
        lambda: read_dataset(
            params['train_data_path'],
            tf.estimator.ModeKeys.TRAIN,
            batch_size,
        ),
        max_steps=params.get('train_steps', 30000),
    )
    exporter = tf.estimator.LatestExporter(
        name='exporter', 
        serving_input_receiver_fn=serving_input_fn,
    )
    eval_spec = tf.estimator.EvalSpec(
        lambda: read_dataset(
            params['eval_data_path'],
            tf.estimator.ModeKeys.EVAL,
            batch_size,
        ),
        exporters=exporter,
        start_delay_secs=params.get('eval_delay_sec', 60),
        throttle_secs=10,
    )

    # Run train_and_evaluate loop
    tf.estimator.train_and_evaluate(
        estimator=estimator, 
        train_spec=train_spec, 
        eval_spec=eval_spec,
    )
