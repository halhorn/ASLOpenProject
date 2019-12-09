from typing import Dict
from data import read_dataset


def create_model(params):
    model_map = {
        'linear': LinearModel,
        'dnn': DNNModel,
    }
    print('----------------------------------')
    print('Model:', params['model'])
    return model_map[params['model']](params)


def multi_hot(indices):
    return tf.reduce_sum(tf.one_hot(indices, CLASS_NUM), axis=-2)


def recall_topk(probabilities, labels, top_k):
    predicted_topk = tf.math.top_k(probabilities, k=top_k).indices
    predicted_topk_multihot = multi_hot(predicted_topk)
    return tf.metrics.recall(
        labels=labels,
        predictions=predicted_topk_multihot,
    )


def create_metrics(probabilities, labels, params):
    threshold = params.get('threshold', 0.5)
    top_k = params.get('top_k', 5)
    predicted_bool = tf.cast(probabilities >= threshold, tf.float32)
    metrics = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_bool),
        'recall': tf.metrics.recall(labels=labels, predictions=predicted_bool),
        'precision': tf.metrics.precision(labels=labels, predictions=predicted_bool),
        'auc': tf.metrics.auc(labels=labels, predictions=predicted_bool),
        'predicted_tag_count': tf.metrics.mean(tf.math.count_nonzero(predicted_bool, axis=-1)),
    }
    for k in [top_k // 2, top_k, top_k * 2]:
        metrics['recall_top{}'.format(k)] = recall_topk(probabilities, labels, top_k=k)
    return metrics


def model_fn(
    features: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    mode: tf.estimator.ModeKeys,
    params: Dict,
) -> tf.estimator.EstimatorSpec:
    threshold = params.get('threshold', 0.5)
    model = create_model(params)
    logits = model(features['mean_rgb'], features['mean_audio'])
    probabilities = tf.nn.sigmoid(logits)

    loss = None
    train_op = None
    eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # Convert string label to int
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
        )
        loss = tf.reduce_mean(cross_entropy)

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
        predictions={
            'probabilities': probabilities,
            'predicted_topk': tf.math.top_k(probabilities, k=params.get('top_k', 5)).indices,
        },
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
    tf.summary.FileWriterCache.clear()
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
        max_steps=params.get('train_steps', 10000),
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
