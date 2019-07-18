import functools
import argparse
import json
import tensorflow as tf
import os
import model
import encoder
import pandas as pd
import numpy as np

def generate_data(mode):
    def pad(encoding):
        padlen = args.len_seq-len(encoding)
        encoding.extend([220]*padlen)
        return encoding

    enc = encoder.get_encoder(args.base_model_name, args.base_model_dir)
    if mode == 'train':
        path = os.path.join(args.data_dir, 'train.tsv')
    elif mode == 'eval':
        path = os.path.join(args.data_dir, 'dev.tsv')
    else:
        path = os.path.join(args.data_dir, 'test.tsv')
    df = pd.read_csv(path, sep='\t', skiprows=1)
    for idx, row in df.iterrows():
        label = np.float32(row[0])
        features = np.array(pad(enc.encode(row[1])[:args.len_seq]),
                            dtype=np.int32)
        yield features, label


def input_fn(params, mode):
    generator_fn = params['generator_fn']
    ds = tf.data.Dataset.from_generator(generator_fn, (tf.int32, tf.float32),
                                        None, (mode,))
    ds = ds.shuffle(1000).repeat().batch(args.batch_size)
    return ds


def my_model_fn(features, labels, params, mode):
    hparams = model.default_hparams()
    with open(os.path.join(args.base_model_dir, args.base_model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    features.set_shape([args.batch_size, args.len_seq])

    net = model.model(hparams, features)
    logits = net['logits']
    dropout = tf.nn.dropout(logits, keep_prob=0.9)
    avg_logits = tf.math.reduce_mean(dropout, axis=1)
    predictions = tf.layers.dense(avg_logits, 1)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=predictions)
    metrics = {'mse': loss}
    tf.summary.scalar('mse', loss)
    optimizer = params['optimizer']
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'prediction': predictions}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def build_tpu_config():
    if args.use_tpu:
        my_project_name = subprocess.check_output([
        'gcloud','config','get-value','project'])
        my_zone = subprocess.check_output([
            'gcloud','config','get-value','compute/zone'])
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                tpu=[tpu_name],
                zone=my_zone,
                project=my_project_name)
        master = tpu_cluster_resolver.get_master()
    else:
        master = ''

    tpu_config = tf.estimator.tpu.RunConfig(master=master)

    return tpu_config

def train_regressor():

    tpu_config = build_tpu_config()
    optimizer = tf.train.AdamOptimizer()
    if args.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    regressor = tf.estimator.tpu.TPUEstimator(
        model_fn = my_model_fn,
        model_dir = args.model_dir,
        params={
                'optimizer': optimizer,
                'generator_fn': generate_data,
               },
        use_tpu=args.use_tpu,
        config=tpu_config
    )

    regressor.train(functools.partial(input_fn, mode='train'), steps=args.num_steps)
    regressor.evaluate(functools.partial(input_fn, mode='eval'), steps=args.num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to data ' + \
                        'directory containing train, dev and test files')
    parser.add_argument('--len_seq', required=False, type=int, default=40,
                        help='Input sequence length')
    parser.add_argument('--base_model_dir', required=True, help='Path to ' + \
                        'directory containing model')
    parser.add_argument('--base_model_name', required=False, default='117M',
                        help='Name of model')
    parser.add_argument('--batch_size', required=False, default=8,
                        help='Sets the batch size', type=int)
    parser.add_argument('--train', action='store_true',
                        required=False, help='Run training')
    parser.add_argument('--test', action='store_true', required=False)
    parser.add_argument('--num_steps', type=int, required=False, default=1000,
                        help='Number of train batches to run')
    parser.add_argument('--model_dir', type=str, required=False,
                        default='modeldir',
                        help='Output directory for checkpoints')
    parser.add_argument('--use_tpu', action='store_true',
                        required=False, default=False, help='Use TPU')
    parser.add_argument('--tpu_name', required=False, type=str, default=None,
                        help='Name of TPU')

    args = parser.parse_args()

    if args.train:
        train_regressor()
