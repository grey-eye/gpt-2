import argparse
import json
import tensorflow as tf
import os
import model
import encoder
import pandas as pd
import numpy as np

def generate_data(data_dir, len_seq, mode='train'):
    def pad(encoding):
        padlen = len_seq-len(encoding)
        encoding.extend([220]*padlen)
        return encoding

    enc = encoder.get_encoder(args.model_name, args.model_dir)
    if mode == 'train':
        train_path = os.path.join(data_dir.decode("utf-8"), 'train.tsv')
        df = pd.read_csv(train_path, sep='\t', skiprows=1)
        for idx, row in df.iterrows():
            label = np.float32(row[0])
            features = np.array(pad(enc.encode(row[1])[:len_seq]),
                                dtype=np.int32)
            yield features, label


def train_input_fn(generator_fn, data_dir, len_seq, batch_size=32):
    ds = tf.data.Dataset.from_generator(generator_fn, (tf.int32, tf.float32),
                                        None, (data_dir, len_seq))
    ds = ds.shuffle(1000).repeat().batch(batch_size)
    return ds

def get_feature_columns(batch_size, len_seq):
    feature_columns = []
    sentence_embedding = tf.feature_column.numeric_column(key='sentence',
                                                          shape=(batch_size,
                                                                 len_seq))
    feature_columns.append(sentence_embedding)
    return feature_columns

def my_model_fn(features, labels, mode, params):
    hparams = model.default_hparams()
    with open(os.path.join(args.model_dir, args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    features.set_shape([params['batch_size'], params['len_seq']])

    net = model.model(hparams, features)
    logits = net['logits']
    dropout = tf.nn.dropout(logits, keep_prob=0.9)
    avg_logits = tf.math.reduce_mean(dropout, axis=1)
    predictions = tf.layers.dense(avg_logits, 1)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=predictions)
    metrics = {'mse': loss}
    tf.summary.scalar('mse', loss)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'prediction': predictions}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metrics_ops=metrics)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def run_classifier(data_dir, batch_size, len_seq):
    classifier = tf.estimator.Estimator(
        model_fn = my_model_fn,
        model_dir = 'modeldir',
        params={'batch_size': batch_size, 'len_seq': len_seq})
    classifier.train(lambda: train_input_fn(generate_data, data_dir, len_seq,
                                            batch_size),
                     steps=1000)

if __name__ == '__main__':
    #tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to data ' + \
                        'directory containing train, dev and test files')
    parser.add_argument('--len_seq', required=False, type=int, default=40,
                        help='Input sequence length')
    parser.add_argument('--model_dir', required=True, help='Path to ' + \
                        'directory containing model')
    parser.add_argument('--model_name', required=False, default='117M',
                        help='Name of model')
    parser.add_argument('--batch_size', required=False, default=8,
                        help='Sets the batch size', type=int)

    args = parser.parse_args()
    ds = train_input_fn(generate_data, args.data_dir, args.len_seq)
    it = ds.make_one_shot_iterator()
    run_classifier(args.data_dir, args.batch_size, args.len_seq)
