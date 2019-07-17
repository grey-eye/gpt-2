import argparse
import json
import tensorflow as tf
import os
import model
import encoder
import pandas as pd
import numpy as np


class Regressor():


    def __init__(self,
                 data_dir,
                 len_seq,
                 base_model_dir,
                 base_model_name,
                 batch_size,
                 num_steps,
                 model_dir,
                 use_tpu,
                 tpu_name
                ):
        self.data_dir = data_dir
        self.len_seq = len_seq
        self.base_model_dir = base_model_dir
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.model_dir = model_dir
        self.use_tpu = use_tpu
        self.tpu_name = tpu_name


    def generate_data(self, mode):
        def pad(encoding):
            padlen = self.len_seq-len(encoding)
            encoding.extend([220]*padlen)
            return encoding

        enc = encoder.get_encoder(self.model_name, self.model_dir)
        if mode == 'train':
            path = os.path.join(self.data_dir.decode("utf-8"), 'train.tsv')
        elif mode == 'eval':
            path = os.path.join(self.data_dir.decode("utf-8"), 'dev.tsv')
        else:
            path = os.path.join(self.data_dir.decode("utf-8"), 'test.tsv')
        df = pd.read_csv(path, sep='\t', skiprows=1)
        for idx, row in df.iterrows():
            label = np.float32(row[0])
            features = np.array(pad(enc.encode(row[1])[:self.len_seq]),
                                dtype=np.int32)
            yield features, label


    def input_fn(self, params, generator_fn, mode):
        ds = tf.data.Dataset.from_generator(generator_fn, (tf.int32, tf.float32),
                                            None, (self.data_dir, self.len_seq, mode))
        ds = ds.shuffle(1000).repeat().batch(self.batch_size)
        return ds


    def my_model_fn(self, features, labels, params):
        hparams = model.default_hparams()
        with open(os.path.join(args.model_dir, args.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        features.set_shape([self.batch_size, self.len_seq])
        mode = params['mode']

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
            return tf.estimator.EstimatorSpec(mode, loss=loss,
                                              eval_metrics_ops=metrics)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def build_tpu_config(self):
        if self.use_tpu:
            my_project_name = subprocess.check_output([
            'gcloud','config','get-value','project'])
            my_zone = subprocess.check_output([
                'gcloud','config','get-value','compute/zone'])
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu=[self.tpu_name],
                    zone=my_zone,
                    project=my_project_name)
            master = tpu_cluster_resolver.get_master()
        else:
            master = ''

        tpu_config = tf.estimator.tpu.RunConfig(master=master)

        return tpu_config

    def train_regressor(self):

        tpu_config = self.build_tpu_config()
        optimizer = tf.train.AdamOptimizer()
        if self.use_tpu:
          optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        regressor = tf.estimator.tpu.TPUEstimator(
            model_fn = self.my_model_fn,
            model_dir = self.model_dir,
            params={'mode': 'train',
                    'optimizer': optimizer},
            use_tpu=self.use_tpu,
            config=tpu_config
        )

        input_params = {'batch_size': self.batch_size}

        regressor.train(lambda: self.input_fn(input_params, self.generate_data, 'train'), steps=self.num_steps)
        regressor.evaluate(lambda: self.input_fn(input_params, self.generate_data, 'eval'), steps=self.num_steps)

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
    regressor = Regressor(args.data_dir,
                          args.len_seq,
                          args.base_model_dir,
                          args.base_model_name,
                          args.batch_size,
                          args.num_steps,
                          args.model_dir,
                          args.use_tpu,
                          args.tpu_name
                         )
    if args.train:
        regressor.train_regressor()
