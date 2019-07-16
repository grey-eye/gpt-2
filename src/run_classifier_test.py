import argparse
import tensorflow as tf
import os
import model
import encoder
import pandas as pd
import numpy as np
import json

def generate_data(data_dir, len_seq, mode='train'):
    def pad(encoding):
        padlen = len_seq-len(encoding)
        encoding.extend([220]*padlen)
        return encoding

    enc = encoder.get_encoder(args.model_name, args.model_dir)
    if mode == 'train':
        train_path = os.path.join(data_dir, 'train.tsv')
        df = pd.read_csv(train_path, sep='\t', skiprows=1)
        for idx, row in df.iterrows():
            label = np.float32(row[0])
            features = pad(enc.encode(row[1])[:len_seq])
            yield features, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to data ' + \
                        'directory containing train, dev and test files')
    parser.add_argument('--len_seq', required=False, type=int, default=40,
                        help='Input sequence length')
    parser.add_argument('--model_dir', required=True, help='Path to ' + \
                        'directory containing model')
    parser.add_argument('--model_name', required=False, default='117M',
                        help='Name of model')
    parser.add_argument('--batch_size', required=False, default=1,
                        help='Sets the batch size')

    args = parser.parse_args()
    with tf.Session(graph=tf.Graph()) as sess:
        X = tf.placeholder(tf.int32, shape=(args.batch_size, args.len_seq), name='sequence')
        y = tf.placeholder(tf.float32, shape=(args.batch_size, 1), name='sentiment')
        hparams = model.default_hparams()
        with open(os.path.join(args.model_dir, args.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        net = model.model(hparams, X, past=None, reuse=tf.AUTO_REUSE)
        logits = net['logits']
        dropout = tf.nn.dropout(logits, keep_prob=0.9)
        avg_logits = tf.math.reduce_mean(dropout, axis=1)
        predictions = tf.layers.dense(avg_logits, 1)

        loss_op = tf.losses.mean_squared_error(labels=y, predictions=predictions)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        init_op = tf.initializers.global_variables()

        sess.run(init_op)

        generator = generate_data(args.data_dir, args.len_seq)

        for i in range(100):
            features, label = next(generator)
            features = np.expand_dims(features, axis=0).tolist()
            label = np.reshape(label, (1,1))
            loss, _ = sess.run([loss_op, train_op], feed_dict={X: features,
                                                               y:label})
            print(loss)

