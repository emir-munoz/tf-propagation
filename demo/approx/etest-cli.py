# -*- coding: utf-8 -*-

import logging
import sys
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from propagation.models import GaussianFields
from propagation.solvers import ExactSolver, JacobiSolver
from propagation.visualization import HintonDiagram


def main(argv):
    # taking a set of tweets as example to build a similarity matrix
    tweet_list = pd.read_csv('uber_tweets.csv', sep=',', header=0)['text'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweet_list)
    print(tfidf_matrix.shape)
    print(tfidf_matrix[:2, :])
    W = 1 - pairwise_distances(tfidf_matrix, metric='cosine', n_jobs=2)
    print(W.shape)
    print(W)

    nb_nodes = W.shape[0]
    print(nb_nodes)
    nb_rows = nb_cols = int(math.sqrt(nb_nodes))
    print(nb_rows, nb_cols)

    l = np.zeros(shape=[nb_rows, nb_cols], dtype='int8')
    y = np.zeros(shape=[nb_rows, nb_cols], dtype='float32')

    l[0, 0] = 1
    y[0, 0] = 1.1

    l[nb_rows - 1, nb_cols - 1] = 1
    y[nb_rows - 1, nb_cols - 1] = - 1.0

    mu, eps = 1.0, 1e-8

    batch_l = np.zeros(shape=[2, nb_nodes], dtype='float32')
    batch_y = np.zeros(shape=[2, nb_nodes], dtype='float32')
    batch_W = np.zeros(shape=[2, nb_nodes, nb_nodes], dtype='float32')

    batch_l[0, :] = l.reshape(nb_nodes)
    batch_y[0, :] = y.reshape(nb_nodes)
    batch_W[0, :, :] = W

    batch_l[1, :] = l.reshape(nb_nodes)
    batch_y[1, :] = y.reshape(nb_nodes)
    batch_W[1, :, :] = - W

    l_ph = tf.placeholder('float32', shape=[None, None], name='l')
    y_ph = tf.placeholder('float32', shape=[None, None], name='y')

    mu_ph = tf.placeholder('float32', [None], name='mu')
    eps_ph = tf.placeholder('float32', [None], name='eps')

    W_ph = tf.placeholder('float32', shape=[None, None, None], name='W')

    solver = ExactSolver()
    # solver = JacobiSolver()
    model = GaussianFields(l=l_ph, y=y_ph,
                           mu=mu_ph, W=W_ph, eps=eps_ph,
                           solver=solver)

    f_star = model.minimize()

    feed_dict = {
        l_ph: batch_l, y_ph: batch_y, W_ph: batch_W,
        mu_ph: np.array([mu] * 2),
        eps_ph: np.array([eps] * 2),
    }

    with tf.Session() as session:
        hd = HintonDiagram()

        f_value = session.run(f_star, feed_dict=feed_dict)

        f_value_0 = f_value[0, :]
        print(hd(f_value_0.reshape((nb_rows, nb_cols))))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
