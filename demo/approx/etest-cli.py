# -*- coding: utf-8 -*-

import logging
import sys

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, classification_report

from propagation.models import GaussianFields
from propagation.solvers import ExactSolver, JacobiSolver
from propagation.visualization import HintonDiagram

# tf.enable_eager_execution()


def main(argv):
    # taking a set of tweets as example to build a similarity matrix
    # tweet_list = pd.read_csv('uber_tweets.csv', sep=',', header=0)['text'].tolist()
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(tweet_list)
    # print(tfidf_matrix.shape)
    # print(tfidf_matrix[:2, :])

    np.random.seed(seed=13)

    # currently and error of shape: Incompatible shapes: [2,989,989] vs. [4,989,989]

    X = pickle.load(open('av_feature_matrix.pkl', 'rb'))
    # print(X)
    y_true = pickle.load(open('av_target_y.pkl', 'rb'))
    y_positives = np.where(y_true == 1)[0]
    y_negatives = np.where(y_true == -1)[0]
    ratio = 0.002  # 0.2% = 1 element
    nb_labeled_nodes = int(y_true.shape[0] * ratio)
    print('# Labeled nodes {}'.format(nb_labeled_nodes))
    y_positives_sample = np.random.choice(y_positives, size=nb_labeled_nodes, replace=False)
    y_negatives_sample = np.random.choice(y_negatives, size=nb_labeled_nodes, replace=False)

    # distance_matrix = pairwise_distances(X, metric='cosine', n_jobs=2)
    # print(distance_matrix)
    # W = 1.0 - distance_matrix
    W = cosine_similarity(X, X)
    print(W.shape)
    print(W)

    batch_size = 2
    nb_nodes = W.shape[0]
    # print(nb_nodes)

    l = np.zeros(shape=[nb_nodes], dtype='int8')
    y = np.zeros(shape=[nb_nodes], dtype='float32')

    # vector l contains 1's in positions labeled and 0 elsewhere
    # vector y contains the labels for each position
    l[y_positives_sample] = 1  # it was just index 0
    y[y_positives_sample] = 1.0

    l[y_negatives_sample] = 1  # it was just index 1
    y[y_negatives_sample] = -1.0

    mu, eps = 1.0, 1e-8

    batch_l = np.zeros(shape=[batch_size, nb_nodes], dtype='float32')
    batch_y = np.zeros(shape=[batch_size, nb_nodes], dtype='float32')
    batch_W = np.zeros(shape=[batch_size, nb_nodes, nb_nodes], dtype='float32')

    batch_l[0, :] = l.reshape(nb_nodes)
    batch_y[0, :] = y.reshape(nb_nodes)
    batch_W[0, :, :] = W

    batch_l[1, :] = l.reshape(nb_nodes)
    batch_y[1, :] = y.reshape(nb_nodes)
    batch_W[1, :, :] = -W

    l_ph = tf.placeholder('float32', shape=[None, None], name='l')
    y_ph = tf.placeholder('float32', shape=[None, None], name='y')
    mu_ph = tf.placeholder('float32', [None], name='mu')
    eps_ph = tf.placeholder('float32', [None], name='eps')
    W_ph = tf.placeholder('float32', shape=[None, None, None], name='W')

    # solver = ExactSolver()
    solver = JacobiSolver()
    model = GaussianFields(l=l_ph, y=y_ph,
                           mu=mu_ph, W=W_ph, eps=eps_ph,
                           solver=solver)
    f_star = model.minimize()

    feed_dict = {
        l_ph: batch_l,
        y_ph: batch_y,
        W_ph: batch_W,
        mu_ph: np.array([mu] * batch_size),
        eps_ph: np.array([eps] * batch_size),
    }

    with tf.Session() as session:
        f_value = session.run(f_star, feed_dict=feed_dict)
        f_value_0 = f_value[0, :]
        # print(f_value_0)
        f_label = [1 if f_value_0[i] > 0 else -1 for i in range(len(f_value_0))]
        print(accuracy_score(y_true, f_label))
        print(classification_report(y_true, f_label, digits=4))
        # hd = HintonDiagram()
        # print(hd(f_value_0))   #.reshape((nb_nodes, nb_nodes))))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
