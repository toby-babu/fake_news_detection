import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """

            self.train_inputs = tf.placeholder(tf.int32, [None, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.float32, [None, parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32, [Config.n_Tokens])

            weights_input = tf.Variable(
                tf.truncated_normal(shape=[Config.n_Tokens * Config.embedding_size, Config.hidden_size],
                                    stddev=math.sqrt(1.0 / Config.hidden_size)))
            weights_output = tf.Variable(
                tf.truncated_normal(shape=[Config.hidden_size_2, parsing_system.numTransitions()],
                                    stddev=math.sqrt(1.0 / parsing_system.numTransitions())))
            weights_input_2 = tf.Variable(
                tf.truncated_normal(shape=[Config.hidden_size, Config.hidden_size_2],
                                    stddev=math.sqrt(1.0 / Config.hidden_size_2)))
            #weights_input_3 = tf.Variable(
                #tf.truncated_normal(shape=[Config.hidden_size_2, Config.hidden_size_3],
                                    #stddev=math.sqrt(1.0 / Config.hidden_size_3)))
            biases_input = tf.Variable(tf.zeros([Config.hidden_size, ]))
            biases_input_2 = tf.Variable(tf.zeros([Config.hidden_size_2, ]))
            #biases_input_3 = tf.Variable(tf.zeros([Config.hidden_size_3, ]))

            train_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embeddings = tf.reshape(train_embeddings, [-1, Config.n_Tokens * Config.embedding_size])
            pred = self.forward_pass(train_embeddings, weights_input, biases_input, weights_output, weights_input_2, biases_input_2)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(self.train_labels, 1),
                                                                       logits=pred)
            weights_input_loss = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(train_embeddings) + tf.nn.l2_loss(weights_input_2) + tf.nn.l2_loss(biases_input_2)
            lambdaval = tf.multiply(weights_input_loss, Config.lam)
            self.loss = tf.add(self.loss, lambdaval)
            self.loss = tf.reduce_mean(self.loss)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output, weights_input_2, biases_input_2)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)

    def forward_pass(self, embed, weights_input, biases_input, weights_output, weights_input_2, biases_input_2):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """

        embed_wt = tf.matmul(embed, weights_input)
        total_term = tf.add(embed_wt, biases_input)
        cube_activation = tf.pow(total_term, 3)
        #pred = tf.matmul(cube_activation, weights_output)

        embed_wt_2 = tf.matmul(cube_activation, weights_input_2)
        total_term_2 = tf.add(embed_wt_2, biases_input_2)
        #cube_activation_2 = tf.pow(total_term_2, 3)
        pred_2 = tf.matmul(total_term_2, weights_output)

        #embed_wt_3 = tf.matmul(cube_activation_2, weights_input_3)
        #total_term_3 = tf.add(embed_wt_3, biases_input_3)
        #cube_activation_3 = tf.pow(total_term_3, 3)
        #pred_3 = tf.matmul(total_term_3, weights_output)


        #relu = tf.nn.relu(total_term)
        #pred = tf.matmul(relu, weights_output)
        #tanh = tf.nn.tanh(total_term)
        #pred = tf.matmul(tanh, weights_output)
        #sigmoid = tf.nn.sigmoid(total_term)
        #pred = tf.matmul(sigmoid, weights_output)
        return pred_2


def genDictionaries(time_interval):
    word = []
    for item in time_interval:
        word_array = item['text'].split()
        for token in word_array:
            word.append(token)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    return wordDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):
    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features = []
    pos_tags = []
    arc_labels = []

    # features.append(c.getLeftChild(1, c.stack[0]))
    for x in xrange(0, 2):
        features.append(c.getLeftChild(c.getStack(x), 1))
        features.append(c.getLeftChild(c.getStack(x), 2))
        features.append(c.getRightChild(c.getStack(x), 1))
        features.append(c.getRightChild(c.getStack(x), 2))
        features.append(c.getLeftChild(c.getLeftChild(c.getStack(x), 1), 1))
        features.append(c.getRightChild(c.getRightChild(c.getStack(x), 1), 1))

    for i in features:
        arc_labels.append(c.getLabel(i))
    for x in xrange(0, 3):
        features.append(c.getStack(x))
    for x in xrange(0, 3):
        features.append(c.getBuffer(x))
    for i in features:
        pos_tags.append(c.getPOS(i))

    pos_ids = []
    label_ids = []
    word_ids = []
    feature_id = []
    for i in pos_tags:
        pos_ids.append(getPosID(i))
    for i in arc_labels:
        label_ids.append(getLabelID(i))
    for i in features:
        word_ids.append(getWordID(c.getWord(i)))

    feature_id.extend(word_ids)
    feature_id.extend(label_ids)
    feature_id.extend(pos_ids)

    return feature_id


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    #trainSents, trainTrees = Util.loadConll('train.conll')
    #devSents, devTrees = Util.loadConll('dev.conll')
    #testSents, _ = Util.loadConll('test.conll')
    genDictionaries(time_intervals[0])

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict)

    '''
    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."
    '''
    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)
