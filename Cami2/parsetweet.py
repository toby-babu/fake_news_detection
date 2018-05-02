import datetime

import nltk
from nltk.corpus import stopwords
import collections
import json
from pprint import pprint
from dateutil import parser
import calendar
import Config
import pickle
import numpy as np
from gensim.models import Doc2Vec
import tensorflow as tf
import numpy as np
import Config

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,  embedding_array):
        # Placeholders for input, output and dropout
        self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
        self.input_x = tf.placeholder(tf.int32, [None, Config.no_intervals_event], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, Config.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.train_embed = tf.nn.embedding_lookup(self.embeddings, self.input_x, None)
        self.embedded_chars_expanded = tf.expand_dims(self.train_embed, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(Config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, Config.embedding_dim, 1, Config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[Config.num_filters]), name="b")
                conv = tf.nn.conv2d( self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, Config.no_intervals_event - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = Config.num_filters * len(Config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat, Config.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, Config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[Config.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + Config.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



def genDictionaries(event_data, is_train):
    word_array = []
    for item in event_data:
        item['tokenized_words'] = nltk.word_tokenize(item['text'])
        tokens = [w.lower() for w in item['tokenized_words']]

        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        item['tokenized_words'] = words
        for token in item['tokenized_words']:
            word_array.append(token)

    if is_train:
        index = 0
        wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
        wordCount.extend(collections.Counter(word_array))
        for word in wordCount:
            wordDict[word] = index
            index += 1


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]

def genWordIdIntervals(time_interval):
    word_id_intervals = []
    for item in time_interval:
        interval = []
        for word in item['tokenized_words']:
            interval.append(getWordID(word))
        word_id_intervals.append(interval)
    if len(word_id_intervals) < 20:
        current_len = len(word_id_intervals)
        for i in xrange(current_len, 20):
            word_id_intervals.append([getWordID(Config.NULL)])
    return word_id_intervals

def load_embeddings(filename, train_sent, dev_sent, test_sent, final_index):
    #dictionary, word_embeds = pickle.load(open(filename, 'rb'))
    word_embeds = Doc2Vec.load(filename)
    #total_len = len(train_sent) + len(test_sent)

    embedding_array = np.zeros((final_index, Config.embedding_dim))
    for item in train_sent:
        #current_line = item.split()
        for para in item:
            if para[1] == 0:
                index = 'TRAIN_NEG_' + str(para[0])
            else:
                index = 'TRAIN_POS_' + str(para[0])
            embedding_array[para[2]] = word_embeds.docvecs[index]

    for item in dev_sent:
        for para in item:
            if para[1] == 0:
                index = 'DEV_NEG_' + str(para[0])
            else:
                index = 'DEV_POS_' + str(para[0])
            embedding_array[para[2]] = word_embeds.docvecs[index]

    for item in test_sent:
        for para in item:
            index = 'TEST_' + str(para[0])
            embedding_array[para[2]] = word_embeds.docvecs[index]

    return embedding_array

dir = "./rumdect/"
train_data = open("train_data.txt", "r")
test_data = open("test_data_mixed.txt", "r")
dev_data = open("dev_data.txt", "r")
#events_file = open(dir+"events_used.txt", "r")
#fake_file = open("fake_used_doc2vec_with_id.txt", "a")
#non_fake_file = open("non_fake_used_doc2vec_with_id.txt", "a")
wordDict = {}
total_data = []
total_dev_data = []
train_full_data = train_data.readlines()
test_full_data = test_data.readlines()
dev_full_data = dev_data.readlines()
train_events = []
train_labels = []
test_events = []
test_labels = []
dev_labels = []
dev_events = []

z = 0
for i in xrange(len(dev_full_data)):
    items = dev_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    dev_events.append([int(items[2]), int(labelid[1])])
    z = z + 1
    if labelid[1] == "0":
        dev_labels.append([0.0, 1.0])
    else:
        dev_labels.append([1.0, 0.0])


for i in xrange(len(train_full_data)):
    items = train_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    #train_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")
    train_events.append([int(items[2]), int(labelid[1])])
    z = z + 1
    if labelid[1] == "0":
        train_labels.append([0.0, 1.0])
    else:
        train_labels.append([1.0, 0.0])


for i in xrange(len(test_full_data)):
    items = test_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    #train_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")
    test_events.append([int(items[2]), int(labelid[1])])
    z = z + 1


def get_train_sents(eventIds, final_index, offset1, offset2):
    time_intervals = []
    for event in eventIds:
        interval = []
        if event[1] == 1:
            k = event[0] - offset1
        else:
            k = event[0] - offset2
        k = k * 20
        for m in xrange(k, k + 20):
            interval.append([m, event[1], final_index])
            final_index = final_index + 1
        time_intervals.append(interval)
    return time_intervals, final_index


def get_test_sents(eventIds, final_index, offset):
    time_intervals = []
    for event in eventIds:
        interval = []
        k = event[0] - offset
        k = k * 20
        for m in xrange(k, k + 20):
            interval.append([m, event[1], final_index])
            final_index = final_index + 1
        time_intervals.append(interval)
    return time_intervals, final_index

embed_index = 0
train_sents, embed_index = get_train_sents(train_events, embed_index, 0, 296)
dev_sents, embed_index  = get_train_sents(dev_events, embed_index, 591, 627)
test_sents, embed_index  = get_test_sents(test_events, embed_index, 663)
#dev_sents = get_event_sents(dev_events)
embedding_filename = 'sentiment140.d2v'#'word2vec.model'
embedding_array = load_embeddings(embedding_filename, train_sents, dev_sents, test_sents, embed_index)

train_ids = []
train_sents = np.array(train_sents)
train_ids = train_sents[:,:,0]
dev_sents = np.array(dev_sents)
dev_ids = dev_sents[:,:,0]
print train_ids
print len(embedding_array)




# Build the graph model
# Training
# ==================================================
print("start training...")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=Config.allow_soft_placement,
        log_device_placement=Config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(embedding_array)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)

        # Initialize all variables
        sess.run(tf.initialize_all_variables())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: Config.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """

            fo = open("result.txt", "w")
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy, scores = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.scores],feed_dict)
            y_label = []
            for i in y_batch:
                for x in xrange(2):
                    if (i[x] == 1):
                        y_label.append(x)

            for ind in xrange(len(scores)):
                fo.write(str(scores[ind]) + "," + str(y_label[ind]) + "\n")
            fo.write("\n")
            time_str = datetime.datetime.now().isoformat()
            print("eval{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #write_json(scores)



        # Training loop. For each batch...
        for step in range(Config.num_epochs):
            start = (step * Config.batch_size) % len(train_ids)
            end = ((step + 1) * Config.batch_size) % len(train_ids)
            if end < start:
                start -= end
                end = len(train_ids)
            batch_inputs, batch_labels = train_ids[start:end], train_labels[start:end]
            train_step(batch_inputs, batch_labels)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % Config.evaluate_every == 0:
                # print("\nEvaluation:")
                dev_step(dev_ids, dev_labels)
            if (current_step == Config.stop_step):
                break