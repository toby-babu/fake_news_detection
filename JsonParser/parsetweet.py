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
        self.input_x = tf.sparse_placeholder(tf.int32, [None, Config.no_intervals_event, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, Config.num_classes], name="input_y")
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
        self.train_embed = tf.nn.embedding_lookup_sparse(self.embeddings, self.input_x, None)
        self.train_embed_shape = tf.shape(self.train_embed)
        self.train_embed = tf.reshape(self.train_embed, [self.train_embed_shape[0], Config.no_intervals_event, -1])
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
        with tf.name_scope("dropout"):
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

def load_embeddings(filename, wordDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict), Config.embedding_dim))
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
            embedding_array[i] = np.random.rand(Config.embedding_dim) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array

dir = "./rumdect/"
train_data = open("train_data.txt", "r")
#fake_file = open("fake_used.txt", "a")
#non_fake_file = open("non_fake_used.txt", "a")
wordDict = {}
total_data = []
total_dev_data = []
train_full_data = train_data.readlines()
train_events = []
train_labels = []
dev_labels = []
for i in xrange(len(train_full_data)):
    items = train_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    train_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")
    if labelid[1] == "0":
        train_labels.append([0.0, 1.0])
    else:
        train_labels.append([1.0, 0.0])

dev_data = open("dev_data.txt", "r")
dev_events = []
dev_full_data = dev_data.readlines()
for i in xrange(len(dev_full_data)):
    items = dev_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    dev_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")
    if labelid[1] == "0":
        dev_labels.append([0.0, 1.0])
    else:
        dev_labels.append([1.0, 0.0])

test_data = open("test_data.txt", "r")
test_events = []
test_full_data = test_data.readlines()
for i in xrange(len(test_full_data)):
    items = test_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    test_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")

def get_event_sents(eventIds):
    for event1 in eventIds:
        json_post_dir = dir+"twitterdata/"
        full_dir = json_post_dir + event1
        print full_dir
        data = json.load(open(full_dir))
        #data = json.load(open("./rumdect/twitterdata/E180_1.txt"))

        pprint(data[0]['created_at'])
        parsed_date = parser.parse(data[0]['created_at'])
        timestamp = calendar.timegm(parsed_date.timetuple())

        data = sorted(data, key=lambda dct: calendar.timegm(parser.parse(dct['created_at']).timetuple()))
        for item in data:
            item['created_at'] = calendar.timegm(parser.parse(item['created_at']).timetuple())

        temp = data[0]['created_at']
        sum = 0
        for item in data:
            item['created_at'] = item['created_at'] - temp
            sum = sum + item['created_at']

        for item in data:
            item['t'] = float(item['created_at'])/sum

        #temp = data[len(data) - 1]['t']
        print (data[len(data) - 1]['created_at'])/20

        j = 0
        time_intervals = []
        interval = []
        current_index_float = 0.0
        current_index = 0
        genDictionaries(data, False)
        wordid_tweets = genWordIdIntervals(data)
        div_val = len(wordid_tweets) / 20
        div_val_float = float(len(wordid_tweets)) / 20
        div_val_float = div_val_float - div_val
        print(div_val_float)
        for i in xrange(len(wordid_tweets)):
            if current_index >= div_val:
                current_index_float = current_index_float + div_val_float
                if current_index_float >= 1.0:
                    current_index_float = current_index_float - 1.0
                    interval.extend(collections.Counter(wordid_tweets[i]))
                    #interval.append(data[i]['text'])
                    current_index = 0
                    time_intervals.append(interval)
                    interval = []
                    continue
                else:
                    current_index = 0
                    time_intervals.append(interval)
                    interval = []
            interval.extend(collections.Counter(wordid_tweets[i]))
            current_index = current_index + 1

        if (current_index_float + 0.000000001) > 1.0:
            time_intervals[len(time_intervals) - 1].extend(collections.Counter(interval))
            interval = []
        if len(interval) > 0:
            time_intervals.append(interval)

        print "Total Number of intervals",len(time_intervals)
        total_dev_data.append(time_intervals)
        if len(time_intervals) != 20:
            print "Error"
    return total_dev_data


def get_train_sents(eventIds):
    for event in eventIds:
        json_post_dir = dir+"twitterdata/"
        full_dir = json_post_dir + event
        print full_dir
        data = json.load(open(full_dir))
        #data = json.load(open("./rumdect/twitterdata/E180_1.txt"))

        pprint(data[0]['created_at'])
        parsed_date = parser.parse(data[0]['created_at'])
        timestamp = calendar.timegm(parsed_date.timetuple())

        data = sorted(data, key=lambda dct: calendar.timegm(parser.parse(dct['created_at']).timetuple()))
        for item in data:
            item['created_at'] = calendar.timegm(parser.parse(item['created_at']).timetuple())

        temp = data[0]['created_at']
        sum = 0
        for item in data:
            item['created_at'] = item['created_at'] - temp
            sum = sum + item['created_at']

        for item in data:
            item['t'] = float(item['created_at'])/sum

        #temp = data[len(data) - 1]['t']
        print (data[len(data) - 1]['created_at'])/20

        j = 0
        time_intervals = []
        interval = []
        current_index_float = 0.0
        current_index = 0
        genDictionaries(data, True)
        wordid_tweets = genWordIdIntervals(data)
        div_val = len(wordid_tweets) / 20
        div_val_float = float(len(wordid_tweets)) / 20
        div_val_float = div_val_float - div_val
        print(div_val_float)
        for i in xrange(len(wordid_tweets)):
            if current_index >= div_val:
                current_index_float = current_index_float + div_val_float
                if current_index_float >= 1.0:
                    current_index_float = current_index_float - 1.0
                    interval.extend(collections.Counter(wordid_tweets[i]))
                    #interval.append(data[i]['text'])
                    current_index = 0
                    #interval = np.array(interval)
                    time_intervals.append(interval)
                    interval = []
                    continue
                else:
                    current_index = 0
                    #interval = np.array(interval)
                    time_intervals.append(interval)
                    interval = []
            interval.extend(collections.Counter(wordid_tweets[i]))
            current_index = current_index + 1

        if (current_index_float + 0.000000001) > 1.0:
            time_intervals[len(time_intervals) - 1].extend(collections.Counter(interval))
            interval = []
        if len(interval) > 0:
            #interval = np.array(interval)
            time_intervals.append(interval)

        print "Total Number of intervals",len(time_intervals)
        #time_intervals = np.array(time_intervals)
        total_data.append(time_intervals)
        if len(time_intervals) != 20:
            print "Error"
            break

    #total_data = np.array(total_data)
    return total_data


train_sents = get_train_sents(train_events)
dev_sents = get_event_sents(dev_events)
embedding_filename = 'word2vec.model'

embedding_array = load_embeddings(embedding_filename, wordDict)
print(embedding_array)

print "Generating Traning Examples"
trainFeats, trainLabels =train_sents, train_labels
print "Done."

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
                cnn.input_x: tf.SparseTensorValue(),
                cnn.input_y: y_batch
            }
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """

            fo = open("result.txt", "w")
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch
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
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]
            train_step(batch_inputs, batch_labels)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % Config.evaluate_every == 0:
                # print("\nEvaluation:")
                dev_step(dev_sents, dev_labels)
            if (current_step == Config.stop_step):
                break


