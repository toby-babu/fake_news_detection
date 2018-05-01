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

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

dir = "./rumdect/"
train_data = open("full_data.txt", "r")
train_full_data = train_data.readlines()
id_with_index = open("FULL_ID.txt", "a")
wordDict = {}
total_data = []
z = 0
for event in train_full_data:
    items =event.split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    id_with_index.write(items[0] + " " + items[1] + " " + str(z) + "\n")
    z = z + 1
    json_post_dir = dir + "twitterdata/"
    full_dir = json_post_dir + eventid[1] + "_" + labelid[1] + ".txt"
    print full_dir
    data = json.load(open(full_dir))
    # data = json.load(open("./rumdect/twitterdata/E180_1.txt"))

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
        item['t'] = float(item['created_at']) / sum

    # temp = data[len(data) - 1]['t']
    print (data[len(data) - 1]['created_at']) / 20

    j = 0
    time_intervals = []
    interval = {}
    interval['words'] = []
    interval['users'] = []
    current_index_float = 0.0
    current_index = 0
    genDictionaries(data, True)
    #wordid_tweets = genWordIdIntervals(data)
    div_val = len(data) / 20
    div_val_float = float(len(data)) / 20
    div_val_float = div_val_float - div_val
    print(div_val_float)
    for i in xrange(len(data)):
        if current_index >= div_val:
            current_index_float = current_index_float + div_val_float
            if current_index_float >= 1.0:
                current_index_float = current_index_float - 1.0
                interval['words'].extend(collections.Counter(data[i]['tokenized_words']))
                interval['users'].append(data[i]['user']['id'])
                # interval.append(data[i]['text'])
                current_index = 0
                if len(interval['words']) == 0:
                    interval['words'].append(Config.UNKNOWN)
                    interval['users'].append(Config.UNKNOWN)
                time_intervals.append(interval)
                interval['words'] = []
                interval['users'] = []
                continue
            else:
                current_index = 0
                if len(interval['words']) == 0:
                    interval['words'].append(Config.UNKNOWN)
                    interval['users'].append(Config.UNKNOWN)
                time_intervals.append(interval)
                interval['words'] = []
                interval['users'] = []
        interval['words'].extend(collections.Counter(data[i]['tokenized_words']))
        interval['users'].append(data[i]['user']['id'])
        current_index = current_index + 1

    if (current_index_float + 0.000000001) > 1.0:
        time_intervals[len(time_intervals) - 1].extend(collections.Counter(interval))
        interval = []
    if len(interval) > 0:
        time_intervals.append(interval)

    if len(time_intervals) < 20:
        cur_len = len(time_intervals)
        for q in xrange(cur_len, 20):
            time_intervals.append(Config.UNKNOWN)
    print "Total Number of intervals", len(time_intervals)
    total_data.append(time_intervals)

print len(total_data)
'''
full_sentences = open("FULL_SENTENCES_FOR_DOC2VEC.txt", "a")

for event in total_data:
    for interval in event:
        for word in interval:
            full_sentences.write(word.encode('utf8') + " ")
        full_sentences.write("\n")

print len(total_data)
'''