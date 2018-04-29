import collections
import json
from pprint import pprint
from dateutil import parser
import calendar
import Config
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


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

dir = "./rumdect/"
train_data = open("train_data.txt", "r")
#fake_file = open("fake_used.txt", "a")
#non_fake_file = open("non_fake_used.txt", "a")
wordDict = {}
total_data = []
twitter_full_data = train_data.readlines()
train_events = []
for i in xrange(len(twitter_full_data)):
    items = twitter_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    train_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")

dev_data = open("dev_data.txt", "r")
dev_events = []
dev_full_data = dev_data.readlines()
for i in xrange(len(dev_full_data)):
    items = dev_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    dev_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")

test_data = open("test_data.txt", "r")
test_events = []
test_full_data = test_data.readlines()
for i in xrange(len(test_full_data)):
    items = test_full_data[i].split()
    eventid = items[0].split(':')
    labelid = items[1].split(':')
    test_events.append(str(eventid[1]) + "_" + str(labelid[1]) + ".txt")

def get_event_sents(event):
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
    #total_data.append(time_intervals)
    if len(time_intervals) != 20:
        print "Error"
    return time_intervals


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
        total_data.append(time_intervals)
        if len(time_intervals) != 20:
            print "Error"
            break
    return total_data


train_sents = get_train_sents(train_events)
#test_sents = get_event_sents(test_events[0])
embedding_filename = 'word2vec.model'

embedding_array = load_embeddings(embedding_filename, wordDict)
print(embedding_array)


