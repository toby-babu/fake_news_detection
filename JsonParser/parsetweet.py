import collections
import json
from pprint import pprint
from dateutil import parser
import calendar
import Config
import pickle
import numpy as np
import nltk

nltk.download('punkt')
def genDictionaries(time_interval):
    word = []
    for item in time_interval:
        item['tokenized_words'] = nltk.word_tokenize(item['text'])
        #word_array = item['text'].split()
        for token in item['tokenized_words']:
            word.append(token)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    return wordDict

def genWordIdIntervals(time_interval):
    word_id_intervals = []
    for item in time_interval:
        interval = []
        for word in item['tokenized_words']:
            interval.append(wordDict[word])
        word_id_intervals.append(interval)
    if len(word_id_intervals) < 20:
        current_len = len(word_id_intervals)
        for i in xrange(current_len, 20):
            word_id_intervals.append([wordDict[Config.NULL]])
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
weibo_data = open(dir + "events_used.txt", "r")
wordDict = {}
total_data = []
weibo_full_data = weibo_data.readlines()
#for event_line in weibo_full_data:
for i in xrange(len(weibo_full_data)):
    #items = event_line.split(' ')
    #print(weibo_full_data[0])
    items = weibo_full_data[i].split()
    print(items[0])
    eventid = items[0].split(':')
    print(eventid[1])
    labelid = items[1].split(':')
    print(labelid[1])

    json_post_dir = dir+"twitterdata/"
    #json_content = open(json_post_dir + eventid[1]+".json")
    full_dir = json_post_dir + str(eventid[1]) + "_" + str(labelid[1]) +".txt"
    print full_dir
    data = json.load(open(full_dir))
    #data = json.load(open("./rumdect/twitterdata/E180_1.txt"))

    pprint(data[0]['created_at'])
    #print "hello",len(data)
    #test = datetime.datetime.strptime(data[0]['created_at'], "%a %b %d %H:%M:%S %z %Y").timetuple()
    parsed_date = parser.parse(data[0]['created_at'])
    timestamp = calendar.timegm(parsed_date.timetuple())
    #print(parsed_date)
    #print(timestamp)
    #print("Mintime = ", data[0]['created_at'])
    #print("Maxtime = ", data[len(data) - 1]['created_at'])

    data = sorted(data, key=lambda dct: calendar.timegm(parser.parse(dct['created_at']).timetuple()))
    #print("Mintime = ", data[0]['created_at'])
    #print("Maxtime = ", data[len(data) - 1]['created_at'])
    for item in data:
        item['created_at'] = calendar.timegm(parser.parse(item['created_at']).timetuple())

    temp = data[0]['created_at']
    sum = 0
    for item in data:
        item['created_at'] = item['created_at'] - temp
        sum = sum + item['created_at']



    #print "Mintime = ", data[0]['created_at']
    #print "Maxtime = ", data[len(data) - 1]['created_at']

    for item in data:
        item['t'] = float(item['created_at'])/sum
    #print sum
    #print "Mintime = ", data[0]['t']
    #print "Maxtime = ", data[len(data) - 1]['t']

    temp = data[len(data) - 1]['t']
    print (data[len(data) - 1]['created_at'])/20

    j = 0
    time_intervals = []
    interval = []
    # time_intervals = []
    current_index_float = 0.0
    current_index = 0
    genDictionaries(data)
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

        #to_be_inserted_index = data[i]['created_at']/div_val
        #interval.append(data[i]['text'])
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


embedding_filename = 'word2vec.model'

embedding_array = load_embeddings(embedding_filename, wordDict)
print(embedding_array)

#del(time_intervals[0])


# sorted_data = sorted(data[0], key=lambda dct: Decimal(dct['t']))

#abc = sorted(data, key=operator.itemgetter('t'))
#pprint(data[0]['t'])

