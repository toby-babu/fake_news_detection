import json
from decimal import Decimal
from pprint import pprint
import operator

dir = "./rumdect/"
weibo_data = open(dir + "Weibo.txt", "r")
weibo_full_data = weibo_data.readlines()
#for event_line in weibo_full_data:
    #items = event_line.split(' ')
print weibo_full_data[0]
items = weibo_full_data[0].split()
print items[0]
eventid = items[0].split(':')
print eventid[1]

json_post_dir = dir+"Weibo/"
#json_content = open(json_post_dir + eventid[1]+".json")
data = json.load(open(json_post_dir + eventid[1]+".json"))

pprint(data[0]['t'])
print "Mintime = ", data[0]['t']
print "Maxtime = ", data[len(data) - 1]['t']
temp = data[0]['t']
sum = 0
for item in data:
    item['t'] = item['t'] - temp
    sum = sum + item['t']



print "Mintime = ", data[0]['t']
print "Maxtime = ", data[len(data) - 1]['t']

for item in data:
    item['t'] = float(item['t'])/sum
print sum
print "Mintime = ", data[0]['t']
print "Maxtime = ", data[len(data) - 1]['t']
temp = data[len(data) - 1]['t']
print len(data)/20
j = 0
time_intervals = []
interval = []
# time_intervals = []
for i in xrange(len(data)):
    if i % 5 == 0:
        time_intervals.append(interval)
        interval = []
    interval.append(data[i])

time_intervals.append(interval)
del(time_intervals[0])
pprint(time_intervals[0])


# sorted_data = sorted(data[0], key=lambda dct: Decimal(dct['t']))

#abc = sorted(data, key=operator.itemgetter('t'))
#pprint(data[0]['t'])
