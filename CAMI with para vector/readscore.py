import glob
import json
import os
import numpy as np
'''
f = open("userlist.txt","w")
idlist = []
for filename in glob.iglob('F:/Courses/NLP/proj/codes/corrected_data/*.txt'):
    try:
        jdat = json.load(open(filename))
        for x in jdat:
            newid = x["user"]["id"]
            if newid not in idlist:
                idlist.append(newid)
                f.write(str(newid)+"\n")
    except ValueError, ve:
        print(filename)
u = []
'''
u = []
with open("userlist.txt","r") as userfile:
    users = userfile.readlines()
    u = [int(u.strip()) for u in users]
print(len(u))
adjmat = np.zeros([len(u),len(u)])
print(u)
for filename in glob.iglob('F:/Courses/NLP/proj/codes/corrected_data/*.txt'):
    try:
        jdat = json.load(open(filename))
        i = 0
        print(filename)
        print("num of tweets " + str(len(jdat)))
        for x in jdat:
            newid = x["user"]["id"]
            j = i+1
            ind1 = u.index(newid)
            print(ind1)
            print("newid is " + str(newid))
            while j < len(jdat):
                otherid = jdat[j]["user"]["id"]
                if newid == otherid:
                    j +=1
                    continue
                ind2 = u.index(otherid)
                print("ind2 is "+str(ind2))
                adjmat[ind1][ind2] += 1
                adjmat[ind2][ind1] += 1
                j+=1
            i +=1
        print("event done for "+filename+"\n")
    except ValueError, ve:
        print(filename)
        print(ve)
'''
adjf = open("adjmat.txt","w")
for i in range(len(u)):
    adjf.write(",".join(str(x) for x in adjmat[i])+"\n")
'''
u,s,vh = np.linalg.svd(adjmat,full_matrices=False)
print(s.shape)
en = [x*x for x in s]
torem = []
totalen = sum(en)
curr_en = totalen
toclip = len(s)
print("totalen is "+str(totalen))
for i in range(len(s)-1,-1,-1):
    curr_en -= s[i]*s[i]
    if curr_en>0.9*totalen:
        toclip=i
    else:
        print("curren is " + str(curr_en))
        break

print(toclip)
clipped_vt = vh[:toclip][:]
y = np.matmul(adjmat,np.transpose(clipped_vt))
print(y.shape)
adjf = open("adjmat_lowdim.txt","w")
for i in range(len(s)):
    adjf.write(",".join(str(x) for x in y[i])+"\n")

