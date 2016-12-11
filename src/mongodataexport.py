from pymongo import MongoClient
import numpy as np

client = MongoClient()
db = client.test
wifi = db.wifi

BSSID = set()

num = 0

for i in wifi.find() :
	for j in i['data'] :
		BSSID.add(j['BSSID'])
		
print(len(BSSID))
		
index = {}
zones = {00:0,01:1,02:2,03:3,10:4,13:5,20:6,23:7,40:8,41:9,42:10,43:11,50:12,53:13,60:14,63:15,70:16,71:17,72:18,73:19,80:20,83:21,90:22,93:23,100:24,101:25,102:26,103:27}
print (zones)
for i in range(len(BSSID)) :
	index[BSSID.pop()] = i
		
for i in wifi.find() :
	for j in i['data'] :
		BSSID.add(j['BSSID'])
		num+=1
		
x_all = np.zeros((wifi.count(),260))
y_all = np.zeros((wifi.count(),1))

cnt = 0
for i in wifi.find() :
	for j in i['data'] :	
		#x_temp[index[j['BSSID']]] = np.float(1+(35+j['level'])/64.0)
		x_all[cnt,index[j['BSSID']]] = j['level']
	y_all[cnt,0] = zones[i['zone']]
	#y_temp[zones[(FIND[i])['zone']]] = np.float(1)
	#y_all.append(y_temp)
	cnt+=1

print(cnt)
#f = open('data','w')
#for i in x_all :
#	f.write(np.array_str(i))
#f.close()
np.savetxt('dataY.csv',y_all,delimiter=',')
np.savetxt('dataX.csv',x_all,delimiter=',')
