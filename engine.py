import serial
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def determineOccupancy(x):
    x_scaled = scaler_train.transform(np.array(x).reshape(1, -1))
    clas_pred = classifier.predict(x_scaled)
    return clas_pred

path = r'C:\Users\adria\pomiary'

header = ['Module-44','Power-44','Module-45','Power-45','Module-46','Power-46','Value']

df = pd.read_csv("./Data.csv", sep=";",header=None,names=header)
df=pd.DataFrame(df)
data = df.iloc[: , :-1]
target = df['Value']
features_train, features_test, labels_train, labels_test = train_test_split(data,target, train_size=0.99,random_state=1)
print(features_test)

scaler_train = preprocessing.StandardScaler().fit(features_train)
scaler_test = preprocessing.StandardScaler().fit(features_test)

features_train = scaler_train.transform(features_train)
features_test = scaler_test.transform(features_test)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(features_train, labels_train)

ser = serial.Serial('/dev/cu.usbserial-A100NJNH', baudrate=9600, timeout=1)

for i in range(1,10):
    data = ser.readline()

new_data = [0,1,2,3,4,5]
old_data = [0,0,0,0,0]

dataToJson = {
    "nodeID": "0",
    "timeStamp": datetime.now(),
    "xAxis": "0",
    "yAxis": "0",
    "zAxis": "0",
    "battVoltage": "0",
    "boardVoltage": "0",
    "battCurrent": "0",
    "battTemp": "0",
    "radioRSSI": "0",
    "radioTransmitLevel":"0",
    "boostEnable": "0",
    "rstSrc": "0",
    "debug": "0"
}


while(1):
    data = ser.readline().strip()
    data1 = data.decode().split(';')
    dataToJson["nodeID"] = data1[1]
    dataToJson["timeStamp"] = datetime.now()
    dataToJson["xAxis"] = data1[3]
    dataToJson["yAxis"] = data1[4]
    dataToJson["zAxis"] = data1[5]
    dataToJson["battVoltage"] = data1[6]
    dataToJson["boardVoltage"] = data1[7]
    dataToJson["battCurrent"] = data1[8]
    dataToJson["battTemp"] = data1[9]
    dataToJson["radioRSSI"] = data1[10]
    dataToJson["radioTransmitLevel"] = data1[11]
    dataToJson["boostEnable"] = data1[12]
    dataToJson["rstSrc"] = data1[13]
    dataToJson["debug"] = data1[14]

    res = requests.post('http://localhost:5000/smart_parking/spot/'+dataToJson["nodeID"], json=json.dumps(dataToJson,default=str))

    if(data1[1] == "44" and new_data[0]!= (abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5])))):
        new_data[0] = abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5]))
        new_data[1] = float(data1[3])**2 + float(data1[4])**2 + float(data1[5])**2
        parkingOccupancu = determineOccupancy(new_data)
        parkingOccupancu = parkingOccupancu[0].split('_')
        dJson={"timeStamp":datetime.now(),"44":parkingOccupancu[0],"45":parkingOccupancu[1],"46":parkingOccupancu[2]}
        res = requests.post('http://localhost:5000/smart_parking', json=json.dumps(dJson,default=str))
        print(new_data)
        print(parkingOccupancu)
    elif(data1[1] == "45" and new_data[2]!= (abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5])))):
        new_data[2] = abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5]))
        new_data[3] = float(data1[3]) ** 2 + float(data1[4]) ** 2 + float(data1[5]) ** 2
        parkingOccupancu = determineOccupancy(new_data)
        parkingOccupancu = parkingOccupancu[0].split('_')
        dJson={"timeStamp":datetime.now(),"44":parkingOccupancu[0],"45":parkingOccupancu[1],"46":parkingOccupancu[2]}
        res = requests.post('http://localhost:5000/smart_parking', json=json.dumps(dJson,default=str))
        print(new_data)
        print(parkingOccupancu)
    elif (data1[1] == "46" and new_data[4]!= (abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5])))):
        new_data[4] = abs(float(data1[3])) + abs(float(data1[4])) + abs(float(data1[5]))
        new_data[5] = float(data1[3]) ** 2 + float(data1[4]) ** 2 + float(data1[5]) ** 2
        parkingOccupancu = determineOccupancy(new_data)
        parkingOccupancu = parkingOccupancu[0].split('_')
        dJson={"timeStamp":datetime.now(),"44":parkingOccupancu[0],"45":parkingOccupancu[1],"46":parkingOccupancu[2]}
        res = requests.post('http://localhost:5000/smart_parking', json=json.dumps(dJson,default=str))
        print(new_data)
        print(parkingOccupancu)

ser.close()
