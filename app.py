from flask import Flask, redirect, url_for, render_template, request, jsonify
import json

app = Flask(__name__)

dataFromSensors = {
    "44": {
            "nodeID":"44",
            "timeStamp": 0,
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
        },
    "45": {
            "nodeID":"45",
            "timeStamp": 0,
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
        },
    "46": {
            "nodeID":"46",
            "timeStamp": 0,
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
}

dataFromSensors = json.dumps(dataFromSensors)

dataFromSensors = json.loads(dataFromSensors)

occupancy = {
    "timeStamp": 0,
    "44": 0,
    "45": 0,
    "46": 0
}

occupancy = json.dumps(occupancy)
occupancy = json.loads(occupancy)
@app.route('/')
def index():
    return render_template('mainPage.html')

@app.route('/smart_parking', methods=['GET'])
def smart_parkingGET():
    return jsonify(occupancy)

@app.route('/smart_parking', methods=['POST'])
def smart_parking():
    content = json.loads(request.json)
    for x in content:
        occupancy[str(x)] = content[x]
    return jsonify(request.json)

@app.route('/smart_parking/spot/<node_id>',methods=['GET'])
def spotGET(node_id):
    return jsonify(dataFromSensors[str(node_id)])


@app.route('/smart_parking/spot/<node_id>', methods=['POST'])
def spot(node_id):
    content = json.loads(request.json)
    for x in content:
        dataFromSensors[str(node_id)][x] = content[x]

    return jsonify({"node_id":node_id})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
