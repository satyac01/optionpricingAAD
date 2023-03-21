import json
import time
import logging

from flask import Flask, jsonify, request , Response
from BlackScholes import BlackScholes
import numpy as np
from PricingModel import *
from datetime import datetime
import pandas as pd

app = Flask("Pricing Application")

instrumentModelMap = {}
size = 8192

@app.route('/model/train/instrument', methods=['POST'])
def trainModelForGivenInstrument():
    if (request.method == 'POST'):
        request_data = request.get_json()
        instrumentId , strikePrice, expiryInYears, spotPrice, volatality =  getRequestParam(request_data)
        response = getRawResponse()
        # resonse_json = getRawJson()
        #
        if modelExist(instrumentId):
            response["data"].append({'instrumentId':instrumentId,'training_data':'Trained Model Already Exist'})

        seed = np.random.randint(0, 10000)
        xTrain,yTrain,dydxTrain,model = generateTrainingDataAndTrainModel(instrumentId,seed,spotPrice,strikePrice, volatality, expiryInYears)

        # Populate Model cache
        populateModelCache(instrumentId,model)

        model_training_data = np.concatenate((xTrain,yTrain,dydxTrain), axis = 1)
        df = pd.DataFrame(model_training_data,columns=['spot','price','differential'])
        #training_data_json=df.to_json(orient="records")
        #
        # resonse_json['instrumentId'] = instrumentId
        # resonse_json['training_data'] = df.to_dict(orient="records")
        sub_response = {}
        sub_response['instrumentId'] = instrumentId
        sub_response['training_data'] = df.to_dict(orient="records")

        response["data"].append(sub_response)


        #return resonse_json
        return jsonify(response)
        #return Response(training_data_json, mimetype='application/json')

@app.route('/model/train/instruments', methods=['POST'])
def trainModelForInstruments():
    if (request.method == 'POST'):
        instrumentList = request.get_json()

        #resonse_json = getRawJson()
        #resonse_json['data'] = {}
        response = getRawResponse()
        seed = np.random.randint(0, 10000)
        for item in instrumentList:
            instrumentId , strikePrice, expiryInYears, spotPrice, volatality =  getRequestParam(item)
            if modelExist(instrumentId):
                response["data"].append({'instrumentId': instrumentId, 'training_data': 'Trained Model Already Exist'})
                continue
            else :
                xTrain,yTrain,dydxTrain,model = generateTrainingDataAndTrainModel(instrumentId,seed,spotPrice,strikePrice, volatality, expiryInYears)
                # Populate Model cache
                populateModelCache(instrumentId, model)

                model_training_data = np.concatenate((xTrain,yTrain,dydxTrain), axis = 1)
                df = pd.DataFrame(model_training_data,columns=['spot','price','differential'])

                response["data"].append({'instrumentId': instrumentId, 'training_data': df.to_dict(orient="records")})

        return jsonify(response)
        #return Response(training_data_json, mimetype='application/json')

@app.route('/model/price/instrument', methods=['POST'])
def getPredictedPriceForInstrument():
    if (request.method == 'POST'):
        request_data = request.get_json()
        instrumentId = request_data['ticker']
        spotPrice = request_data['spotprice']

        spotPriceArr = np.array(spotPrice).reshape([-1, 1])

        pricerModel = instrumentModelMap.get(instrumentId)
        #response_json = getRawJson()
        response = getRawResponse()
        if not pricerModel:
            print("No model found for given Instrument")
            response["data"].append({'instrumentId': instrumentId, 'values': ''})
        else:
            predictions, deltas = predictPriceForInstrument(instrumentId , spotPriceArr, pricerModel)
            values = np.concatenate((spotPriceArr, predictions, deltas), axis=1)
            df = pd.DataFrame(values, columns=['SpotPrice', 'Predicted Price', 'Deltas'])
            print(predictions)
            print(deltas)
            #response_json['instrumentId'] = instrumentId
            #response_json['values'] = df.to_dict(orient="records")
            response["data"].append({'instrumentId': instrumentId, 'values': df.to_dict(orient="records")})


        return jsonify(response)

@app.route('/model/price/instruments', methods=['POST'])
def getPredictedPriceForInstruments():
    if (request.method == 'POST'):
        instrumentList = request.get_json()

        response = getRawResponse()
        for item in instrumentList:
            instrumentId = item['ticker']
            spotPrice = item['spotprice']

            spotPriceArr = np.array(spotPrice).reshape([-1, 1])


            pricerModel = instrumentModelMap.get(instrumentId)
            if not pricerModel:
                print("No Model found for given instrument ", instrumentId)
                response["data"].append({'instrumentId': instrumentId, 'values': ''})
                continue # "No model found for given Instrument"
            else:
                predictions, deltas = predictPriceForInstrument(instrumentId, spotPriceArr, pricerModel)
                values = np.concatenate((spotPriceArr, predictions, deltas), axis=1)
                df = pd.DataFrame(values, columns=['SpotPrice', 'Predicted Price', 'Deltas'])
                print(predictions)
                print(deltas)
                # response_json['instrumentId'] = instrumentId
                # response_json['values'] = df.to_dict(orient="records")
                response["data"].append({'instrumentId': instrumentId, 'values': df.to_dict(orient="records")})

    return jsonify(response)


def predictPriceForInstrument(instrumentId , spotPrice ,model):
    predictions, deltas = model.predict_values_and_derivs(spotPrice)
    return predictions , deltas

def generateTrainingDataAndTrainModel(instrumentId,seed,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    simulSeed = seed
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))

    xTrain, yTrain, dydxTrain = generator.trainingSet(size, seed)

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print('Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.prepare(size, True, weight_seed=simulSeed)
    print('Model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

    t0 = time.time()
    print('Model Training start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    regressor.train("differential training")
    print('Model Training end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
    t1 = time.time()

    training_time = t1 - t0
    print('Training time =', training_time)

    #instrumentModelMap.__setitem__(instrumentId, regressor)
    return xTrain, yTrain, dydxTrain, regressor

def getRawResponse():
    return {"data":[]}

def getRequestParam(request_data):
    instrumentId = request_data['ticker']
    strikePrice = float(request_data['strikeprice'])
    expiryInYears = float(request_data['expiry'])
    spotPrice = float(request_data['spotprice'])
    volatality = float(request_data['volatility'])

    return instrumentId , strikePrice, expiryInYears, spotPrice, volatality

def populateModelCache(instrumentId, model):
    if not instrumentModelMap.get(instrumentId):
        print("Adding Model in cache for ", instrumentId)
        instrumentModelMap.__setitem__(instrumentId, model)
    else:
        print("Model Already Exist in cache")

def modelExist(instrumentId):
    return True if instrumentModelMap.get(instrumentId) else False

if __name__ == '__main__':
    app.run(debug=True)
