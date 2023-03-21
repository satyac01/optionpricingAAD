import matplotlib.pyplot as plt

from BlackScholes import BlackScholes
from PricingModel import *
from datetime import datetime
import pandas as pd

instrumentModelMap = {}


def test(generator,
         sizes,
         nTest,
         simulSeed=None,
         testSeed=None,
         weightSeed=None,
         deltidx=0,
         instrumentId=None):
    # simulation
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSet(max(sizes), seed=simulSeed)

    comb_arr = np.concatenate((xTrain,yTrain,dydxTrain), axis = 1)

    xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num=nTest, seed=testSeed,spotvariation=0.65)
    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")

    predvalues = {}
    preddeltas = {}
    for size in sizes:
        print("\nsize %d" % size)
        print('Model prep start time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
        regressor.prepare(size, False, weight_seed=weightSeed)
        print('model prep end time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

        t0 = time.time()
        print('*** Started Training time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
        regressor.train("standard training")

        print('*** Training Finished , time = ', datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

        instrumentModelMap.__setitem__(instrumentId,regressor)

        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()

        regressor.prepare(size, True, weight_seed=weightSeed)

        t0 = time.time()
        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()

    return xAxis, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas


def graph(title,
          predictions,
          xAxis,
          xAxisName,
          yAxisName,
          targets,
          sizes,
          computeRmse=False,
          weights=None,
          imagePath=None):
    numRows = len(sizes)
    numCols = 2

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

    for i, size in enumerate(sizes):
        ax[i, 0].annotate("size %d" % size, xy=(0, 0.5),
                          xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                          xycoords=ax[i, 0].yaxis.label, textcoords='offset points',
                          ha='right', va='center')

    ax[0, 0].set_title("standard")
    ax[0, 1].set_title("differential")

    for i, size in enumerate(sizes):
        for j, regType, in enumerate(["standard", "differential"]):

            if computeRmse:
                errors = 100 * (predictions[(regType, size)] - targets)
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName

            ax[i, j].set_xlabel(t)
            ax[i, j].set_ylabel(yAxisName)

            ax[i, j].plot(xAxis * 100, predictions[(regType, size)] * 100, 'co', \
                          markersize=2, markerfacecolor='white', label="predicted")
            ax[i, j].plot(xAxis * 100, targets * 100, 'r.', markersize=0.5, label='targets')

            ax[i, j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    if not imagePath:
        plt.show()
    else :
        imageName = "BS_"+yAxisName+"_"+str(np.random.randint(0, 10000))
        #absolutePath = "C:\\Users\\sumit\\temp\\{fileName}".format(fileName = imageName
        absolutePath = imagePath+"\\"+imageName
        plt.savefig(absolutePath)


if __name__ == '__main__':
    # simulation set sizes to perform
    sizes = [1024,8192]
    # show delta?
    showDeltas = True
    # seed
    # simulSeed = 1234
    simulSeed = np.random.randint(0, 10000)
    print("using seed %d" % simulSeed)
    weightSeed = None

    # number of test scenarios
    nTest = 100

    # Option Instrument Data
    option_data = [['AAPL230915C00150000', 1.55, 1.5, 0.3266, 1.3]]#,['AAPL230915C00160000', 1.55, 1.5, 0.3266, 1.3]]
    df = pd.DataFrame(option_data, columns=['InstrumentId', 'SpotPrice', 'StrikePrice', 'ImpliedVol', 'Expiry'])
    print(df)

    generator = BlackScholes()
    for ind in df.index:
        instrument = df['InstrumentId'][ind]
        spot = df['SpotPrice'][ind]
        strike = df['StrikePrice'][ind]
        implVol = df['ImpliedVol'][ind]
        expiry = df['Expiry'][ind]

        generator.__init__(spot=spot, K=strike, vol=implVol, T2=expiry)

        xAxis, yTest, dydxTest, vegas, values, deltas = test(generator, sizes, nTest, simulSeed, None, weightSeed,instrumentId=instrument)

        path = "C:\\Users\\sumit\\temp"
        # show predicitions
        graph("Black & Scholes", values, xAxis, "", "values", yTest, sizes, True,imagePath=path)

        # show deltas
        if showDeltas:
            graph("Black & Scholes", deltas, xAxis, "", "deltas", dydxTest, sizes, True,imagePath=path)
