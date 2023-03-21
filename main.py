
from BlackScholes import BlackScholes
import pandas as pd
import numpy as np

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    option_data = [['AAPL230915C00150000', 155.00 , 150.00 , 0.3539 , 0.5],['AAPL230915C00160000', 155.00 , 160.00 , 0.3266 , 0.5]]
    df = pd.DataFrame(option_data, columns=['InstrumentId', 'SpotPrice' , 'StrikePrice','ImpliedVol','Expiary'])
    print(df)

    size = 4
    gen = BlackScholes()
    for ind in df.index:
        spot = df['SpotPrice'][ind]
        strike = df['StrikePrice'][ind]
        implVol = df['ImpliedVol'][ind]

        gen.__init__(spot=spot , K=strike )

        xTrain, yTrain, dydxTrain = gen.trainingSet(m=size)
        result = np.concatenate(xTrain, yTrain, dydxTrain)
        print(result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
